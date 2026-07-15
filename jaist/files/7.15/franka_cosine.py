#!/usr/bin/env python3
"""
franka.py - native Cartesian velocity + raised-cosine smoother.

Use with spacemouse.py generated together with this file.

This version is based on the test that worked:
    test_native_cartesian_velocity_ramp.py

Main idea:
    - Use Robot.start_cartesian_velocity_control()
    - Send CartesianVelocities, not CartesianPose
    - Never step velocity directly
    - Smooth every target change with a raised-cosine transition
"""

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from pylibfranka import (
    Robot,
    Gripper,
    GripperState,
    RealtimeConfig,
    ControllerMode,
    CartesianVelocities,
)


VelocityGetter = Callable[[], Tuple[float, float, float, float, float]]
# returns: vx, vy, vz, wz, stamp


@dataclass
class NativeVelocityConfig:
    max_linear_speed: float = 0.025
    max_yaw_speed: float = 0.040

    transition_time_up: float = 0.60
    transition_time_down: float = 0.25

    min_retarget_dt: float = 0.10
    linear_target_deadband: float = 0.0004
    yaw_target_deadband: float = 0.0008

    command_timeout: float = 0.35
    startup_zero_time: float = 0.8


class CosineVelocitySmoother:
    """
    Smoothly transitions a 4D velocity command [vx, vy, vz, wz]
    using:
        s = 0.5 - 0.5*cos(pi*t/T)
    """

    def __init__(
        self,
        transition_time_up=0.60,
        transition_time_down=0.25,
        min_retarget_dt=0.10,
        linear_target_deadband=0.0004,
        yaw_target_deadband=0.0008,
    ):
        self.output = np.zeros(4, dtype=float)
        self.start = self.output.copy()
        self.goal = self.output.copy()

        self.transition_start = time.monotonic()
        self.transition_duration = float(transition_time_up)

        self.transition_time_up = float(transition_time_up)
        self.transition_time_down = float(transition_time_down)
        self.min_retarget_dt = float(min_retarget_dt)

        self.linear_target_deadband = float(linear_target_deadband)
        self.yaw_target_deadband = float(yaw_target_deadband)

        self.last_retarget = 0.0

    def _is_zero(self, target):
        return float(np.linalg.norm(target[:3])) < 1e-10 and abs(float(target[3])) < 1e-10

    def _target_changed(self, target):
        if float(np.linalg.norm(target[:3] - self.goal[:3])) > self.linear_target_deadband:
            return True
        if abs(float(target[3] - self.goal[3])) > self.yaw_target_deadband:
            return True
        return False

    def set_target(self, target, force=False):
        target = np.array(target, dtype=float).reshape(4)
        now = time.monotonic()

        if not force:
            if not self._target_changed(target):
                return

            # Do not restart too often for nonzero targets.
            # Zero target is allowed immediately so release stops fast.
            if not self._is_zero(target):
                if now - self.last_retarget < self.min_retarget_dt:
                    return

        self.update()

        current_norm = float(np.linalg.norm(self.output))
        target_norm = float(np.linalg.norm(target))

        if target_norm < current_norm:
            duration = self.transition_time_down
        else:
            duration = self.transition_time_up

        self.start = self.output.copy()
        self.goal = target.copy()
        self.transition_start = now
        self.transition_duration = max(duration, 1e-3)
        self.last_retarget = now

    def update(self):
        now = time.monotonic()
        t = now - self.transition_start
        T = self.transition_duration

        if t >= T:
            self.output = self.goal.copy()
            return self.output.copy()

        if t <= 0.0:
            self.output = self.start.copy()
            return self.output.copy()

        u = t / T
        s = 0.5 - 0.5 * np.cos(np.pi * u)

        self.output = self.start + s * (self.goal - self.start)
        return self.output.copy()


class FrankaController:
    def __init__(
        self,
        ip: str = "172.16.0.2",
        use_gripper: bool = False,
        do_gripper_homing: bool = False,
        realtime: bool = False,
    ):
        self.ip = ip

        self.robot: Optional[Robot] = None
        self.gripper: Optional[Gripper] = None

        self.motion_lock = threading.Lock()
        self.state_lock = threading.Lock()

        self.stop_event = threading.Event()
        self.velocity_servo_running = threading.Event()
        self.velocity_thread: Optional[threading.Thread] = None

        self.latest_state = None
        self.latest_O_T_EE = None
        self.latest_q = None
        self.is_moving = False

        rt_config = RealtimeConfig.kEnforce if realtime else RealtimeConfig.kIgnore
        self.robot = Robot(ip, rt_config)

        self.set_default_collision_behavior()
        print(f"[FrankaController] Robot connected: {ip}")

        if use_gripper:
            try:
                self.gripper = Gripper(ip)
                print("[FrankaController] Gripper connected")

                if do_gripper_homing:
                    ok = self.gripper.homing()
                    print(f"[FrankaController] Gripper homing result: {ok}")

            except Exception as e:
                self.gripper = None
                print(f"[FrankaController] Gripper connection failed: {e}")

        self.update_idle_state()

    def set_default_collision_behavior(self):
        lower_torque = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
        upper_torque = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]

        lower_force = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]
        upper_force = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]

        self.robot.set_collision_behavior(
            lower_torque,
            upper_torque,
            lower_force,
            upper_force,
        )
        print("[FrankaController] Collision behavior set")

    def recover_errors(self):
        self.robot.automatic_error_recovery()
        print("[FrankaController] automatic_error_recovery called")

    def _cache_state(self, state):
        with self.state_lock:
            self.latest_state = state
            self.latest_O_T_EE = np.array(state.O_T_EE, dtype=float).copy()
            self.latest_q = np.array(state.q, dtype=float).copy()

    def update_idle_state(self):
        state = self.robot.read_once()
        self._cache_state(state)
        return state

    def get_joint(self):
        if self.is_moving:
            with self.state_lock:
                if self.latest_q is None:
                    raise RuntimeError("No cached joint state")
                return self.latest_q.copy()

        return np.array(self.update_idle_state().q, dtype=float).copy()

    def get_ee(self):
        if self.is_moving:
            with self.state_lock:
                if self.latest_O_T_EE is None:
                    raise RuntimeError("No cached EE state")
                return self.latest_O_T_EE.copy()

        return np.array(self.update_idle_state().O_T_EE, dtype=float).copy()

    def get_gripper(self) -> Optional[GripperState]:
        if self.gripper is None:
            return None
        return self.gripper.read_once()

    def force_stop(self):
        self.stop_event.set()
        self.velocity_servo_running.clear()

        try:
            self.robot.stop()
        except Exception:
            pass

        print("[FrankaController] Force stop requested")

    def start_ee_velocity_servo(
        self,
        velocity_getter: VelocityGetter,
        controller_mode=ControllerMode.JointImpedance,
        cfg: Optional[NativeVelocityConfig] = None,
        print_debug: bool = False,
        print_dt: float = 1.0,
    ):
        if self.velocity_thread is not None and self.velocity_thread.is_alive():
            raise RuntimeError("Velocity servo already running")

        cfg = cfg or NativeVelocityConfig()

        self.stop_event.clear()
        self.velocity_servo_running.set()

        self.velocity_thread = threading.Thread(
            target=self._velocity_loop,
            args=(velocity_getter, controller_mode, cfg, print_debug, print_dt),
            daemon=True,
        )
        self.velocity_thread.start()

        print("[FrankaController] Native Cartesian velocity servo started")

    # Compatibility aliases.
    def start_ee_velocity_thread(self, *args, **kwargs):
        return self.start_ee_velocity_servo(*args, **kwargs)

    def stop_ee_velocity_servo(self):
        self.stop_event.set()
        self.velocity_servo_running.clear()

        if self.velocity_thread is not None:
            self.velocity_thread.join(timeout=2.0)

        self.velocity_thread = None
        print("[FrankaController] Native Cartesian velocity servo stopped")

    def stop_ee_velocity_thread(self):
        return self.stop_ee_velocity_servo()

    def _velocity_loop(
        self,
        velocity_getter: VelocityGetter,
        controller_mode,
        cfg: NativeVelocityConfig,
        print_debug: bool,
        print_dt: float,
    ):
        with self.motion_lock:
            self.is_moving = True

            smoother = CosineVelocitySmoother(
                transition_time_up=cfg.transition_time_up,
                transition_time_down=cfg.transition_time_down,
                min_retarget_dt=cfg.min_retarget_dt,
                linear_target_deadband=cfg.linear_target_deadband,
                yaw_target_deadband=cfg.yaw_target_deadband,
            )

            start_time = time.time()
            last_print = start_time
            last_target = np.zeros(4, dtype=float)

            try:
                print("[FrankaController] Starting native Cartesian velocity control...")
                active = self.robot.start_cartesian_velocity_control(controller_mode)
                print("[FrankaController] Native Cartesian velocity control is active")
                print("[FrankaController] Raised-cosine velocity smoother is enabled")

                while self.velocity_servo_running.is_set() and not self.stop_event.is_set():
                    state, duration = active.readOnce()
                    self._cache_state(state)

                    now = time.time()

                    try:
                        vx, vy, vz, wz, stamp = velocity_getter()
                    except Exception as e:
                        print(f"[FrankaController] velocity_getter failed: {e}")
                        vx = vy = vz = wz = 0.0
                        stamp = 0.0

                    age = now - stamp if stamp > 0.0 else 999.0
                    startup_zero = (now - start_time) < cfg.startup_zero_time

                    if startup_zero or age > cfg.command_timeout:
                        target = np.zeros(4, dtype=float)
                    else:
                        target = np.array([vx, vy, vz, wz], dtype=float)
                        target[:3] = np.clip(target[:3], -cfg.max_linear_speed, +cfg.max_linear_speed)
                        target[3] = float(np.clip(target[3], -cfg.max_yaw_speed, +cfg.max_yaw_speed))

                    force_zero = bool(np.linalg.norm(target) < 1e-12 and np.linalg.norm(last_target) > 1e-12)
                    smoother.set_target(target, force=force_zero)
                    last_target = target.copy()

                    out = smoother.update()
                    out[np.abs(out) < 1e-6] = 0.0

                    vel6 = [
                        float(out[0]),
                        float(out[1]),
                        float(out[2]),
                        0.0,
                        0.0,
                        float(out[3]),
                    ]

                    active.writeOnce(CartesianVelocities(vel6))

                    if print_debug and now - last_print >= print_dt:
                        last_print = now
                        print(
                            "[FrankaController] "
                            f"age={age:.3f}, "
                            f"target=[{target[0]:+.4f}, {target[1]:+.4f}, {target[2]:+.4f}, {target[3]:+.4f}], "
                            f"send=[{out[0]:+.4f}, {out[1]:+.4f}, {out[2]:+.4f}, {out[3]:+.4f}]"
                        )

                # Smooth stop before finishing.
                smoother.set_target(np.zeros(4, dtype=float), force=True)

                stop_t0 = time.monotonic()
                while time.monotonic() - stop_t0 < cfg.transition_time_down:
                    state, duration = active.readOnce()
                    self._cache_state(state)

                    out = smoother.update()
                    out[np.abs(out) < 1e-6] = 0.0

                    vel6 = [
                        float(out[0]),
                        float(out[1]),
                        float(out[2]),
                        0.0,
                        0.0,
                        float(out[3]),
                    ]

                    active.writeOnce(CartesianVelocities(vel6))

                    if float(np.linalg.norm(out)) < 1e-5:
                        break

                finish = CartesianVelocities([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                finish.motion_finished = True
                active.writeOnce(finish)

            except Exception as e:
                print(f"[FrankaController] Native velocity loop exception: {e}")
                self.force_stop()
                raise

            finally:
                self.is_moving = False
                self.velocity_servo_running.clear()
                self.stop_event.clear()

                try:
                    self.update_idle_state()
                except Exception:
                    pass
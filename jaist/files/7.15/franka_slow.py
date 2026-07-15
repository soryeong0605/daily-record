#!/usr/bin/env python3
"""
franka.py

Clean Franka controller for SpaceMouse teleop.

Design:
    - Franka control loop owns the timing.
    - The control loop reads velocity command directly from a getter.
    - The command is filtered/ramped INSIDE the Franka control loop.
    - No external thread needs to call set_ee_velocity at 50/100 Hz.

This is intended to be used by spacemouse.py:

    from franka import FrankaController

Required pylibfranka API:
    Robot
    RealtimeConfig
    ControllerMode
    CartesianPose
"""

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from pylibfranka import (
    Robot,
    Gripper,
    GripperState,
    RealtimeConfig,
    ControllerMode,
    CartesianPose,
    JointPositions,
)


VelocityGetter = Callable[[], Tuple[float, float, float, float, float]]
# returns: vx, vy, vz, wz, stamp


@dataclass
class MotionConfig:
    max_cart_distance: float = 1.0


class FrankaController:
    def __init__(
        self,
        ip: str = "172.16.0.2",
        use_gripper: bool = False,
        do_gripper_homing: bool = False,
        realtime: bool = False,
        motion_cfg: Optional[MotionConfig] = None,
    ):
        self.ip = ip
        self.motion_cfg = motion_cfg or MotionConfig()

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

        self.active_control = None
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

    # ============================================================
    # Safety / state
    # ============================================================

    def set_default_collision_behavior(self):
        if self.robot is None:
            raise RuntimeError("Robot is not connected")

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
        if self.robot is None:
            raise RuntimeError("Robot is not connected")

        self.robot.automatic_error_recovery()
        print("[FrankaController] automatic_error_recovery called")

    def _cache_state(self, robot_state):
        with self.state_lock:
            self.latest_state = robot_state
            self.latest_O_T_EE = np.array(robot_state.O_T_EE, dtype=float).copy()
            self.latest_q = np.array(robot_state.q, dtype=float).copy()

    def update_idle_state(self):
        if self.robot is None:
            raise RuntimeError("Robot is not connected")

        state = self.robot.read_once()
        self._cache_state(state)
        return state

    def get_ee(self) -> np.ndarray:
        if self.is_moving:
            with self.state_lock:
                if self.latest_O_T_EE is None:
                    raise RuntimeError("No cached EE state yet")
                return self.latest_O_T_EE.copy()

        state = self.update_idle_state()
        return np.array(state.O_T_EE, dtype=float).copy()

    def get_joint(self) -> np.ndarray:
        if self.is_moving:
            with self.state_lock:
                if self.latest_q is None:
                    raise RuntimeError("No cached joint state yet")
                return self.latest_q.copy()

        state = self.update_idle_state()
        return np.array(state.q, dtype=float).copy()

    def get_gripper(self) -> Optional[GripperState]:
        if self.gripper is None:
            return None
        return self.gripper.read_once()

    def force_stop(self):
        self.stop_event.set()
        self.velocity_servo_running.clear()

        if self.gripper is not None:
            try:
                self.gripper.stop()
            except Exception:
                pass

        if self.robot is not None:
            try:
                self.robot.stop()
            except Exception:
                pass

        print("[FrankaController] Force stop requested")

    # ============================================================
    # Velocity servo
    # ============================================================

    def start_ee_velocity_servo(
        self,
        velocity_getter: VelocityGetter,
        controller_mode=ControllerMode.JointImpedance,
        max_linear_speed: float = 0.010,
        max_yaw_speed: float = 0.020,
        max_linear_acc: float = 0.006,
        max_yaw_acc: float = 0.015,
        target_lpf_alpha: float = 0.10,
        command_timeout: float = 0.40,
        startup_zero_time: float = 1.0,
        max_distance_from_start: float = 0.80,
        yaw_in_base_frame: bool = False,
        print_debug: bool = False,
        print_dt: float = 1.0,
    ):
        """
        Start Cartesian velocity servo.

        velocity_getter must be very light and return:
            vx, vy, vz, wz, stamp

        Units:
            vx,vy,vz: m/s
            wz: rad/s

        The getter target is NOT sent directly.
        It is clipped, low-pass filtered, and ramp-limited inside this loop.
        """

        if self.velocity_thread is not None and self.velocity_thread.is_alive():
            raise RuntimeError("EE velocity servo is already running")

        self.stop_event.clear()
        self.velocity_servo_running.set()

        self.velocity_thread = threading.Thread(
            target=self._ee_velocity_loop,
            args=(
                velocity_getter,
                controller_mode,
                max_linear_speed,
                max_yaw_speed,
                max_linear_acc,
                max_yaw_acc,
                target_lpf_alpha,
                command_timeout,
                startup_zero_time,
                max_distance_from_start,
                yaw_in_base_frame,
                print_debug,
                print_dt,
            ),
            daemon=True,
        )

        self.velocity_thread.start()
        print("[FrankaController] EE velocity servo thread started")

    # Backward-compatible name.
    def start_ee_velocity_thread(self, *args, **kwargs):
        return self.start_ee_velocity_servo(*args, **kwargs)

    def stop_ee_velocity_servo(self):
        self.stop_event.set()
        self.velocity_servo_running.clear()

        if self.velocity_thread is not None:
            self.velocity_thread.join(timeout=2.0)

        self.velocity_thread = None
        print("[FrankaController] EE velocity servo thread stopped")

    # Backward-compatible name.
    def stop_ee_velocity_thread(self):
        return self.stop_ee_velocity_servo()

    def _ee_velocity_loop(
        self,
        velocity_getter: VelocityGetter,
        controller_mode,
        max_linear_speed: float,
        max_yaw_speed: float,
        max_linear_acc: float,
        max_yaw_acc: float,
        target_lpf_alpha: float,
        command_timeout: float,
        startup_zero_time: float,
        max_distance_from_start: float,
        yaw_in_base_frame: bool,
        print_debug: bool,
        print_dt: float,
    ):
        if self.robot is None:
            raise RuntimeError("Robot is not connected")

        with self.motion_lock:
            self.is_moving = True
            self.active_control = None

            target_v_f = np.zeros(3, dtype=float)
            target_wz_f = 0.0

            send_v = np.zeros(3, dtype=float)
            send_wz = 0.0

            start_wall = time.time()
            last_print = start_wall

            try:
                print("[FrankaController] Starting Cartesian pose control...")
                active = self.robot.start_cartesian_pose_control(controller_mode)
                self.active_control = active

                state, duration = active.readOnce()
                self._cache_state(state)

                desired_T = self._pose_to_matrix(state.O_T_EE)
                start_pos = desired_T[:3, 3].copy()

                # Send current pose once.
                active.writeOnce(CartesianPose(self._matrix_to_pose(desired_T)))

                print("[FrankaController] Cartesian pose control is active")
                print("[FrankaController] Internal velocity smoothing is enabled")

                while self.velocity_servo_running.is_set() and not self.stop_event.is_set():
                    state, duration = active.readOnce()
                    self._cache_state(state)

                    dt = float(duration.to_sec())

                    # Keep integration stable if duration is weird.
                    if dt <= 0.0 or dt > 0.004:
                        dt = 0.001

                    now = time.time()

                    try:
                        vx, vy, vz, wz, stamp = velocity_getter()
                    except Exception as e:
                        print(f"[FrankaController] velocity_getter failed: {e}")
                        vx = vy = vz = wz = 0.0
                        stamp = 0.0

                    age = now - stamp if stamp > 0.0 else 999.0
                    startup_zero = (now - start_wall) < startup_zero_time

                    if startup_zero or age > command_timeout:
                        target_v = np.zeros(3, dtype=float)
                        target_wz = 0.0
                    else:
                        target_v = np.array([vx, vy, vz], dtype=float)
                        target_v = np.clip(target_v, -max_linear_speed, +max_linear_speed)

                        target_wz = float(np.clip(wz, -max_yaw_speed, +max_yaw_speed))

                    # Low-pass filter target.
                    a = float(np.clip(target_lpf_alpha, 0.0, 1.0))

                    target_v_f = a * target_v + (1.0 - a) * target_v_f
                    target_wz_f = a * target_wz + (1.0 - a) * target_wz_f

                    # Ramp actual sent velocity.
                    send_v = self._ramp_vector(
                        current=send_v,
                        target=target_v_f,
                        max_delta=max_linear_acc * dt,
                    )

                    send_wz = self._ramp_scalar(
                        current=send_wz,
                        target=target_wz_f,
                        max_delta=max_yaw_acc * dt,
                    )

                    send_v[np.abs(send_v) < 1e-6] = 0.0
                    if abs(send_wz) < 1e-6:
                        send_wz = 0.0

                    # Integrate position, but do not stop the servo at distance limit.
                    # Instead clamp to a sphere so user can move back.
                    proposed_pos = desired_T[:3, 3] + send_v * dt
                    offset = proposed_pos - start_pos
                    dist = float(np.linalg.norm(offset))

                    if dist > max_distance_from_start:
                        if dist > 1e-12:
                            proposed_pos = start_pos + offset / dist * max_distance_from_start

                        # Do not keep pushing against the boundary.
                        send_v[:] = 0.0

                    desired_T[:3, 3] = proposed_pos

                    # Integrate yaw.
                    if abs(send_wz) > 1e-12:
                        dRz = self._rotz(send_wz * dt)

                        if yaw_in_base_frame:
                            desired_T[:3, :3] = dRz @ desired_T[:3, :3]
                        else:
                            desired_T[:3, :3] = desired_T[:3, :3] @ dRz

                    active.writeOnce(CartesianPose(self._matrix_to_pose(desired_T)))

                    if print_debug and (now - last_print) >= print_dt:
                        last_print = now
                        xyz = desired_T[:3, 3]
                        print(
                            "[FrankaController] "
                            f"age={age:.3f}, "
                            f"target=[{target_v[0]:+.4f}, {target_v[1]:+.4f}, {target_v[2]:+.4f}, {target_wz:+.4f}], "
                            f"send=[{send_v[0]:+.4f}, {send_v[1]:+.4f}, {send_v[2]:+.4f}, {send_wz:+.4f}], "
                            f"xyz=[{xyz[0]:+.4f}, {xyz[1]:+.4f}, {xyz[2]:+.4f}]"
                        )

                # Finish gracefully.
                finish_cmd = CartesianPose(self._matrix_to_pose(desired_T))
                finish_cmd.motion_finished = True
                active.writeOnce(finish_cmd)

            except Exception as e:
                print(f"[FrankaController] EE velocity loop exception: {e}")
                self.force_stop()
                raise

            finally:
                self.active_control = None
                self.is_moving = False
                self.velocity_servo_running.clear()
                self.stop_event.clear()

                try:
                    self.update_idle_state()
                except Exception:
                    pass

    # ============================================================
    # Optional simple position motions
    # ============================================================

    def move_ee(
        self,
        target_O_T_EE: Sequence[float],
        controller_mode=ControllerMode.JointImpedance,
        duration: float = 3.0,
    ):
        """
        Very simple linear interpolation in Cartesian pose.
        Kept only for basic compatibility.
        """
        target = np.array(target_O_T_EE, dtype=float).reshape(16)

        with self.motion_lock:
            self.is_moving = True
            self.stop_event.clear()

            try:
                active = self.robot.start_cartesian_pose_control(controller_mode)
                self.active_control = active

                state, _ = active.readOnce()
                self._cache_state(state)

                start = np.array(state.O_T_EE, dtype=float).copy()
                active.writeOnce(CartesianPose(start))

                t0 = time.time()

                while True:
                    state, dt_robot = active.readOnce()
                    self._cache_state(state)

                    t = time.time() - t0
                    alpha = float(np.clip(t / duration, 0.0, 1.0))
                    s = alpha * alpha * (3.0 - 2.0 * alpha)

                    pose = (1.0 - s) * start + s * target
                    cmd = CartesianPose(pose)

                    if alpha >= 1.0 or self.stop_event.is_set():
                        cmd.motion_finished = True
                        active.writeOnce(cmd)
                        break

                    active.writeOnce(cmd)

            finally:
                self.active_control = None
                self.is_moving = False
                self.stop_event.clear()
                self.update_idle_state()

    # ============================================================
    # Math helpers
    # ============================================================

    @staticmethod
    def _pose_to_matrix(pose16: Sequence[float]) -> np.ndarray:
        return np.array(pose16, dtype=float).reshape(4, 4, order="F").copy()

    @staticmethod
    def _matrix_to_pose(T: np.ndarray) -> np.ndarray:
        return np.array(T, dtype=float).reshape(16, order="F").copy()

    @staticmethod
    def _rotz(theta: float) -> np.ndarray:
        c = float(np.cos(theta))
        s = float(np.sin(theta))

        return np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)

    @staticmethod
    def _ramp_scalar(current: float, target: float, max_delta: float) -> float:
        delta = target - current

        if abs(delta) <= max_delta:
            return target

        return current + float(np.sign(delta)) * max_delta

    @staticmethod
    def _ramp_vector(current: np.ndarray, target: np.ndarray, max_delta: float) -> np.ndarray:
        delta = target - current
        norm = float(np.linalg.norm(delta))

        if norm <= max_delta or norm < 1e-12:
            return target.copy()

        return current + delta / norm * max_delta
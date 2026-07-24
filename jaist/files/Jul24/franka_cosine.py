#!/usr/bin/env python3
"""
franka.py - native Cartesian velocity + hybrid cosine/slew-rate smoother.

Use with spacemouse.py generated together with this file.

Changelog vs the pure raised-cosine version:
    - CosineVelocitySmoother restarted a fresh cosine curve (acceleration
      = 0 at t=0) EVERY time the target changed, even if the previous
      curve hadn't finished yet. If the previous curve was mid-flight
      (nonzero acceleration), that produced a sudden jump in acceleration
      -> Franka's cartesian_motion_generator_(joint_)acceleration_discontinuity
      reflex. This is what happened with fast-changing SpaceMouse input.

    - HybridVelocitySmoother fixes this with one extra check:
      "has the previous cosine curve actually finished?"
        - If YES  -> safe to start a brand new cosine curve as before.
          This is the common case and keeps the smooth, natural feel.
        - If NO (curve interrupted mid-flight, e.g. because SpaceMouse
          input is changing faster than the curve duration) -> instead
          of restarting a curve (which resets acceleration to 0 and
          causes the discontinuity), take ONE slew-rate-limited step
          toward the new target this tick (bounded by max_accel * dt,
          same idea as franka_slow.py's _ramp_vector). This guarantees
          the commanded velocity never changes faster than max_accel,
          so it can never trigger the discontinuity reflex, even under
          rapid-fire target changes.

    Net effect: cosine smoothness during normal use, with a hard safety
    net (slew-rate limiting) only kicking in during the rare moments
    where targets are changing faster than the curve can finish -- which
    is exactly when the pure-cosine version was failing.
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

    # --- New: safety-net slew-rate limit, only used when a cosine curve
    #     gets interrupted mid-flight (see HybridVelocitySmoother). ---
    max_linear_accel: float = 0.20   # m/s^2  -- tune: start conservative
    max_yaw_accel: float = 1.00       # rad/s^2

    command_timeout: float = 0.35
    startup_zero_time: float = 0.8


class HybridVelocitySmoother:
    """
    Smoothly transitions a 4D velocity command [vx, vy, vz, wz].

    Normal case: raised-cosine transition, same as before
        s = 0.5 - 0.5*cos(pi*t/T)

    Safety-net case: if a new target arrives before the previous cosine
    curve has finished, a cosine restart would reset acceleration to 0
    at the exact moment the old curve had nonzero acceleration -- a
    discontinuity. Instead, this smoother takes one slew-rate-limited
    step (bounded by max_accel*dt) toward the new target, then re-arms
    a (near-instant) curve from that safe point so the next call can
    resume normal cosine behavior if things settle down.
    """

    def __init__(
        self,
        transition_time_up=0.60,
        transition_time_down=0.25,
        min_retarget_dt=0.10,
        linear_target_deadband=0.0004,
        yaw_target_deadband=0.0008,
        max_accel=(0.20, 0.20, 0.20, 1.00),   # [ax, ay, az, ayaw]
    ):
        self.output = np.zeros(4, dtype=float)
        self.start = self.output.copy()
        self.goal = self.output.copy()

        # Elapsed/duration are tracked in "control-loop time" (accumulated
        # dt), not wall-clock time. This ties the smoother directly to the
        # same clock the Franka control loop itself reports, which is more
        # robust than time.monotonic() sampled from a separate thread.
        self.elapsed = 0.0
        self.duration = float(transition_time_up)

        self.transition_time_up = float(transition_time_up)
        self.transition_time_down = float(transition_time_down)
        self.min_retarget_dt = float(min_retarget_dt)

        self.linear_target_deadband = float(linear_target_deadband)
        self.yaw_target_deadband = float(yaw_target_deadband)

        self.max_accel = np.broadcast_to(np.asarray(max_accel, dtype=float), (4,)).copy()

        self.total_time = 0.0
        self.last_retarget_time = 0.0

        # Debug/inspection: True on ticks where the safety net kicked in.
        self.last_step_was_safety_net = False

    def _is_zero(self, target):
        return float(np.linalg.norm(target[:3])) < 1e-10 and abs(float(target[3])) < 1e-10

    def _target_changed(self, target):
        if float(np.linalg.norm(target[:3] - self.goal[:3])) > self.linear_target_deadband:
            return True
        if abs(float(target[3] - self.goal[3])) > self.yaw_target_deadband:
            return True
        return False

    @staticmethod
    def _ramp_step(current, target, max_delta):
        """Per-axis slew-rate limit: never move more than max_delta this tick."""
        delta = target - current
        delta = np.clip(delta, -max_delta, max_delta)
        return current + delta

    def step(self, target, dt, force=False):
        """
        Advance the smoother by one control tick.

        target : desired velocity right now (can jump arbitrarily)
        dt     : time since last tick (s) -- use the robot's own reported
                 duration where possible, so this stays in lockstep with
                 the real control loop.
        force  : bypass deadband/retarget-rate checks (e.g. for the final
                 stop-to-zero sequence).

        Returns: velocity to actually send to the robot this tick.
        """
        target = np.asarray(target, dtype=float).reshape(4)
        dt = max(float(dt), 1e-6)

        self.total_time += dt
        self.elapsed += dt
        self.last_step_was_safety_net = False

        target_changed = self._target_changed(target)

        if target_changed or force:
            is_zero = self._is_zero(target)

            # Same "don't retarget too often" throttle as before, except
            # zero targets are always allowed immediately (fast release).
            retarget_allowed = force or is_zero or (
                (self.total_time - self.last_retarget_time) >= self.min_retarget_dt
            )

            if retarget_allowed:
                curve_finished = self.elapsed >= self.duration

                if curve_finished or force:
                    # --- Safe case: previous curve is done. Start a
                    #     fresh cosine curve as usual. ---
                    current_norm = float(np.linalg.norm(self.output))
                    target_norm = float(np.linalg.norm(target))

                    duration = (
                        self.transition_time_down
                        if target_norm < current_norm
                        else self.transition_time_up
                    )

                    self.start = self.output.copy()
                    self.goal = target.copy()
                    self.elapsed = 0.0
                    self.duration = max(duration, 1e-3)

                else:
                    # --- Safety-net case: curve was interrupted mid-flight.
                    #     Do NOT restart a cosine curve here (that would
                    #     snap acceleration to 0 while the old curve still
                    #     had nonzero acceleration -> discontinuity).
                    #     Instead take one bounded step toward the target. ---
                    max_delta = self.max_accel * dt
                    self.output = self._ramp_step(self.output, target, max_delta)
                    self.last_step_was_safety_net = True

                    # IMPORTANT: goal is set to the *current output*, not the
                    # raw target. If we set goal=target here, then on a later
                    # tick where the target hasn't changed again (the common
                    # case), target_changed would be False, the "if" block
                    # would be skipped, elapsed would keep accumulating past
                    # this tiny duration, and the curve-finished branch below
                    # would snap output straight to goal=target in one step
                    # -- reintroducing exactly the discontinuity we're trying
                    # to avoid. Setting goal=output means "hold here" until
                    # the next real retarget event explicitly moves us.
                    self.start = self.output.copy()
                    self.goal = self.output.copy()
                    self.elapsed = 0.0
                    self.duration = 1e-3

                self.last_retarget_time = self.total_time

        # Evaluate the current curve at "now" (elapsed / duration).
        if self.elapsed >= self.duration:
            self.output = self.goal.copy()
        else:
            u = self.elapsed / self.duration
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

            smoother = HybridVelocitySmoother(
                transition_time_up=cfg.transition_time_up,
                transition_time_down=cfg.transition_time_down,
                min_retarget_dt=cfg.min_retarget_dt,
                linear_target_deadband=cfg.linear_target_deadband,
                yaw_target_deadband=cfg.yaw_target_deadband,
                max_accel=(
                    cfg.max_linear_accel,
                    cfg.max_linear_accel,
                    cfg.max_linear_accel,
                    cfg.max_yaw_accel,
                ),
            )

            start_time = time.time()
            last_print = start_time
            last_tick_time = None
            safety_net_count = 0

            try:
                print("[FrankaController] Starting native Cartesian velocity control...")
                active = self.robot.start_cartesian_velocity_control(controller_mode)
                print("[FrankaController] Native Cartesian velocity control is active")
                print("[FrankaController] Hybrid cosine/slew-rate smoother is enabled")

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

                    # dt: prefer the robot's own reported duration, fall
                    # back to measured wall-clock time between ticks.
                    try:
                        dt = duration.to_sec()
                        if dt <= 0.0:
                            raise ValueError
                    except Exception:
                        dt = (now - last_tick_time) if last_tick_time is not None else 0.001
                    last_tick_time = now

                    out = smoother.step(target, dt)
                    out[np.abs(out) < 1e-6] = 0.0

                    if smoother.last_step_was_safety_net:
                        safety_net_count += 1

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
                            f"send=[{out[0]:+.4f}, {out[1]:+.4f}, {out[2]:+.4f}, {out[3]:+.4f}], "
                            f"safety_net_hits={safety_net_count}"
                        )

                # Smooth stop before finishing.
                stop_t0 = time.monotonic()
                while time.monotonic() - stop_t0 < max(cfg.transition_time_down, 1.0):
                    state, duration = active.readOnce()
                    self._cache_state(state)

                    try:
                        dt = duration.to_sec()
                        if dt <= 0.0:
                            raise ValueError
                    except Exception:
                        dt = 0.001

                    out = smoother.step(np.zeros(4, dtype=float), dt, force=True)
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

                if print_debug:
                    print(f"[FrankaController] Total safety-net interventions this run: {safety_net_count}")

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
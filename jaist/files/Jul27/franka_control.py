#!/usr/bin/env python3

import argparse
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import scipy.spatial.transform as st

from pylibfranka import (
    Robot,
    Gripper,
    GripperState,
    RealtimeConfig,
    ControllerMode,
    CartesianPose,
    JointPositions,
    CartesianVelocities,
)


@dataclass
class MotionConfig:
    max_cart_vel: float = 0.2     # m/s, conservative default
    max_cart_acc: float = 0.20     # m/s^2
    max_joint_vel: float = 0.20    # rad/s
    max_joint_acc: float = 0.50    # rad/s^2
    max_cart_distance: float = 0.80  # m, reject too-large Cartesian command
    max_ang_vel: float = 0.5       # rad/s, angular speed cap used by the servo loops
    max_ang_acc: float = 1.0       # rad/s^2, angular acceleration cap used by the servo loops


class FrankaController:
    """
    Franka Panda control via pylibfranka.

    Public API mirrors UR5eRobot (learning/hardware/ur5e_rtde.py) and
    FrankaRobot (learning/hardware/franka_robot.py, franky-based):
        get_tcp_pose()       -> np.ndarray (6,)  [x,y,z,rx,ry,rz] m/rad (rotvec)
        get_joint_angles()   -> np.ndarray (7,)  joint angles rad
        move_tcp_pose()      -> blocking (or async) point-to-point Cartesian move
        move_joints()        -> blocking (or async) point-to-point joint move
        stop()                -> stop motion/servo, keep the connection alive
        recover()             -> automatic error recovery
        disconnect()          -> stop and release

    Plus Franka-specific extras (no UR5e/FrankaRobot equivalent):
        get_ee_wrench()             -> np.ndarray (6,) estimated EE force/torque
        get_joint_external_torques() -> np.ndarray (7,) estimated joint torques
        set_ee_velocity()            -> real-time Cartesian velocity control
                                         (e.g. spacemouse teleop)
    """

    def __init__(
        self,
        robot_ip: str = "172.16.0.2",
        frequency: float = 30.0,
        use_gripper: bool = True,
        do_gripper_homing: bool = False,
        realtime: bool = False,
        motion_cfg: Optional[MotionConfig] = None,
    ):
        """
        Parameters
        ----------
        robot_ip    : Franka FCI IP (default 172.16.0.2)
        frequency   : Advisory only, for signature parity with UR5eRobot/FrankaRobot.
                      libfranka supplies its own real per-cycle dt via readOnce(),
                      so this is not enforced anywhere internally.
        """
        self.robot_ip = robot_ip
        self.frequency = frequency
        self.motion_cfg = motion_cfg or MotionConfig()

        self.motion_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.servo_lock = threading.Lock()
        self.stop_event = threading.Event()

        self.robot: Optional[Robot] = None
        self.gripper: Optional[Gripper] = None
        self.active_control = None
        self.is_moving = False

        self._servo_thread: Optional[threading.Thread] = None
        self._servo_kind: Optional[str] = None   # "joint" or "cartesian_velocity"
        self._async_thread: Optional[threading.Thread] = None
        self.servo_target = None

        # Ramped per-joint speed state for _joint_servo_loop; reset to zero
        # whenever a joint servo session starts.
        self._servo_joint_vel = np.zeros(7)

        # Velocity-control (set_ee_velocity) state: reset whenever that
        # servo session starts. servo_vel_target/last_cmd_time are shared
        # with the caller under servo_lock; servo_vel_current is only
        # touched by the background loop.
        self.servo_vel_target = np.zeros(6)
        self.servo_vel_last_cmd_time = 0.0
        self._servo_vel_current = np.zeros(6)
        self._servo_vel_timeout = 0.5

        self.latest_state = None
        self.latest_O_T_EE = None
        self.latest_q = None
        self.latest_wall_time = None

        rt_config = RealtimeConfig.kEnforce if realtime else RealtimeConfig.kIgnore
        self.robot = Robot(robot_ip, rt_config)
        self.set_default_collision_behavior()

        print(f"[FrankaController] Robot connected: {robot_ip}")

        if use_gripper:
            try:
                self.gripper = Gripper(robot_ip)
                print("[FrankaController] Gripper connected")

                if do_gripper_homing:
                    print("[FrankaController] Homing gripper...")
                    ok = self.gripper.homing()
                    print(f"[FrankaController] Gripper homing result: {ok}")
                else:
                    print("[FrankaController] Gripper homing skipped")

            except Exception as e:
                self.gripper = None
                print(f"[FrankaController] No gripper or gripper connection failed: {e}")

        self.update_idle_state()

    # -------------------------
    # Safety / configuration
    # -------------------------

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

    def set_load(self, mass: float, com_xyz: Sequence[float], inertia_3x3_colmajor: Sequence[float]):
        if self.robot is None:
            raise RuntimeError("Robot is not connected")
        self.robot.set_load(mass, list(com_xyz), list(inertia_3x3_colmajor))

    def recover(self):
        """Recover from robot errors (e.g. after a collision)."""
        if self.robot is None:
            raise RuntimeError("Robot is not connected")
        self.robot.automatic_error_recovery()
        print("[FrankaController] Automatic error recovery called")

    def force_stop(self):
        self.stop_event.set()

        if self.gripper is not None:
            try:
                self.gripper.stop()
            except Exception as e:
                print(f"[FrankaController] Gripper stop failed: {e}")

        if self.robot is not None:
            try:
                self.robot.stop()
            except Exception as e:
                print(f"[FrankaController] Robot stop failed: {e}")

        print("[FrankaController] Force stop requested")

    # -------------------------
    # State
    # -------------------------

    def _cache_state(self, robot_state):
        with self.state_lock:
            self.latest_state = robot_state
            self.latest_O_T_EE = np.array(robot_state.O_T_EE, dtype=float).copy()
            self.latest_q = np.array(robot_state.q, dtype=float).copy()
            self.latest_wall_time = time.time()

    def update_idle_state(self):
        if self.robot is None:
            raise RuntimeError("Robot is not connected")

        state = self.robot.read_once()
        self._cache_state(state)
        return state

    def _get_O_T_EE(self) -> np.ndarray:
        """
        Return current EE pose as 16 values in column-major order.

        If robot is moving, return cached state updated by the active control loop.
        If robot is idle, read directly from robot.
        """
        if self.is_moving:
            with self.state_lock:
                if self.latest_O_T_EE is None:
                    raise RuntimeError("No cached state yet")
                return self.latest_O_T_EE.copy()

        state = self.update_idle_state()
        return np.array(state.O_T_EE, dtype=float).copy()

    def _get_q(self) -> np.ndarray:
        if self.is_moving:
            with self.state_lock:
                if self.latest_q is None:
                    raise RuntimeError("No cached joint state yet")
                return self.latest_q.copy()

        state = self.update_idle_state()
        return np.array(state.q, dtype=float).copy()

    def _get_robot_state(self):
        """Return the latest cached RobotState if moving, else read fresh."""
        if self.is_moving:
            with self.state_lock:
                if self.latest_state is None:
                    raise RuntimeError("No cached state yet")
                return self.latest_state
        return self.update_idle_state()

    # -------------------------
    # Gripper
    # -------------------------

    def get_gripper(self) -> Optional[GripperState]:
        if self.gripper is None:
            return None
        return self.gripper.read_once()

    def gripper_move(self, width: float = 0.08, speed: float = 0.05) -> bool:
        if self.gripper is None:
            raise RuntimeError("Gripper is not available")
        return self.gripper.move(width, speed)

    def gripper_grasp(
        self,
        width: float = 0.02,
        speed: float = 0.03,
        force: float = 30.0,
        epsilon_inner: float = 0.005,
        epsilon_outer: float = 0.005,
    ) -> bool:
        if self.gripper is None:
            raise RuntimeError("Gripper is not available")
        return self.gripper.grasp(width, speed, force, epsilon_inner, epsilon_outer)

    # -------------------------
    # Public API (matches UR5eRobot / FrankaRobot)
    # -------------------------

    def get_tcp_pose(self) -> np.ndarray:
        """Return current TCP pose as (6,) [x, y, z, rx, ry, rz] m/rad (rotation vector)."""
        return self._O_T_EE_to_pose6(self._get_O_T_EE())

    def get_joint_angles(self) -> np.ndarray:
        """Return current joint angles as (7,) rad. Franka has 7 joints (UR5e has 6)."""
        return self._get_q().astype(np.float32)

    def get_ee_wrench(self, frame: str = "base") -> np.ndarray:
        """
        Return Franka's built-in estimated external wrench at the EE.

        This comes from the arm's joint-torque sensing (no external F/T
        sensor needed) -- libfranka's O_F_ext_hat_K / K_F_ext_hat_K.

        Parameters
        ----------
        frame : "base" -> wrench expressed in the robot base frame (O_F_ext_hat_K)
                "ee"   -> wrench expressed in the stiffness/EE frame (K_F_ext_hat_K)

        Returns
        -------
        np.ndarray (6,) [Fx, Fy, Fz, Tx, Ty, Tz] in N / N*m.
        """
        state = self._get_robot_state()
        field = state.O_F_ext_hat_K if frame == "base" else state.K_F_ext_hat_K
        return np.asarray(field, dtype=np.float32).reshape(6)

    def get_joint_external_torques(self) -> np.ndarray:
        """Return filtered estimated external joint torques (7,) in N*m."""
        return np.asarray(self._get_robot_state().tau_ext_hat_filtered, dtype=np.float32).reshape(7)

    def move_tcp_pose(
        self,
        target_pose,
        velocity: float = 0.5,
        acceleration: float = 1.0,
        asynchronous: bool = False,
    ):
        """
        Blocking (or async) point-to-point move to a target TCP pose.

        Parameters
        ----------
        target_pose  : (6,) [x, y, z, rx, ry, rz] m/rad (rotation vector)
        velocity     : Max Cartesian speed (m/s) for the trapezoidal profile.
        acceleration : Max Cartesian acceleration (m/s^2).
        asynchronous : If True, run the move in a background thread and return
                       immediately (the thread object is returned for optional
                       joining).
        """
        target16 = self._pose6_to_O_T_EE(target_pose)
        if asynchronous:
            thread = threading.Thread(
                target=self._move_ee,
                kwargs=dict(target_O_T_EE=target16, max_vel=velocity, max_acc=acceleration),
                daemon=True,
            )
            self._async_thread = thread
            thread.start()
            return thread
        return self._move_ee(target16, max_vel=velocity, max_acc=acceleration)

    def set_ee_velocity(
        self,
        linear_velocity: Sequence[float],
        angular_velocity: Optional[Sequence[float]] = None,
        max_vel: Optional[float] = None,
        max_ang_vel: Optional[float] = None,
    ):
        """
        Command a Cartesian EE velocity, base frame (e.g. from a spacemouse).
        No UR5e/FrankaRobot equivalent -- this is a Franka-only extra.

        Lazily starts a persistent Cartesian velocity-control session on the
        first call (pylibfranka's native start_cartesian_velocity_control,
        not position-servo chasing a moving target). Every call after that
        just updates the commanded velocity, which the background loop
        itself acceleration-ramps -- so a jumpy input device can't cause a
        step change on the robot. If no call arrives for ~0.5s, the loop
        ramps the commanded velocity down to zero on its own (protects
        against a disconnected/crashed input device).

        Parameters
        ----------
        linear_velocity  : (3,) m/s, base frame.
        angular_velocity : (3,) rad/s, base frame. Defaults to zero.
        max_vel, max_ang_vel : Speed caps; only used on the first call, when
                       the session starts.
        """
        if self._servo_kind == "joint":
            raise RuntimeError("joint servo is active; call stop() first")
        if self._servo_kind != "cartesian_velocity":
            self._start_cartesian_velocity_servo(max_vel=max_vel, max_ang_vel=max_ang_vel)
        self._set_cartesian_velocity_target(linear_velocity, angular_velocity)

    def move_joints(
        self,
        target_joints,
        velocity: float = 0.5,
        acceleration: float = 1.0,
        asynchronous: bool = False,
    ):
        """
        Blocking (or async) point-to-point move to a target joint configuration.

        Parameters
        ----------
        target_joints : (7,) target joint angles in rad.
        """
        target_joints = np.asarray(target_joints, dtype=float)
        if asynchronous:
            thread = threading.Thread(
                target=self._move_joint,
                kwargs=dict(target_q=target_joints, max_vel=velocity, max_acc=acceleration),
                daemon=True,
            )
            self._async_thread = thread
            thread.start()
            return thread
        return self._move_joint(target_joints, max_vel=velocity, max_acc=acceleration)

    def stop(self):
        """Stop any active motion or servo loop. Mirrors UR5e's stop(): halts motion, keeps the connection alive."""
        if self._servo_kind == "joint":
            self._stop_joint_servo()
        elif self._servo_kind == "cartesian_velocity":
            self._stop_cartesian_velocity_servo()
        elif self.is_moving:
            self.stop_event.set()

    def disconnect(self):
        """Stop any motion and release. Best-effort; mirrors FrankaRobot.disconnect()."""
        try:
            self.stop()
        except Exception:
            pass
        print(f"[FrankaController] Disconnected: {self.robot_ip}")

    # -------------------------
    # Cartesian motion (internal, 16-value O_T_EE)
    # -------------------------

    def _move_ee(
        self,
        target_O_T_EE: Sequence[float],
        max_vel: Optional[float] = None,
        max_acc: Optional[float] = None,
        controller_mode=ControllerMode.JointImpedance,
    ):
        """
        Move EE from current pose to target pose.

        target_O_T_EE:
            16 values, column-major homogeneous transform.
            Translation is index 12, 13, 14.
        """
        if self.is_moving:
            raise RuntimeError("Already moving; stop the current motion or servo loop first")

        target = self._validate_pose16(target_O_T_EE)
        max_vel = max_vel or self.motion_cfg.max_cart_vel
        max_acc = max_acc or self.motion_cfg.max_cart_acc

        with self.motion_lock:
            self.stop_event.clear()
            self.is_moving = True
            self.active_control = None

            try:
                active = self.robot.start_cartesian_pose_control(controller_mode)
                self.active_control = active

                state, duration = active.readOnce()
                self._cache_state(state)

                start_pose = np.array(state.O_T_EE, dtype=float).copy()
                start_pos = start_pose[[12, 13, 14]]
                target_pos = target[[12, 13, 14]]

                delta = target_pos - start_pos
                distance = float(np.linalg.norm(delta))

                if distance > self.motion_cfg.max_cart_distance:
                    raise ValueError(
                        f"Cartesian target too far: {distance:.3f} m > "
                        f"{self.motion_cfg.max_cart_distance:.3f} m"
                    )

                # Send current pose once to avoid first-command jump.
                active.writeOnce(CartesianPose(start_pose))

                R0 = self._pose_to_matrix(start_pose)[:3, :3]
                R1 = self._pose_to_matrix(target)[:3, :3]
                q0 = self._rot_to_quat(R0)
                q1 = self._rot_to_quat(R1)
                dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
                if dot < 0.0:
                    q1 = -q1
                    dot = -dot
                angle = 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))

                if distance < 1e-9 and angle < 1e-9:
                    cmd = CartesianPose(target)
                    cmd.motion_finished = True
                    active.writeOnce(cmd)
                    return

                direction = delta / distance if distance > 1e-9 else np.zeros(3)
                max_ang_vel = self.motion_cfg.max_ang_vel
                max_ang_acc = self.motion_cfg.max_ang_acc

                time_elapsed = 0.0
                motion_finished = False

                while not motion_finished:
                    state, duration = active.readOnce()
                    self._cache_state(state)

                    dt = duration.to_sec()
                    time_elapsed += dt

                    # Translation and rotation are timed independently (each
                    # respecting its own vel/acc cap) and synchronized by
                    # progress, not by sharing one trapezoid derived from
                    # distance alone. Driving the slerp off the translation
                    # trapezoid's progress meant a small position delta with
                    # a large orientation change (e.g. a few cm but 140
                    # degrees) forced the whole rotation through in whatever
                    # short time the tiny translation needed, well past
                    # libfranka's angular velocity/acceleration limits --
                    # exactly the "cartesian_motion_generator_*_discontinuity"
                    # reflex this was tripping.
                    desired_s, total_time_trans = self._trapezoid_position(
                        time_elapsed, distance, max_vel, max_acc,
                    )
                    desired_angle, total_time_rot = self._trapezoid_position(
                        time_elapsed, angle, max_ang_vel, max_ang_acc,
                    )
                    total_time = max(total_time_trans, total_time_rot)

                    progress_trans = float(np.clip(desired_s / distance, 0.0, 1.0)) if distance > 1e-9 else 1.0
                    progress_rot = float(np.clip(desired_angle / angle, 0.0, 1.0)) if angle > 1e-9 else 1.0

                    desired_pos = start_pos + direction * (progress_trans * distance)
                    desired_R = self._quat_to_rot(self._slerp(q0, q1, progress_rot))

                    desired_T = self._pose_to_matrix(start_pose)
                    desired_T[:3, :3] = desired_R
                    desired_T[:3, 3] = desired_pos

                    desired_pose = self._matrix_to_pose(desired_T)
                    cmd = CartesianPose(desired_pose)

                    if self.stop_event.is_set() or time_elapsed >= total_time:
                        cmd.motion_finished = True
                        motion_finished = True

                    active.writeOnce(cmd)
                return 1


            except Exception:
                self.force_stop()
                raise

            finally:
                self.active_control = None
                self.is_moving = False
                self.stop_event.clear()
                try:
                    self.update_idle_state()
                except Exception:
                    pass

    # -------------------------
    # Joint motion (internal, 7-value q)
    # -------------------------

    def _move_joint(
        self,
        target_q: Sequence[float],
        max_vel: Optional[float] = None,
        max_acc: Optional[float] = None,
        controller_mode=ControllerMode.CartesianImpedance,
    ):
        """
        Move from current joint position to target joint position.

        target_q:
            7 joint values in rad.
        """
        if self.is_moving:
            raise RuntimeError("Already moving; stop the current motion or servo loop first")

        target_q = self._validate_joint7(target_q)
        max_vel = max_vel or self.motion_cfg.max_joint_vel
        max_acc = max_acc or self.motion_cfg.max_joint_acc

        with self.motion_lock:
            self.stop_event.clear()
            self.is_moving = True
            self.active_control = None

            try:
                active = self.robot.start_joint_position_control(controller_mode)
                self.active_control = active

                state, duration = active.readOnce()
                self._cache_state(state)

                start_q = np.array(state.q, dtype=float).copy()
                delta = target_q - start_q
                distance = float(np.linalg.norm(delta))

                # Send current joint position once to avoid first-command jump.
                active.writeOnce(JointPositions(start_q.tolist()))

                if distance < 1e-8:
                    cmd = JointPositions(target_q.tolist())
                    cmd.motion_finished = True
                    active.writeOnce(cmd)
                    return

                direction = delta / distance

                time_elapsed = 0.0
                motion_finished = False

                while not motion_finished:
                    state, duration = active.readOnce()
                    self._cache_state(state)

                    dt = duration.to_sec()
                    time_elapsed += dt

                    desired_s, total_time = self._trapezoid_position(
                        time_elapsed,
                        distance,
                        max_vel,
                        max_acc,
                    )

                    desired_q = start_q + direction * desired_s
                    cmd = JointPositions(desired_q.tolist())

                    if self.stop_event.is_set() or time_elapsed >= total_time:
                        cmd.motion_finished = True
                        motion_finished = True

                    active.writeOnce(cmd)

                return 1

            except Exception:
                self.force_stop()
                raise

            finally:
                self.active_control = None
                self.is_moving = False
                self.stop_event.clear()
                try:
                    self.update_idle_state()
                except Exception:
                    pass

    # -------------------------
    # Cartesian velocity servo (persistent realtime loop, internal)
    # -------------------------
    #
    # pylibfranka has a native velocity-control mode: start_cartesian_
    # velocity_control() + writeOnce(CartesianVelocities(...)). There's no
    # "distance to target" to brake against here -- set_ee_velocity() just
    # feeds a live commanded velocity (e.g. from a spacemouse), which this
    # loop acceleration-ramps toward every cycle via the _ramp_toward helper
    # (also used by _joint_servo_loop).

    def _start_cartesian_velocity_servo(
        self,
        max_vel: Optional[float] = None,
        max_ang_vel: Optional[float] = None,
        velocity_watchdog: float = 0.5,
        controller_mode=ControllerMode.JointImpedance,
    ):
        """
        Start a persistent Cartesian velocity-control session.

        Call _set_cartesian_velocity_target() from an outer loop to update
        the commanded velocity; call _stop_cartesian_velocity_servo() when
        done. Do not call _move_ee() while this is active.
        """
        with self.motion_lock:
            if self.is_moving:
                raise RuntimeError("Already moving; call stop() first")

            max_vel = max_vel or self.motion_cfg.max_cart_vel
            max_ang_vel = max_ang_vel or self.motion_cfg.max_ang_vel

            with self._servo_session("cartesian_velocity"):
                active = self.robot.start_cartesian_velocity_control(controller_mode)
                self.active_control = active

                state, _ = active.readOnce()
                self._cache_state(state)
                active.writeOnce(CartesianVelocities([0.0] * 6))

            with self.servo_lock:
                self.servo_vel_target = np.zeros(6)
                self.servo_vel_last_cmd_time = time.time()
            self._servo_vel_current = np.zeros(6)
            self._servo_vel_timeout = velocity_watchdog

            self._servo_thread = threading.Thread(
                target=self._cartesian_velocity_servo_loop,
                args=(active, max_vel, max_ang_vel),
                daemon=True,
            )
            self._servo_thread.start()

    def _set_cartesian_velocity_target(
        self,
        linear_velocity: Sequence[float],
        angular_velocity: Optional[Sequence[float]] = None,
    ):
        """Update the commanded EE velocity chased by the running velocity servo loop."""
        if self._servo_kind != "cartesian_velocity" or self._servo_thread is None:
            raise RuntimeError("Cartesian velocity servo is not running; call _start_cartesian_velocity_servo() first")
        lin = np.asarray(linear_velocity, dtype=float).reshape(3)
        ang = np.zeros(3) if angular_velocity is None else np.asarray(angular_velocity, dtype=float).reshape(3)
        if not np.all(np.isfinite(lin)) or not np.all(np.isfinite(ang)):
            raise ValueError("Velocity command contains NaN or Inf")
        with self.servo_lock:
            self.servo_vel_target = np.concatenate([lin, ang])
            self.servo_vel_last_cmd_time = time.time()

    def _stop_cartesian_velocity_servo(self, timeout: float = 2.0):
        """Stop the running Cartesian velocity servo loop and release control."""
        if self._servo_kind != "cartesian_velocity" or self._servo_thread is None:
            return
        self._stop_servo_thread(timeout)

    def _cartesian_velocity_servo_loop(self, active, max_vel: float, max_ang_vel: float):
        max_step = np.concatenate([
            np.full(3, self.motion_cfg.max_cart_acc),
            np.full(3, self.motion_cfg.max_ang_acc),
        ])
        try:
            while True:
                state, duration = active.readOnce()
                self._cache_state(state)
                dt = duration.to_sec()

                stopping = self.stop_event.is_set()
                if stopping:
                    target = np.zeros(6)
                else:
                    with self.servo_lock:
                        target = self.servo_vel_target.copy()
                        stale = (time.time() - self.servo_vel_last_cmd_time) > self._servo_vel_timeout
                    if stale:
                        target = np.zeros(6)
                    target[:3] = np.clip(target[:3], -max_vel, max_vel)
                    target[3:] = np.clip(target[3:], -max_ang_vel, max_ang_vel)

                self._servo_vel_current = self._ramp_toward(self._servo_vel_current, target, max_step * dt)

                # Only finish once actually back down near zero velocity --
                # cutting the session while still moving would itself be a
                # velocity discontinuity.
                if stopping and np.all(np.abs(self._servo_vel_current) < 1e-4):
                    cmd = CartesianVelocities([0.0] * 6)
                    cmd.motion_finished = True
                    active.writeOnce(cmd)
                    break

                active.writeOnce(CartesianVelocities(self._servo_vel_current.tolist()))

        except Exception as e:
            print(f"[FrankaController] Cartesian velocity servo loop error: {e}")
            self.force_stop()

        finally:
            self.active_control = None
            self.is_moving = False
            self._servo_kind = None
            try:
                self.update_idle_state()
            except Exception:
                pass

    # -------------------------
    # Joint servo (persistent realtime loop, internal)
    # -------------------------

    def _start_joint_servo(
        self,
        max_vel: Optional[float] = None,
        controller_mode=ControllerMode.CartesianImpedance,
    ):
        """
        Start a persistent joint-position control session for realtime servoing.

        Call _set_joint_target() from an outer loop to update the goal joint
        position; call _stop_joint_servo() when done. Do not call _move_joint()
        while a servo session is active.
        """
        with self.motion_lock:
            if self.is_moving:
                raise RuntimeError("Already moving; call _stop_joint_servo() first")

            max_vel = max_vel or self.motion_cfg.max_joint_vel

            with self._servo_session("joint"):
                active = self.robot.start_joint_position_control(controller_mode)
                self.active_control = active

                state, _ = active.readOnce()
                self._cache_state(state)
                current_q = np.array(state.q, dtype=float).copy()
                active.writeOnce(JointPositions(current_q.tolist()))

            with self.servo_lock:
                self.servo_target = current_q.copy()

            self._servo_joint_vel = np.zeros(7)

            self._servo_thread = threading.Thread(
                target=self._joint_servo_loop,
                args=(active, max_vel),
                daemon=True,
            )
            self._servo_thread.start()

    def _set_joint_target(self, target_q: Sequence[float]):
        """Update the goal joint position chased by the running joint servo loop."""
        if self._servo_kind != "joint" or self._servo_thread is None:
            raise RuntimeError("Joint servo is not running; call _start_joint_servo() first")
        target = self._validate_joint7(target_q)
        with self.servo_lock:
            self.servo_target = target

    def _stop_joint_servo(self, timeout: float = 2.0):
        """Stop the running joint servo loop and release control."""
        if self._servo_kind != "joint" or self._servo_thread is None:
            return
        self._stop_servo_thread(timeout)

    def _joint_servo_loop(self, active, max_vel: float):
        max_acc = self.motion_cfg.max_joint_acc
        try:
            while not self.stop_event.is_set():
                state, duration = active.readOnce()
                self._cache_state(state)
                dt = duration.to_sec()

                with self.servo_lock:
                    target_q = self.servo_target.copy()

                current_q = np.array(state.q, dtype=float)
                delta = target_q - current_q

                # Per-joint velocity, ramped by at most max_acc*dt per cycle
                # (not just position-clipped) so the commanded trajectory
                # stays smooth -- see _braking_speed / _ramp_toward.
                desired_vel = np.sign(delta) * self._braking_speed(max_vel, max_acc, delta)
                vel = self._ramp_toward(self._servo_joint_vel, desired_vel, max_acc * dt)

                # Clamp the *position* step to not overshoot, but never force
                # vel itself to zero here -- that would be a second, separate
                # discontinuity. Let it keep decaying smoothly via the ramp
                # above instead, since desired_speed already -> 0 as delta -> 0.
                step = vel * dt
                step = np.where(np.abs(step) > np.abs(delta), delta, step)
                next_q = current_q + step
                self._servo_joint_vel = vel

                cmd = JointPositions(next_q.tolist())
                if self.stop_event.is_set():
                    cmd.motion_finished = True
                active.writeOnce(cmd)

        except Exception as e:
            print(f"[FrankaController] Joint servo loop error: {e}")
            self.force_stop()

        finally:
            self.active_control = None
            self.is_moving = False
            self._servo_kind = None
            self.servo_target = None
            try:
                self.update_idle_state()
            except Exception:
                pass

    def _stop_servo_thread(self, timeout: float):
        self.stop_event.set()
        thread = self._servo_thread
        if thread is not None:
            thread.join(timeout=timeout)
        self._servo_thread = None
        self.stop_event.clear()

    @contextmanager
    def _servo_session(self, kind: str):
        """
        Mark a servo session active for the duration of its setup (opening
        the pylibfranka control session and priming the first command),
        resetting is_moving/_servo_kind if that setup raises before the
        background thread starts. Without this, a failure here (e.g. the
        robot is in "Reflex" or "User Stopped" mode) leaves is_moving stuck
        True forever: stop()/_stop_*_servo() treat "_servo_thread is None"
        as "nothing to do" and return without resetting anything, since
        they're normally only cleaning up an already-running session.
        """
        self.stop_event.clear()
        self.is_moving = True
        self._servo_kind = kind
        try:
            yield
        except Exception:
            self.active_control = None
            self.is_moving = False
            self._servo_kind = None
            raise

    # -------------------------
    # Helpers
    # -------------------------

    @staticmethod
    def _braking_speed(max_speed: float, max_acc: float, error):
        """
        Proportional (not parabolic) braking law: desired speed shrinks
        toward 0 as `error` (distance/angle remaining) shrinks, at a rate
        _ramp_toward's max_acc*dt step can track without lag. A physically
        "exact" parabolic curve (sqrt(2*max_acc*error)) looks appealing but
        desyncs from the ramped speed state once the position step gets
        clamped near arrival, producing a residual discontinuity right at
        the end -- this proportional law doesn't have that failure mode.
        Works for scalar or array `error` (see _joint_servo_loop).
        """
        k = max_acc / max_speed if max_speed > 1e-9 else 0.0
        return np.minimum(max_speed, k * np.abs(error))

    @staticmethod
    def _ramp_toward(current, desired, max_step):
        """Move `current` toward `desired` by at most `max_step` (scalar or array)."""
        return np.clip(desired, current - max_step, current + max_step)

    @staticmethod
    def _validate_pose16(pose: Sequence[float]) -> np.ndarray:
        pose = np.array(pose, dtype=float).reshape(-1)
        if pose.size != 16:
            raise ValueError(f"O_T_EE must have 16 values, got {pose.size}")
        if not np.all(np.isfinite(pose)):
            raise ValueError("O_T_EE contains NaN or Inf")

        T = pose.reshape(4, 4, order="F")
        if not np.allclose(T[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
            raise ValueError("Invalid homogeneous pose: last row should be [0, 0, 0, 1]")

        return pose

    @staticmethod
    def _validate_joint7(q: Sequence[float]) -> np.ndarray:
        q = np.array(q, dtype=float).reshape(-1)
        if q.size != 7:
            raise ValueError(f"Joint command must have 7 values, got {q.size}")
        if not np.all(np.isfinite(q)):
            raise ValueError("Joint command contains NaN or Inf")
        return q

    @staticmethod
    def _trapezoid_position(t: float, distance: float, max_vel: float, max_acc: float):
        """
        Return desired traveled distance at time t.
        This is a scalar path profile.
        """
        if distance <= 1e-12:
            return 0.0, 0.0

        t_acc = max_vel / max_acc
        d_acc = 0.5 * max_acc * t_acc**2

        # Triangular profile: path is too short to reach max_vel.
        if 2.0 * d_acc >= distance:
            t_acc = np.sqrt(distance / max_acc)
            v_peak = max_acc * t_acc
            total_time = 2.0 * t_acc

            if t < t_acc:
                s = 0.5 * max_acc * t**2
            elif t < total_time:
                td = t - t_acc
                s = 0.5 * distance + v_peak * td - 0.5 * max_acc * td**2
            else:
                s = distance

            return float(s), float(total_time)

        # Trapezoidal profile: acceleration, constant velocity, deceleration.
        t_flat = (distance - 2.0 * d_acc) / max_vel
        total_time = 2.0 * t_acc + t_flat

        if t < t_acc:
            s = 0.5 * max_acc * t**2
        elif t < t_acc + t_flat:
            s = d_acc + max_vel * (t - t_acc)
        elif t < total_time:
            td = t - t_acc - t_flat
            s = d_acc + max_vel * t_flat + max_vel * td - 0.5 * max_acc * td**2
        else:
            s = distance

        return float(s), float(total_time)

    @staticmethod
    def _pose_to_matrix(pose16: Sequence[float]) -> np.ndarray:
        return np.array(pose16, dtype=float).reshape(4, 4, order="F")

    @staticmethod
    def _matrix_to_pose(T: np.ndarray) -> np.ndarray:
        return np.array(T, dtype=float).reshape(16, order="F")

    @staticmethod
    def _pose6_to_O_T_EE(pose6: Sequence[float]) -> np.ndarray:
        """Convert (6,) [x, y, z, rx, ry, rz] rotation-vector pose to a 16-value column-major O_T_EE."""
        pose6 = np.asarray(pose6, dtype=float).reshape(6)
        T = np.eye(4)
        T[:3, :3] = st.Rotation.from_rotvec(pose6[3:]).as_matrix()
        T[:3, 3] = pose6[:3]
        return FrankaController._matrix_to_pose(T)

    @staticmethod
    def _O_T_EE_to_pose6(pose16: Sequence[float]) -> np.ndarray:
        """Convert a 16-value column-major O_T_EE to (6,) [x, y, z, rx, ry, rz] rotation-vector pose."""
        T = FrankaController._pose_to_matrix(pose16)
        pos = T[:3, 3]
        rotvec = st.Rotation.from_matrix(T[:3, :3]).as_rotvec()
        return np.concatenate([pos, rotvec]).astype(np.float32)

    @staticmethod
    def _rot_to_quat(R: np.ndarray) -> np.ndarray:
        """
        Rotation matrix to quaternion [w, x, y, z].
        """
        R = np.array(R, dtype=float)
        tr = np.trace(R)

        if tr > 0.0:
            s = np.sqrt(tr + 1.0) * 2.0
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s

        q = np.array([w, x, y, z], dtype=float)
        return q / np.linalg.norm(q)

    @staticmethod
    def _quat_to_rot(q: np.ndarray) -> np.ndarray:
        """
        Quaternion [w, x, y, z] to rotation matrix.
        """
        q = np.array(q, dtype=float)
        q = q / np.linalg.norm(q)
        w, x, y, z = q

        return np.array([
            [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),     2 * (x * z + y * w)],
            [    2 * (x * y + z * w), 1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
            [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])

    @staticmethod
    def _slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
        """
        Quaternion spherical linear interpolation.
        q0, q1: [w, x, y, z]
        alpha: 0 -> q0, 1 -> q1
        """
        q0 = np.array(q0, dtype=float) / np.linalg.norm(q0)
        q1 = np.array(q1, dtype=float) / np.linalg.norm(q1)

        dot = float(np.dot(q0, q1))

        if dot < 0.0:
            q1 = -q1
            dot = -dot

        if dot > 0.9995:
            q = q0 + alpha * (q1 - q0)
            return q / np.linalg.norm(q)

        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)

        theta = theta_0 * alpha
        sin_theta = np.sin(theta)

        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0

        return s0 * q0 + s1 * q1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="172.16.0.2")
    parser.add_argument("--gripper", action="store_true")
    parser.add_argument("--homing", action="store_true")
    args = parser.parse_args()

    franka = FrankaController(
        robot_ip=args.ip,
        use_gripper=args.gripper,
        do_gripper_homing=args.homing,
        realtime=False,
    )


    print(franka.get_tcp_pose())



    # def robot_thread_task():
    #     pose = franka.get_tcp_pose()
    #     target = pose.copy()

    #     # Demo: move +2 cm in x direction.
    #     target[0] += 0.02

    #     franka.move_tcp_pose(
    #         target,
    #         velocity=0.02,
    #         acceleration=0.10,
    #     )

    #     if franka.gripper is not None:
    #         franka.gripper_move(width=0.08, speed=0.05)

    # robot_thread = threading.Thread(target=robot_thread_task, daemon=True)
    # robot_thread.start()

    # try:
    #     while robot_thread.is_alive():
    #         pose = franka.get_tcp_pose()
    #         xyz = pose[:3]
    #         print(f"Current EE xyz: {xyz}")
    #         time.sleep(0.2)

    # except KeyboardInterrupt:
    #     franka.force_stop()

    # robot_thread.join()
    # print("Done")


if __name__ == "__main__":
    main()

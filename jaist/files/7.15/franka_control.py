#!/usr/bin/env python3

import argparse
import threading
import time
from dataclasses import dataclass
from typing import Optional, Sequence

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


@dataclass
class MotionConfig:
    max_cart_vel: float = 0.2     # m/s, conservative default
    max_cart_acc: float = 0.20     # m/s^2
    max_joint_vel: float = 0.20    # rad/s
    max_joint_acc: float = 0.50    # rad/s^2
    max_cart_distance: float = 0.80  # m, reject too-large Cartesian command


class FrankaController:
    def __init__(
        self,
        ip: str = "172.16.0.2",
        use_gripper: bool = True,
        do_gripper_homing: bool = False,
        realtime: bool = False,
        motion_cfg: Optional[MotionConfig] = None,
    ):
        self.ip = ip
        self.motion_cfg = motion_cfg or MotionConfig()

        self.motion_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.stop_event = threading.Event()

        self.robot: Optional[Robot] = None
        self.gripper: Optional[Gripper] = None
        self.active_control = None
        self.is_moving = False

        self.latest_state = None
        self.latest_O_T_EE = None
        self.latest_q = None
        self.latest_wall_time = None

        rt_config = RealtimeConfig.kEnforce if realtime else RealtimeConfig.kIgnore
        self.robot = Robot(ip, rt_config)
        self.set_default_collision_behavior()

        print(f"[FrankaController] Robot connected: {ip}")

        if use_gripper:
            try:
                self.gripper = Gripper(ip)
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

    def recover_errors(self):
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

    def get_ee(self) -> np.ndarray:
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

    def get_joint(self) -> np.ndarray:
        if self.is_moving:
            with self.state_lock:
                if self.latest_q is None:
                    raise RuntimeError("No cached joint state yet")
                return self.latest_q.copy()

        state = self.update_idle_state()
        return np.array(state.q, dtype=float).copy()

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
    # Cartesian motion
    # -------------------------

    def move_ee(
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

                if distance < 1e-6:
                    cmd = CartesianPose(target)
                    cmd.motion_finished = True
                    active.writeOnce(cmd)
                    return

                direction = delta / distance

                R0 = self._pose_to_matrix(start_pose)[:3, :3]
                R1 = self._pose_to_matrix(target)[:3, :3]
                q0 = self._rot_to_quat(R0)
                q1 = self._rot_to_quat(R1)

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

                    progress = float(np.clip(desired_s / distance, 0.0, 1.0))
                    desired_pos = start_pos + direction * desired_s

                    desired_R = self._quat_to_rot(self._slerp(q0, q1, progress))

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
    # Joint motion
    # -------------------------

    def move_joint(
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
    # Helpers
    # -------------------------

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
        ip=args.ip,
        use_gripper=args.gripper,
        do_gripper_homing=args.homing,
        realtime=False,
    )


    print(franka.get_ee())
    


    # def robot_thread_task():
    #     pose = franka.get_ee()
    #     target = pose.copy()

    #     # Demo: move +2 cm in x direction.
    #     target[12] += 0.02

    #     franka.move_ee(
    #         target,
    #         max_vel=0.02,
    #         max_acc=0.10,
    #     )

    #     if franka.gripper is not None:
    #         franka.gripper_move(width=0.08, speed=0.05)

    # robot_thread = threading.Thread(target=robot_thread_task, daemon=True)
    # robot_thread.start()

    # try:
    #     while robot_thread.is_alive():
    #         pose = franka.get_ee()
    #         xyz = pose[[12, 13, 14]]
    #         print(f"Current EE xyz: {xyz}")
    #         time.sleep(0.2)

    # except KeyboardInterrupt:
    #     franka.force_stop()

    # robot_thread.join()
    # print("Done")


if __name__ == "__main__":
    main()
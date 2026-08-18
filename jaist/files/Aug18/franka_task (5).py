import csv
import os
import threading
import time

import numpy as np
import pyspacemouse

from franky import Robot, CartesianVelocityMotion, CartesianVelocityStopMotion, Twist

# =========================
# SpaceMouse -> velocity 매핑 (digital)
# =========================

DEADBAND_X = 0.20
DEADBAND_Y = 0.20
DEADBAND_Z = 0.20
DEADBAND_YAW = 0.20

RAW_X_MIN, RAW_X_MAX = -0.80, 0.80
RAW_Y_MIN, RAW_Y_MAX = -0.80, 0.80
RAW_Z_MIN, RAW_Z_MAX = -0.80, 0.80
RAW_YAW_MIN, RAW_YAW_MAX = -1.00, 1.00

MAX_LINEAR = 0.03
SLOW_LINEAR = 0.01
MAX_YAW = 0.4
MIN_RETARGET_DT = 0.05
KEEPALIVE_DT = 0.15

YAW_EXCLUSIVITY_RATIO = 1.5

# franka가 "멈췄다"고 판단하는 실측 속도 임계값. 이 아래로 내려가야
# flowbot으로 전환을 확정함 -- 완전한 0이 아니라 노이즈 감안한 여유치.
STOP_LINEAR_THRESHOLD = 0.003   # m/s
STOP_ANGULAR_THRESHOLD = 0.01   # rad/s


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def map_axis(raw, raw_min, raw_max, deadband):
    if abs(raw) < deadband:
        return 0.0
    if raw > 0.0:
        if raw_max <= deadband:
            return 0.0
        return clamp((raw - deadband) / (raw_max - deadband), 0.0, 1.0)
    if raw_min >= -deadband:
        return 0.0
    return clamp((raw + deadband) / (abs(raw_min) - deadband), -1.0, 0.0)


def axis_to_velocity(unit, max_pos, max_neg):
    if unit > 0.0:
        return max_pos
    if unit < 0.0:
        return max_neg
    return 0.0


def spacemouse_target(state, max_linear, max_yaw):
    x_unit = map_axis(state.x, RAW_X_MIN, RAW_X_MAX, DEADBAND_X)
    y_unit = map_axis(state.y, RAW_Y_MIN, RAW_Y_MAX, DEADBAND_Y)
    z_unit = map_axis(state.z, RAW_Z_MIN, RAW_Z_MAX, DEADBAND_Z)
    yaw_unit = map_axis(state.yaw, RAW_YAW_MIN, RAW_YAW_MAX, DEADBAND_YAW)

    trans_mag = max(abs(state.x) / RAW_X_MAX, abs(state.y) / RAW_Y_MAX, abs(state.z) / RAW_Z_MAX)
    yaw_mag = abs(state.yaw) / RAW_YAW_MAX

    if yaw_mag > trans_mag * YAW_EXCLUSIVITY_RATIO:
        x_unit = y_unit = z_unit = 0.0
    elif trans_mag > yaw_mag * YAW_EXCLUSIVITY_RATIO:
        yaw_unit = 0.0
    else:
        x_unit = y_unit = z_unit = yaw_unit = 0.0

    vx = axis_to_velocity(x_unit, max_linear, -max_linear)
    vy = axis_to_velocity(y_unit, max_linear, -max_linear)
    vz = axis_to_velocity(z_unit, max_linear, -max_linear)
    wz = axis_to_velocity(yaw_unit, max_yaw, -max_yaw)

    vx, vy, vz = normalize_linear(vx, vy, vz, max_linear)

    return np.array([vx, vy, vz, wz], dtype=float)


def targets_differ(a, b, lin_deadband=0.002, yaw_deadband=0.005):
    if np.linalg.norm(a[:3] - b[:3]) > lin_deadband:
        return True
    if abs(a[3] - b[3]) > yaw_deadband:
        return True
    return False


def normalize_linear(vx, vy, vz, max_speed):
    v = np.array([vx, vy, vz], dtype=float)
    norm = float(np.linalg.norm(v))
    if norm > max_speed and norm > 1e-9:
        v = v / norm * max_speed
    return float(v[0]), float(v[1]), float(v[2])


class FrankaTask(threading.Thread):
    """
    active_mode == "franka"일 때만 SpaceMouse(pyspacemouse)를 열어서 씀.
    1번 버튼이 눌리면(edge 감지) 즉시 정지 명령을 보내고, 실측 속도가
    STOP_*_THRESHOLD 아래로 떨어질 때까지 기다린 뒤에야 디바이스를 닫고
    active_mode를 "flowbot"으로 넘김 -- "franka가 멈춘 뒤에만 전환"
    요구사항을 실측 속도 기반으로 구현.

    active_mode != "franka"인 동안은 디바이스를 아예 안 열고 대기만 함
    (flowbot 쪽이 자기 라이브러리로 디바이스를 독점할 수 있게).
    """

    def __init__(self, shared, ip="172.16.0.2", dynamics_factor=0.05):
        super().__init__()
        self.shared = shared
        self.ip = ip
        self.dynamics_factor = dynamics_factor
        self.robot = None
        self.plot_csv_file = None
        self.plot_writer = None

    def _open_plot_log(self):
        os.makedirs("sessions", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = f"sessions/session_{timestamp}.csv"
        self.plot_csv_file = open(filepath, "w", newline="")
        self.plot_writer = csv.writer(self.plot_csv_file)
        self.plot_writer.writerow([
            "t", "raw_x", "raw_y", "raw_z", "raw_yaw",
            "vx", "vy", "vz", "wz",
            "act_vx", "act_vy", "act_vz", "act_wz",
        ])
        print(f"[Franka] Plot log -> {filepath}")

    def _close_plot_log(self):
        if self.plot_csv_file:
            self.plot_csv_file.close()
            self.plot_csv_file = None

    def run(self):

        print(f"[Franka] Connecting to {self.ip} ...")
        self.robot = Robot(self.ip)
        self.robot.relative_dynamics_factor = self.dynamics_factor
        self.robot.recover_from_errors()
        print(f"[Franka] Connected. relative_dynamics_factor={self.dynamics_factor}")

        while self.shared.running:

            with self.shared.lock:
                mode = self.shared.active_mode

            if mode != "franka":
                # 내 차례가 아니면 디바이스 안 열고 그냥 대기
                time.sleep(0.05)
                continue

            # =========================
            # 내 차례 -> SpaceMouse 열고 제어 루프 시작
            # =========================
            self._open_plot_log()
            print("[Franka] active -> SpaceMouse 여는 중...")

            with pyspacemouse.open() as device:
                print("[Franka] SpaceMouse connected")

                last_target = np.zeros(4, dtype=float)
                last_retarget_time = 0.0
                last_state_query = 0.0
                STATE_QUERY_DT = 0.05
                session_start = time.monotonic()

                last_button0 = False
                stopping = False   # 1번 버튼 눌려서 정지 대기 중인지

                last_button14 = False
                slow_mode = False   # 14번 버튼으로 토글

                while self.shared.running:

                    with self.shared.lock:
                        mode = self.shared.active_mode
                    if mode != "franka":
                        break  # 이미 다른 쪽이 활성화됨 (안전장치)

                    state = device.read()
                    if state is None:
                        time.sleep(0.01)
                        continue
                    time.sleep(0.001)  # busy-spin 방지

                    buttons = list(state.buttons)
                    button0 = bool(buttons[0]) if len(buttons) > 0 else False
                    edge = button0 and not last_button0
                    last_button0 = button0

                    button14 = bool(buttons[14]) if len(buttons) > 14 else False
                    edge14 = button14 and not last_button14
                    last_button14 = button14
                    if edge14:
                        slow_mode = not slow_mode
                        print(f"[Franka] 저속모드 {'ON (0.01 m/s)' if slow_mode else 'OFF (0.03 m/s)'}")

                    if edge and not stopping:
                        stopping = True
                        print("[Franka] 1번 버튼 감지 -> 정지 후 flowbot으로 전환 준비")

                    now = time.monotonic()

                    if stopping:
                        # 더 이상 마우스 입력으로 새 target을 만들지 않고
                        # 그냥 0속도로 고정 -> 정지 명령만 유지
                        target = np.zeros(4, dtype=float)
                        if targets_differ(target, last_target) or (now - last_retarget_time) >= KEEPALIVE_DT:
                            self.robot.move(CartesianVelocityStopMotion(), asynchronous=True)
                            last_target = target.copy()
                            last_retarget_time = now
                    else:
                        target = spacemouse_target(
                            state,
                            max_linear=(SLOW_LINEAR if slow_mode else MAX_LINEAR),
                            max_yaw=MAX_YAW,
                        )

                        changed = targets_differ(target, last_target)
                        elapsed = now - last_retarget_time
                        should_retarget = (
                            (changed and elapsed >= MIN_RETARGET_DT)
                            or elapsed >= KEEPALIVE_DT
                        )
                        if should_retarget:
                            vx, vy, vz, wz = target
                            motion = CartesianVelocityMotion(Twist([vx, vy, vz], [0.0, 0.0, wz]))
                            self.robot.move(motion, asynchronous=True)
                            last_target = target.copy()
                            last_retarget_time = now

                    # =========================
                    # 로봇 상태 조회 + 세션 로그 (20Hz)
                    # =========================
                    act_vx = act_vy = act_vz = act_wz = 0.0
                    if now - last_state_query >= STATE_QUERY_DT:
                        last_state_query = now
                        try:
                            cart_state = self.robot.current_cartesian_state
                            act_v = cart_state.velocity.end_effector_twist
                            act_vx, act_vy, act_vz = act_v.linear
                            act_wz = act_v.angular[2]
                        except Exception as e:
                            print(f"[Franka] state read failed: {e}")

                        self.plot_writer.writerow([
                            f"{now - session_start:.4f}",
                            f"{state.x:.4f}", f"{state.y:.4f}", f"{state.z:.4f}", f"{state.yaw:.4f}",
                            f"{target[0]:.5f}", f"{target[1]:.5f}", f"{target[2]:.5f}", f"{target[3]:.5f}",
                            f"{act_vx:.5f}", f"{act_vy:.5f}", f"{act_vz:.5f}", f"{act_wz:.5f}",
                        ])

                        # =========================
                        # 정지 확인 -> 전환 확정
                        # =========================
                        if stopping:
                            lin_speed = float(np.linalg.norm([act_vx, act_vy, act_vz]))
                            if lin_speed < STOP_LINEAR_THRESHOLD and abs(act_wz) < STOP_ANGULAR_THRESHOLD:
                                print("[Franka] 정지 확인됨 -> flowbot 모드로 전환")
                                with self.shared.lock:
                                    self.shared.active_mode = "flowbot"
                                break

            self._close_plot_log()
            print("[Franka] SpaceMouse 닫음 (flowbot 차례)")

        # =========================
        # 전체 종료
        # =========================
        try:
            self.robot.move(CartesianVelocityStopMotion(), asynchronous=True)
            self.robot.join_motion()
        except Exception as e:
            print(f"[Franka] Stop failed: {e}")

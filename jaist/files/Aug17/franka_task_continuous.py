import csv
import os
import threading
import time

import numpy as np
import pyspacemouse

from franky import Robot, CartesianVelocityMotion, CartesianVelocityStopMotion, Twist

# =========================
# SpaceMouse -> velocity 매핑
# =========================
# digital(on/off) 방식 폐기 -> deadband 넘은 뒤의 unit(0~1)을
# 그대로 0~MAX_LINEAR 사이로 선형 매핑해서 연속적인 속도가 나오도록 함.
# (살짝 밀면 느리게, 세게 밀면 빠르게)

DEADBAND_X = 0.12
DEADBAND_Y = 0.12
DEADBAND_YAW = 0.12

RAW_X_MIN, RAW_X_MAX = -0.80, 0.80
RAW_Y_MIN, RAW_Y_MAX = -0.80, 0.80
RAW_YAW_MIN, RAW_YAW_MAX = -1.00, 1.00

NORMAL_MAX_LINEAR = 0.05   # m/s -- 끝까지 밀었을 때 최대 속도 (기본모드)
SLOW_MAX_LINEAR = 0.01     # m/s -- 저속모드 ('s' 키로 토글)

NORMAL_MAX_YAW = 0.4       # rad/s -- yaw도 동일하게 연속 매핑
SLOW_MAX_YAW = 0.1

MIN_RETARGET_DT = 0.05   # 이 시간보다 자주 move()를 재호출하지 않음
KEEPALIVE_DT = 0.15       # 값이 안 바뀌어도 이 주기마다 같은 명령을 다시 던짐
                          # (계속 누르고 있는데도 멈추는 문제 방지용)

# yaw 비틀 때 손목이 새어나가는 x/y 를 죽이기 위한 상호배타 비율
# (한쪽 크기가 다른 쪽의 이 배수 이상이어야 "확실한 의도"로 보고 반대쪽을 죽임.
#  애매하게 비슷한 크기면 실수 입력으로 보고 둘 다 죽임)
YAW_EXCLUSIVITY_RATIO = 1.5


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def map_axis(raw, raw_min, raw_max, deadband):
    """deadband 밖의 입력을 -1~1 사이 연속값(unit)으로 재스케일링."""
    if abs(raw) < deadband:
        return 0.0
    if raw > 0.0:
        if raw_max <= deadband:
            return 0.0
        return clamp((raw - deadband) / (raw_max - deadband), 0.0, 1.0)
    if raw_min >= -deadband:
        return 0.0
    return clamp((raw + deadband) / (abs(raw_min) - deadband), -1.0, 0.0)


def scale_velocity(unit, max_speed):
    """
    unit(-1~1, deadband 밖에서만 0이 아님)을 0~max_speed 사이 속도로 선형 매핑.
    deadband 넘자마자 0에서 시작해서 끝까지 밀면 max_speed까지 연속적으로 증가.
    """
    return unit * max_speed


def spacemouse_target(state, max_linear, max_yaw):
    x_unit = map_axis(state.x, RAW_X_MIN, RAW_X_MAX, DEADBAND_X)
    y_unit = map_axis(state.y, RAW_Y_MIN, RAW_Y_MAX, DEADBAND_Y)
    yaw_unit = map_axis(state.yaw, RAW_YAW_MIN, RAW_YAW_MAX, DEADBAND_YAW)

    # =========================
    # YAW / TRANSLATION 상호배타
    # =========================
    # yaw를 비틀 때 손목이 미세하게 x/y로도 힘을 줘서 의도치 않은
    # 병진 이동이 같이 발생하는 문제를 막기 위함. 연속 매핑으로 바뀌어도
    # 이 간섭 문제 자체는 그대로 있어서 로직은 유지.

    trans_mag = max(abs(state.x) / RAW_X_MAX, abs(state.y) / RAW_Y_MAX)
    yaw_mag = abs(state.yaw) / RAW_YAW_MAX

    if yaw_mag > trans_mag * YAW_EXCLUSIVITY_RATIO:
        # yaw가 확실히 우세 -> 새어나온 x/y는 죽임
        x_unit = 0.0
        y_unit = 0.0
    elif trans_mag > yaw_mag * YAW_EXCLUSIVITY_RATIO:
        # translation이 확실히 우세 -> 새어나온 yaw는 죽임
        yaw_unit = 0.0
    else:
        # 애매하게 둘 다 비슷한 크기 -> 의도 파악 불가로 보고 둘 다 죽임
        x_unit = 0.0
        y_unit = 0.0
        yaw_unit = 0.0

    vx = scale_velocity(x_unit, max_linear)
    vy = scale_velocity(y_unit, max_linear)
    wz = scale_velocity(yaw_unit, max_yaw)

    # z버튼은 아날로그 값이 없는 물리 on/off 버튼이라 여기만 digital 유지
    buttons = list(state.buttons)
    b0 = buttons[0] if len(buttons) > 0 else 0
    # 물리적 +Z 버튼은 index 1이 아니라 index 14에서 잡힘
    # (test_buttons_only.py로 확인된 사항, spacemouse_cosine.py에서도 동일하게 처리)
    b1 = buttons[14] if len(buttons) > 14 else 0
    if b0 and not b1:
        vz = -max_linear
    elif b1 and not b0:
        vz = max_linear
    else:
        vz = 0.0

    # x/y/z 속도 정규화 (yaw 제외)
    # 대각선으로 동시에 밀면 sqrt(vx^2+vy^2+vz^2)가 max_linear보다 커질 수 있어서
    # 방향은 유지한 채 크기만 max_linear로 눌러줌 -> "최대 속력은 max_linear를
    # 넘지 않는다"는 제약을 유지하기 위해 필요.
    vx, vy, vz = normalize_linear(vx, vy, vz, max_linear)

    return np.array([vx, vy, vz, wz], dtype=float)


def targets_differ(a, b, lin_deadband=0.002, yaw_deadband=0.005):
    if np.linalg.norm(a[:3] - b[:3]) > lin_deadband:
        return True
    if abs(a[3] - b[3]) > yaw_deadband:
        return True
    return False


def normalize_linear(vx, vy, vz, max_speed):
    """
    x/y/z를 동시에 밀면(대각선) 각 축이 독립적으로 max_speed까지 나올 수 있어서
    sqrt(vx^2+vy^2+vz^2)가 max_speed보다 커져 단일 축보다 빨라짐. 방향은 유지한
    채 크기만 max_speed로 눌러줌. yaw(wz)는 회전이라 여기 포함 안 함.
    """
    v = np.array([vx, vy, vz], dtype=float)
    norm = float(np.linalg.norm(v))
    if norm > max_speed and norm > 1e-9:
        v = v / norm * max_speed
    return float(v[0]), float(v[1]), float(v[2])


class FrankaTask(threading.Thread):
    """
    SpaceMouse 읽기 + robot.move() 호출이 한 루프 안에서 같이 돎.
    franky는 Ruckig 기반 실시간 제어를 C++ 쪽(libfranka)에서 처리하기 때문에,
    target이 바뀌면 그냥 move()를 다시 부르면 franky가 알아서
    현재 진행 중인 모션을 안전하게 대체(motion preemption)함.

    속도 자체는 이제 digital(on/off)이 아니라 SpaceMouse를 민 정도에
    비례하는 연속값 -- franky/Ruckig는 "그 target까지 어떻게 부드럽게
    도달할지"(가속 곡선)만 담당하고, target 자체를 조작량에 비례시키는 건
    별개로 spacemouse_target()이 책임짐.

    그래프는 실시간으로 그리지 않고 sessions/*.csv로 파일에만 저장함
    (로봇 제어 중 matplotlib GUI가 GIL을 잡아먹어 reflex fault 나던 문제 방지).
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
        """
        episode CSV랑 같은 방식으로, 실시간으로 그래프를 그리지 않고
        세션 전체를 파일로만 기록. 로봇 제어 스레드 안에서 자체적으로
        열고 닫기 때문에 shared.lock도, matplotlib도 필요 없음.
        """
        os.makedirs("sessions", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = f"sessions/session_{timestamp}.csv"

        self.plot_csv_file = open(filepath, "w", newline="")
        self.plot_writer = csv.writer(self.plot_csv_file)
        self.plot_writer.writerow([
            "t",
            "raw_x", "raw_y", "raw_yaw", "raw_z_btn",
            "vx", "vy", "vz", "wz"
        ])
        print(f"[Franka] Plot log -> {filepath}")

    def _close_plot_log(self):
        if self.plot_csv_file:
            self.plot_csv_file.close()

    def run(self):

        self._open_plot_log()

        print(f"[Franka] Connecting to {self.ip} ...")
        self.robot = Robot(self.ip)
        self.robot.relative_dynamics_factor = self.dynamics_factor
        self.robot.recover_from_errors()
        print(f"[Franka] Connected. relative_dynamics_factor={self.dynamics_factor}")

        with pyspacemouse.open() as device:

            print("SpaceMouse connected")

            last_target = np.zeros(4, dtype=float)
            last_retarget_time = 0.0
            last_pose_query = 0.0
            POSE_QUERY_DT = 0.05

            last_plot_log = 0.0
            PLOT_LOG_DT = 0.05   # 세션 파일 기록 주기 (20Hz) -- move() 호출 빈도와는 별개
            session_start = time.monotonic()

            while self.shared.running:

                state = device.read()

                if state is None:
                    time.sleep(0.01)
                    continue

                with self.shared.lock:
                    slow_mode = self.shared.slow_mode

                if slow_mode:
                    max_linear, max_yaw = SLOW_MAX_LINEAR, SLOW_MAX_YAW
                else:
                    max_linear, max_yaw = NORMAL_MAX_LINEAR, NORMAL_MAX_YAW

                target = spacemouse_target(state, max_linear=max_linear, max_yaw=max_yaw)
                now = time.monotonic()

                changed = targets_differ(target, last_target)
                elapsed = now - last_retarget_time

                # 값이 바뀌었으면 (최소 주기 지켜서) 재호출.
                # 값이 안 바뀌었어도 KEEPALIVE_DT 이상 지났으면 같은 값을
                # 다시 던져서 "계속 누르고 있는데 멈추는" 문제를 방지.
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

                    with self.shared.lock:
                        self.shared.sent_velocity = last_target.tolist()

                # =========================
                # TCP POSE (로거용, 너무 자주 조회하면 루프가 느려지니
                # POSE_QUERY_DT 간격으로만)
                # =========================

                if now - last_pose_query >= POSE_QUERY_DT:
                    last_pose_query = now
                    try:
                        pose = np.array(self.robot.current_pose.end_effector_pose.translation)
                        with self.shared.lock:
                            self.shared.robot_pose = pose
                    except Exception as e:
                        print(f"[Franka] pose read failed: {e}")

                # =========================
                # PLOT LOG (raw 입력 vs 커맨드 속도, 20Hz, 파일로만 저장)
                # =========================
                # 실시간으로 화면에 그리지 않고 그냥 파일에 씀. 연속 매핑이라
                # vx/vy/wz는 raw 크기에 비례해서 부드럽게 나올 거고, vz는
                # 여전히 버튼(digital)이라 계단식으로 나옴 (정상).

                if now - last_plot_log >= PLOT_LOG_DT:
                    last_plot_log = now

                    buttons = list(state.buttons)
                    b0 = buttons[0] if len(buttons) > 0 else 0
                    b1 = buttons[14] if len(buttons) > 14 else 0
                    raw_z_btn = 1 if (b1 and not b0) else (-1 if (b0 and not b1) else 0)

                    self.plot_writer.writerow([
                        f"{now - session_start:.4f}",
                        f"{state.x:.4f}", f"{state.y:.4f}", f"{state.yaw:.4f}", raw_z_btn,
                        f"{target[0]:.5f}", f"{target[1]:.5f}", f"{target[2]:.5f}", f"{target[3]:.5f}",
                    ])

        # =========================
        # SHUTDOWN
        # =========================

        self._close_plot_log()

        try:
            self.robot.move(CartesianVelocityStopMotion(), asynchronous=True)
            self.robot.join_motion()
        except Exception as e:
            print(f"[Franka] Stop failed: {e}")

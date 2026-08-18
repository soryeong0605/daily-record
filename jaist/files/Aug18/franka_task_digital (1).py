import csv
import os
import threading
import time

import numpy as np
import pyspacemouse

from franky import Robot, CartesianVelocityMotion, CartesianVelocityStopMotion, Twist

# =========================
# SpaceMouse -> velocity 매핑 (franky_teleop.py 원본과 동일)
# =========================

DEADBAND_X = 0.12
DEADBAND_Y = 0.12
DEADBAND_YAW = 0.12

RAW_X_MIN, RAW_X_MAX = -0.80, 0.80
RAW_Y_MIN, RAW_Y_MAX = -0.80, 0.80
RAW_YAW_MIN, RAW_YAW_MAX = -1.00, 1.00

MAX_LINEAR = 0.03    # m/s (기본 속도)
SLOW_LINEAR = 0.01   # m/s (저속모드, 's' 키로 토글 -- yaw는 대상 아님)
MAX_YAW = 0.4        # rad/s -- franky/Ruckig가 스무딩을 알아서 해주니까
                      # pylibfranka 때보다 덜 보수적으로 잡아도 됨

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
    if abs(raw) < deadband:
        return 0.0
    if raw > 0.0:
        if raw_max <= deadband:
            return 0.0
        return clamp((raw - deadband) / (raw_max - deadband), 0.0, 1.0)
    if raw_min >= -deadband:
        return 0.0
    return clamp((raw + deadband) / (abs(raw_min) - deadband), -1.0, 0.0)


def axis_to_velocity(unit, max_pos, max_neg, mode):
    if mode == "digital":
        if unit > 0.0:
            return max_pos
        if unit < 0.0:
            return max_neg
        return 0.0
    if unit > 0.0:
        return unit * max_pos
    if unit < 0.0:
        return abs(unit) * max_neg
    return 0.0


def spacemouse_target(state, mode, max_linear=MAX_LINEAR):
    x_unit = map_axis(state.x, RAW_X_MIN, RAW_X_MAX, DEADBAND_X)
    y_unit = map_axis(state.y, RAW_Y_MIN, RAW_Y_MAX, DEADBAND_Y)
    yaw_unit = map_axis(state.yaw, RAW_YAW_MIN, RAW_YAW_MAX, DEADBAND_YAW)

    # =========================
    # YAW / TRANSLATION 상호배타
    # =========================
    # yaw를 비틀 때 손목이 미세하게 x/y로도 힘을 줘서 의도치 않은
    # 병진 이동이 같이 발생하는 문제를 막기 위함.

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

    vx = axis_to_velocity(x_unit, max_linear, -max_linear, mode)
    vy = axis_to_velocity(y_unit, max_linear, -max_linear, mode)
    wz = axis_to_velocity(yaw_unit, MAX_YAW, -MAX_YAW, mode)

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
    x/y/z를 동시에 밀면(대각선) digital 모드 특성상 sqrt(vx^2+vy^2+vz^2)가
    max_speed보다 커져서 단일 축보다 빨라짐. 방향은 유지한 채 크기만
    max_speed로 눌러줌. yaw(wz)는 회전이라 여기 포함 안 함.
    """
    v = np.array([vx, vy, vz], dtype=float)
    norm = float(np.linalg.norm(v))
    if norm > max_speed and norm > 1e-9:
        v = v / norm * max_speed
    return float(v[0]), float(v[1]), float(v[2])


class FrankaTask(threading.Thread):
    """
    franky_teleop.py 원본을 그대로 옮김.

    SpaceMouse 읽기 + robot.move() 호출이 원본처럼 한 루프 안에서 같이 돎.
    franky는 Ruckig 기반 실시간 제어를 C++ 쪽(libfranka)에서 처리하기 때문에,
    pylibfranka + 손으로 짠 코사인 스무더 때와 달리 Python 쪽에서 별도
    실시간 루프를 스레드/프로세스로 분리해서 지켜줄 필요가 없음 --
    target이 바뀌면 그냥 move()를 다시 부르면 franky가 알아서
    현재 진행 중인 모션을 안전하게 대체(motion preemption)함.
    """

    def __init__(self, shared, ip="172.16.0.2", mode="digital", dynamics_factor=0.05):
        super().__init__()
        self.shared = shared
        self.ip = ip
        self.mode = mode
        self.dynamics_factor = dynamics_factor
        self.robot = None
        self.plot_csv_file = None
        self.plot_writer = None

    def _open_plot_log(self):
        """
        episode CSV랑 같은 방식으로, 실시간으로 그래프를 그리지 않고
        세션 전체를 파일로만 기록. 로봇 제어 스레드 안에서 자체적으로
        열고 닫기 때문에 shared.lock도, matplotlib도 필요 없음
        (GIL contention으로 인한 reflex fault 재발 방지).
        """
        os.makedirs("sessions", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = f"sessions/session_{timestamp}.csv"

        self.plot_csv_file = open(filepath, "w", newline="")
        self.plot_writer = csv.writer(self.plot_csv_file)
        self.plot_writer.writerow([
            "t",
            "raw_x", "raw_y", "raw_yaw", "raw_z_btn",
            "vx", "vy", "vz", "wz",                      # 커맨드(target) 속도
            "act_vx", "act_vy", "act_vz", "act_wz",       # 로봇이 실제로 낸 속도
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
        print(f"[Franka] Connected. relative_dynamics_factor={self.dynamics_factor}, mode={self.mode}")

        with pyspacemouse.open() as device:

            print("SpaceMouse connected")

            last_target = np.zeros(4, dtype=float)
            last_retarget_time = 0.0

            last_state_query = 0.0
            STATE_QUERY_DT = 0.05   # 로봇 상태 조회(위치+실측 속도) + 세션 로그 기록 주기 (20Hz)
            session_start = time.monotonic()

            while self.shared.running:

                state = device.read()

                if state is None:
                    time.sleep(0.01)
                    continue

                # =========================
                # CPU 양보 (busy-spin 방지)
                # =========================
                # device.read()가 non-blocking이라 위에서 sleep 없이 그냥
                # 넘어가면 이 루프가 CPU 코어 하나를 100% 점유하면서
                # 무한 반복하게 됨. 이렇게 되면 franky의 실시간 통신
                # 스레드(1kHz)가 같은 코어에서 스케줄링을 순간적으로 놓쳐
                # communication_constraints_violation이 날 수 있음.
                # 1ms 정도의 아주 짧은 sleep만 넣어도 루프 반응성은 거의
                # 그대로 유지하면서 CPU를 양보할 수 있음.
                time.sleep(0.001)

                with self.shared.lock:
                    slow_mode = self.shared.slow_mode
                max_linear = SLOW_LINEAR if slow_mode else MAX_LINEAR

                target = spacemouse_target(state, self.mode, max_linear=max_linear)
                now = time.monotonic()

                if state.x != 0 or state.y != 0 or state.yaw != 0:
                    print(f"[DEBUG raw] x={state.x:.4f} y={state.y:.4f} yaw={state.yaw:.4f} "
                          f"-> target vx={target[0]:.4f} vy={target[1]:.4f} wz={target[3]:.4f}")

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
                    print(f"[DEBUG] move() 호출: vx={vx:.4f} vy={vy:.4f} vz={vz:.4f} wz={wz:.4f}")
                    self.robot.move(motion, asynchronous=True)
                    last_target = target.copy()
                    last_retarget_time = now

                    with self.shared.lock:
                        self.shared.sent_velocity = last_target.tolist()

                # =========================
                # 로봇 상태 조회 + 세션 로그 기록 (20Hz, 하나로 통합)
                # =========================
                # current_pose와 current_cartesian_state를 따로 두 번 부르면
                # 로봇 상태 조회 통신이 두 번 나가서 communication_constraints_
                # violation 위험이 커짐. current_cartesian_state 하나로
                # pose와 velocity를 한 번에 받아오고, 같은 블록 안에서
                # 바로 CSV에도 기록. digital 모드라 target(vx/vy/wz)은
                # -max_linear/0/+max_linear 몇 값만 나오니까 그래프는
                # 계단식으로 보일 거고, act_v*(실측)는 franky/Ruckig의
                # 저크 제한 때문에 그 계단을 부드럽게 따라가는 모습일 거임
                # (둘 다 정상).

                if now - last_state_query >= STATE_QUERY_DT:
                    last_state_query = now

                    try:
                        cart_state = self.robot.current_cartesian_state
                        pose = np.array(cart_state.pose.end_effector_pose.translation)
                        act_v = cart_state.velocity.end_effector_twist
                        act_vx, act_vy, act_vz = act_v.linear
                        act_wz = act_v.angular[2]

                        with self.shared.lock:
                            self.shared.robot_pose = pose
                    except Exception as e:
                        print(f"[Franka] state read failed: {e}")
                        act_vx = act_vy = act_vz = act_wz = float("nan")

                    buttons = list(state.buttons)
                    b0 = buttons[0] if len(buttons) > 0 else 0
                    b1 = buttons[14] if len(buttons) > 14 else 0
                    raw_z_btn = 1 if (b1 and not b0) else (-1 if (b0 and not b1) else 0)

                    self.plot_writer.writerow([
                        f"{now - session_start:.4f}",
                        f"{state.x:.4f}", f"{state.y:.4f}", f"{state.yaw:.4f}", raw_z_btn,
                        f"{target[0]:.5f}", f"{target[1]:.5f}", f"{target[2]:.5f}", f"{target[3]:.5f}",
                        f"{act_vx:.5f}", f"{act_vy:.5f}", f"{act_vz:.5f}", f"{act_wz:.5f}",
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

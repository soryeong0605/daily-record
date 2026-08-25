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

RAW_X_MIN, RAW_X_MAX = -0.80, 0.80
RAW_Y_MIN, RAW_Y_MAX = -0.80, 0.80
RAW_Z_MIN, RAW_Z_MAX = -0.80, 0.80

MAX_LINEAR = 0.03
SLOW_LINEAR = 0.015
MIN_RETARGET_DT = 0.05
KEEPALIVE_DT = 0.15

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


def spacemouse_target(state, max_linear, max_yaw=None):
    """
    max_yaw 파라미터는 호출부(FrankaTask.run)와의 시그니처 호환을 위해
    남겨뒀지만 실제로는 안 씀 -- yaw(joint7 회전) 컨트롤 자체를 없앴기
    때문에 state.yaw는 아예 읽지 않고, wz는 항상 0.0으로 고정.

    x/y 축 스왑: flowbot 쪽 스페이스마우스 입력 방향과 맞추기 위해,
    state.x는 vy에, state.y는 vx에 매핑함 (원래는 x_unit->vx, y_unit->vy
    였는데 서로 바꿈). deadband/범위(DEADBAND_X/Y, RAW_X/Y_MIN/MAX)는
    그대로 각 축 이름에 맞춰 유지 -- 매핑 대상만 바뀜.

    x축 반전: vx(state.y 기반)의 부호를 뒤집음 -- 반대 방향으로 움직이던
    걸 바로잡기 위함. vy는 반전 없이 원래대로, vz는 그대로.
    """
    x_unit = map_axis(state.x, RAW_X_MIN, RAW_X_MAX, DEADBAND_X)
    y_unit = map_axis(state.y, RAW_Y_MIN, RAW_Y_MAX, DEADBAND_Y)
    z_unit = map_axis(state.z, RAW_Z_MIN, RAW_Z_MAX, DEADBAND_Z)

    vx = axis_to_velocity(-y_unit, max_linear, -max_linear)   # state.y -> vx (반전)
    vy = axis_to_velocity(x_unit, max_linear, -max_linear)    # state.x -> vy
    vz = axis_to_velocity(z_unit, max_linear, -max_linear)
    wz = 0.0

    vx, vy, vz = normalize_linear(vx, vy, vz, max_linear)

    return np.array([vx, vy, vz, wz], dtype=float)


def targets_differ(a, b, lin_deadband=0.002):
    return np.linalg.norm(a[:3] - b[:3]) > lin_deadband


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

                # ---- 진단용: 루프 자체가 실제로 몇 Hz로 도는지 실측 ----
                # GIL 경쟁/스레드 스케줄링 지연을 의심할 때, "루프 top에서
                # 다음 루프 top까지" 걸린 시간을 직접 재서 확인하기 위함.
                # device.read()가 이벤트 없을 때 즉시 반환(non-blocking)하니까
                # 정상이면 이 dt는 몇 ms 이내여야 함 -- 그보다 훨씬 크게 튀면
                # 이 스레드가 그동안 GIL을 못 받고 있었다는 뜻.
                STALL_THRESHOLD_MS = 30.0
                _loop_last_time = None
                _loop_dt_window = []
                _loop_iter_count = 0

                while self.shared.running:

                    _loop_now = time.monotonic()
                    if _loop_last_time is not None:
                        _loop_dt = _loop_now - _loop_last_time
                        _loop_dt_ms = _loop_dt * 1000.0

                        if _loop_dt_ms > STALL_THRESHOLD_MS:
                            print(f"[Franka][STALL] 루프 지연 감지: {_loop_dt_ms:.1f}ms "
                                  f"(정상이면 수 ms 이내여야 함 -- GIL 경쟁/스케줄링 지연 의심)")

                        _loop_dt_window.append(_loop_dt)
                        if len(_loop_dt_window) > 200:
                            _loop_dt_window.pop(0)
                        _loop_iter_count += 1
                        if _loop_iter_count % 200 == 0:
                            avg_dt = sum(_loop_dt_window) / len(_loop_dt_window)
                            avg_hz = 1.0 / avg_dt if avg_dt > 0 else float("inf")
                            max_dt_ms = max(_loop_dt_window) * 1000.0
                            print(f"[Franka][LOOP] 최근 200회 평균: {avg_hz:.1f}Hz "
                                  f"(avg_dt={avg_dt*1000:.2f}ms, max_dt={max_dt_ms:.1f}ms)")
                    _loop_last_time = _loop_now

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

                    # ---- 로깅용: 매 read마다 최신 spacemouse raw 값 노출 ----
                    with self.shared.lock:
                        self.shared.sm_raw = np.array(
                            [state.x, state.y, state.z, state.yaw,
                             float(button0), float(button14)],
                            dtype=float,
                        )

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
                            pose = np.array(cart_state.pose.end_effector_pose.translation)
                            act_v = cart_state.velocity.end_effector_twist
                            act_vx, act_vy, act_vz = act_v.linear
                            act_wz = act_v.angular[2]

                            # ---- 로깅용: 조회 성공했을 때만 갱신 (실패 시 이전 값 유지) ----
                            with self.shared.lock:
                                self.shared.franka_pose = pose
                                self.shared.franka_velocity = np.array(
                                    [act_vx, act_vy, act_vz, act_wz], dtype=float
                                )
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

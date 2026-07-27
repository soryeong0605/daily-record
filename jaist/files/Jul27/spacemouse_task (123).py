import threading
import time
import queue
import multiprocessing as mp
from dataclasses import dataclass

import os
import numpy as np
import pyspacemouse

# =========================
# TUNABLE PARAMETERS (spacemouse.py 원본과 동일)
# =========================

SPM_READ_HZ = 60.0
SPM_READ_DT = 1.0 / SPM_READ_HZ
SPM_CONNECT_TIMEOUT = 20.0

DEADBAND_X = 0.12
DEADBAND_Y = 0.12
DEADBAND_YAW = 0.12

RAW_X_MIN = -0.80
RAW_X_MAX = +0.80

RAW_Y_MIN = -0.80
RAW_Y_MAX = +0.80

RAW_YAW_MIN = -1.00
RAW_YAW_MAX = +1.00

MAX_VX_POS = +0.03
MAX_VX_NEG = -0.03

MAX_VY_POS = +0.03
MAX_VY_NEG = -0.03

BUTTON_Z_SPEED = 0.03

MAX_WZ_POS = +0.5
MAX_WZ_NEG = -0.5

SIGN_X = +1.0
SIGN_Y = +1.0
SIGN_Z = +1.0
SIGN_YAW = +1.0

YAW_EXCLUSIVITY_RATIO = 1.5

INPUT_MODE = "digital"

IDX_STAMP = 0
IDX_VX = 1
IDX_VY = 2
IDX_VZ = 3
IDX_WZ = 4
SHARED_SIZE = 5


@dataclass
class SPMCommand:
    stamp: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    wz: float = 0.0


def clamp(value, low, high):
    return max(low, min(high, value))


def map_raw_axis_to_unit(raw_value, raw_min, raw_max, deadband):
    if abs(raw_value) < deadband:
        return 0.0

    if raw_value > 0.0:
        if raw_max <= deadband:
            return 0.0
        return clamp((raw_value - deadband) / (raw_max - deadband), 0.0, 1.0)

    if raw_min >= -deadband:
        return 0.0

    return clamp((raw_value + deadband) / (abs(raw_min) - deadband), -1.0, 0.0)


def unit_to_velocity(unit_value, max_positive_velocity, max_negative_velocity, mode="digital"):
    if mode == "digital":
        if unit_value > 0.0:
            return max_positive_velocity
        if unit_value < 0.0:
            return max_negative_velocity
        return 0.0

    if unit_value > 0.0:
        return unit_value * max_positive_velocity
    if unit_value < 0.0:
        return abs(unit_value) * max_negative_velocity
    return 0.0


def normalize_linear(vx, vy, max_speed):
    v = np.array([vx, vy], dtype=float)
    norm = float(np.linalg.norm(v))
    if norm > max_speed and norm > 1e-9:
        v = v / norm * max_speed
    return float(v[0]), float(v[1])


def command_from_state(state, mode="digital"):
    now = time.time()

    raw_x = float(state.x)
    raw_y = float(state.y)
    raw_yaw = float(state.yaw)
    buttons = list(state.buttons)

    b0 = int(buttons[0]) if len(buttons) > 0 else 0
    # 물리적 +Z 버튼은 index 14에서 뜸 (test_buttons_only.py로 확인된 사항)
    b1 = int(buttons[14]) if len(buttons) > 14 else 0

    x_unit = map_raw_axis_to_unit(SIGN_X * raw_x, RAW_X_MIN, RAW_X_MAX, DEADBAND_X)
    y_unit = map_raw_axis_to_unit(SIGN_Y * raw_y, RAW_Y_MIN, RAW_Y_MAX, DEADBAND_Y)
    yaw_unit = map_raw_axis_to_unit(SIGN_YAW * raw_yaw, RAW_YAW_MIN, RAW_YAW_MAX, DEADBAND_YAW)

    trans_mag = max(abs(raw_x) / RAW_X_MAX, abs(raw_y) / RAW_Y_MAX)
    yaw_mag = abs(raw_yaw) / RAW_YAW_MAX

    if yaw_mag > trans_mag * YAW_EXCLUSIVITY_RATIO:
        x_unit = 0.0
        y_unit = 0.0
    elif trans_mag > yaw_mag * YAW_EXCLUSIVITY_RATIO:
        yaw_unit = 0.0
    else:
        x_unit = 0.0
        y_unit = 0.0
        yaw_unit = 0.0

    vx = unit_to_velocity(x_unit, MAX_VX_POS, MAX_VX_NEG, mode)
    vy = unit_to_velocity(y_unit, MAX_VY_POS, MAX_VY_NEG, mode)
    wz = unit_to_velocity(yaw_unit, MAX_WZ_POS, MAX_WZ_NEG, mode)

    if b0 == 1 and b1 == 0:
        vz = -SIGN_Z * BUTTON_Z_SPEED
    elif b1 == 1 and b0 == 0:
        vz = +SIGN_Z * BUTTON_Z_SPEED
    else:
        vz = 0.0

    vx, vy = normalize_linear(vx, vy, MAX_VX_POS)

    return SPMCommand(stamp=now, vx=vx, vy=vy, vz=vz, wz=wz)


def zero_command():
    return SPMCommand(stamp=time.time())


def write_array(array, cmd):
    array[IDX_STAMP] = cmd.stamp
    array[IDX_VX] = cmd.vx
    array[IDX_VY] = cmd.vy
    array[IDX_VZ] = cmd.vz
    array[IDX_WZ] = cmd.wz


def read_array(array):
    return SPMCommand(
        stamp=float(array[IDX_STAMP]),
        vx=float(array[IDX_VX]),
        vy=float(array[IDX_VY]),
        vz=float(array[IDX_VZ]),
        wz=float(array[IDX_WZ]),
    )


# =========================
# 별도 프로세스에서 도는 워커
# =========================
# Franka 실시간 서보 스레드랑 같은 프로세스 안에서 GIL을 나눠쓰면
# (스레드로 하면), 실시간 루프가 바쁠 때 "손 뗐다"는 갱신이 늦게
# 반영될 수 있음. spacemouse.py 원본처럼 완전히 분리된 OS 프로세스로 뺌.

def spacemouse_process(array, connected_event, stop_event, status_q, mode="digital"):
    try:
        try:
            os.nice(10)
        except Exception:
            pass

        status_q.put(("opening", time.time()))

        device = pyspacemouse.open()

        if not device:
            status_q.put(("error", "SpaceMouse not found", time.time()))
            return

        status_q.put(("connected", time.time()))
        connected_event.set()

        next_read = time.monotonic()

        while not stop_event.is_set():
            state = device.read()

            if state is not None:
                cmd = command_from_state(state, mode)
                write_array(array, cmd)

            next_read += SPM_READ_DT
            sleep_time = next_read - time.monotonic()

            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                next_read = time.monotonic()
                time.sleep(SPM_READ_DT)

    except BaseException as e:
        status_q.put(("error", repr(e), time.time()))

    finally:
        write_array(array, zero_command())
        status_q.put(("closed", time.time()))


class SpaceMouseTask(threading.Thread):
    """
    실제 SpaceMouse 폴링 + 속도 계산은 별도 프로세스(spacemouse_process)가
    담당 -- spacemouse.py 원본과 동일한 구조. 이 스레드는 그 프로세스가
    mp.Array에 써둔 값을 shared.spacemouse dict로 복사만 하는 가벼운
    다리 역할이라 GIL 부담이 거의 없음.
    """

    def __init__(self, shared, mode=INPUT_MODE):
        super().__init__()
        self.shared = shared
        self.mode = mode

        self.ctx = mp.get_context("spawn")
        self.array = self.ctx.Array("d", SHARED_SIZE, lock=False)
        self.connected_event = self.ctx.Event()
        self.stop_event = self.ctx.Event()
        self.status_q = self.ctx.Queue()

        write_array(self.array, zero_command())

        self.proc = self.ctx.Process(
            target=spacemouse_process,
            args=(self.array, self.connected_event, self.stop_event, self.status_q, self.mode),
            daemon=True,
        )

    def run(self):

        self.proc.start()

        print("[SpaceMouse] Waiting for connection...")
        if not self.connected_event.wait(timeout=SPM_CONNECT_TIMEOUT):
            print("[SpaceMouse] Connection timeout")
            return

        while True:
            try:
                msg = self.status_q.get_nowait()
                print("[SpaceMouse]", msg)
            except queue.Empty:
                break

        print("SpaceMouse connected")

        while self.shared.running:

            cmd = read_array(self.array)

            with self.shared.lock:
                self.shared.spacemouse["vx"] = cmd.vx
                self.shared.spacemouse["vy"] = cmd.vy
                self.shared.spacemouse["vz"] = cmd.vz
                self.shared.spacemouse["wz"] = cmd.wz
                self.shared.spacemouse["stamp"] = cmd.stamp

            time.sleep(0.01)  # 브릿징만 하니까 가볍게 100Hz

        # =========================
        # SHUTDOWN
        # =========================

        self.stop_event.set()

        if self.proc.is_alive():
            self.proc.join(timeout=2.0)

        if self.proc.is_alive():
            self.proc.terminate()

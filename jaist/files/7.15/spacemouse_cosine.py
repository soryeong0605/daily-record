#!/usr/bin/env python3
"""
spacemouse.py - native Cartesian velocity + raised-cosine smoother.

Use with franka.py generated together with this file.

Run:
    python3 spacemouse.py --ip 172.16.0.2 --recover

Debug:
    python3 spacemouse.py --ip 172.16.0.2 --recover --debug --debug-franka
"""

import argparse
import multiprocessing as mp
import os
import queue
import time
from dataclasses import dataclass

import pyspacemouse

from franka_cosine import FrankaController, NativeVelocityConfig


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

MAX_VX_POS = +0.06
MAX_VX_NEG = -0.06

MAX_VY_POS = +0.06
MAX_VY_NEG = -0.06

BUTTON_Z_SPEED = 0.06

MAX_WZ_POS = +0.5
MAX_WZ_NEG = -0.5

SIGN_X = +1.0
SIGN_Y = +1.0
SIGN_Z = +1.0
SIGN_YAW = +1.0

IDX_STAMP = 0
IDX_VX = 1
IDX_VY = 2
IDX_VZ = 3
IDX_WZ = 4

IDX_RAW_X = 5
IDX_RAW_Y = 6
IDX_RAW_YAW = 7

IDX_X_UNIT = 8
IDX_Y_UNIT = 9
IDX_YAW_UNIT = 10

IDX_B0 = 11
IDX_B1 = 12
IDX_ACTIVE = 13
IDX_SEQ = 14

SHARED_SIZE = 15


@dataclass
class SPMCommand:
    stamp: float = 0.0

    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    wz: float = 0.0

    raw_x: float = 0.0
    raw_y: float = 0.0
    raw_yaw: float = 0.0

    x_unit: float = 0.0
    y_unit: float = 0.0
    yaw_unit: float = 0.0

    b0: int = 0
    b1: int = 0
    active: bool = False
    seq: int = 0


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


def unit_to_velocity(unit_value, max_positive_velocity, max_negative_velocity):
    if unit_value > 0.0:
        return unit_value * max_positive_velocity

    if unit_value < 0.0:
        return abs(unit_value) * max_negative_velocity

    return 0.0


def command_from_state(state, seq):
    now = time.time()

    raw_x = float(state.x)
    raw_y = float(state.y)
    raw_yaw = float(state.yaw)
    buttons = list(state.buttons)

    b0 = int(buttons[0]) if len(buttons) > 0 else 0
    b1 = int(buttons[1]) if len(buttons) > 1 else 0

    x_unit = map_raw_axis_to_unit(SIGN_X * raw_x, RAW_X_MIN, RAW_X_MAX, DEADBAND_X)
    y_unit = map_raw_axis_to_unit(SIGN_Y * raw_y, RAW_Y_MIN, RAW_Y_MAX, DEADBAND_Y)
    yaw_unit = map_raw_axis_to_unit(SIGN_YAW * raw_yaw, RAW_YAW_MIN, RAW_YAW_MAX, DEADBAND_YAW)

    vx = unit_to_velocity(x_unit, MAX_VX_POS, MAX_VX_NEG)
    vy = unit_to_velocity(y_unit, MAX_VY_POS, MAX_VY_NEG)
    wz = unit_to_velocity(yaw_unit, MAX_WZ_POS, MAX_WZ_NEG)

    if b0 == 1 and b1 == 0:
        vz = -SIGN_Z * BUTTON_Z_SPEED
    elif b1 == 1 and b0 == 0:
        vz = +SIGN_Z * BUTTON_Z_SPEED
    else:
        vz = 0.0

    active = (
        abs(vx) > 1e-9
        or abs(vy) > 1e-9
        or abs(vz) > 1e-9
        or abs(wz) > 1e-9
        or b0 == 1
        or b1 == 1
    )

    return SPMCommand(
        stamp=now,
        vx=vx,
        vy=vy,
        vz=vz,
        wz=wz,
        raw_x=raw_x,
        raw_y=raw_y,
        raw_yaw=raw_yaw,
        x_unit=x_unit,
        y_unit=y_unit,
        yaw_unit=yaw_unit,
        b0=b0,
        b1=b1,
        active=active,
        seq=seq,
    )


def zero_command():
    return SPMCommand(stamp=time.time())


def write_shared(shared, cmd):
    shared[IDX_STAMP] = cmd.stamp

    shared[IDX_VX] = cmd.vx
    shared[IDX_VY] = cmd.vy
    shared[IDX_VZ] = cmd.vz
    shared[IDX_WZ] = cmd.wz

    shared[IDX_RAW_X] = cmd.raw_x
    shared[IDX_RAW_Y] = cmd.raw_y
    shared[IDX_RAW_YAW] = cmd.raw_yaw

    shared[IDX_X_UNIT] = cmd.x_unit
    shared[IDX_Y_UNIT] = cmd.y_unit
    shared[IDX_YAW_UNIT] = cmd.yaw_unit

    shared[IDX_B0] = float(cmd.b0)
    shared[IDX_B1] = float(cmd.b1)
    shared[IDX_ACTIVE] = 1.0 if cmd.active else 0.0
    shared[IDX_SEQ] = float(cmd.seq)


def read_shared(shared):
    return SPMCommand(
        stamp=float(shared[IDX_STAMP]),

        vx=float(shared[IDX_VX]),
        vy=float(shared[IDX_VY]),
        vz=float(shared[IDX_VZ]),
        wz=float(shared[IDX_WZ]),

        raw_x=float(shared[IDX_RAW_X]),
        raw_y=float(shared[IDX_RAW_Y]),
        raw_yaw=float(shared[IDX_RAW_YAW]),

        x_unit=float(shared[IDX_X_UNIT]),
        y_unit=float(shared[IDX_Y_UNIT]),
        yaw_unit=float(shared[IDX_YAW_UNIT]),

        b0=int(shared[IDX_B0]),
        b1=int(shared[IDX_B1]),
        active=bool(shared[IDX_ACTIVE] > 0.5),
        seq=int(shared[IDX_SEQ]),
    )


def spacemouse_process(shared, connected_event, stop_event, status_q, read_count):
    try:
        # Lower priority so SpaceMouse cannot starve Franka.
        try:
            os.nice(10)
        except Exception:
            pass

        status_q.put(("opening", time.time()))

        with pyspacemouse.open() as device:
            status_q.put(("connected", time.time()))
            connected_event.set()

            seq = 0
            next_read = time.monotonic()

            while not stop_event.is_set():
                state = device.read()

                if state is not None:
                    seq += 1
                    cmd = command_from_state(state, seq)
                    write_shared(shared, cmd)

                    with read_count.get_lock():
                        read_count.value += 1

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
        write_shared(shared, zero_command())
        status_q.put(("closed", time.time()))


def drain_status(status_q):
    has_error = False

    while True:
        try:
            msg = status_q.get_nowait()
        except queue.Empty:
            break

        print("[SpaceMouse]", msg)

        if msg and msg[0] == "error":
            has_error = True

    return has_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="172.16.0.2")
    parser.add_argument("--recover", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-franka", action="store_true")
    parser.add_argument("--no-robot-motion", action="store_true")

    # Useful tuning knobs.
    parser.add_argument("--transition-up", type=float, default=0.60)
    parser.add_argument("--transition-down", type=float, default=0.25)

    args = parser.parse_args()

    ctx = mp.get_context("spawn")

    shared = ctx.Array("d", SHARED_SIZE, lock=False)
    connected_event = ctx.Event()
    stop_event = ctx.Event()
    status_q = ctx.Queue()
    read_count = ctx.Value("i", 0)

    write_shared(shared, zero_command())

    proc = ctx.Process(
        target=spacemouse_process,
        args=(shared, connected_event, stop_event, status_q, read_count),
        daemon=True,
    )
    proc.start()

    print("[Main] Waiting for SpaceMouse...")
    if not connected_event.wait(timeout=SPM_CONNECT_TIMEOUT):
        drain_status(status_q)
        stop_event.set()
        proc.join(timeout=2.0)
        raise RuntimeError("SpaceMouse did not connect")

    drain_status(status_q)

    franka = None

    def velocity_getter():
        cmd = read_shared(shared)
        return cmd.vx, cmd.vy, cmd.vz, cmd.wz, cmd.stamp

    try:
        if not args.no_robot_motion:
            franka = FrankaController(
                ip=args.ip,
                use_gripper=False,
                realtime=False,
            )

            if args.recover:
                franka.recover_errors()

            cfg = NativeVelocityConfig(
                max_linear_speed=0.06,
                max_yaw_speed=0.2,
                transition_time_up=args.transition_up,
                transition_time_down=args.transition_down,
                min_retarget_dt=0.10,
                linear_target_deadband=0.0004,
                yaw_target_deadband=0.0008,
                command_timeout=0.35,
                startup_zero_time=0.8,
            )

            franka.start_ee_velocity_servo(
                velocity_getter=velocity_getter,
                cfg=cfg,
                print_debug=args.debug_franka,
                print_dt=1.0,
            )
        else:
            print("[Main] no-robot-motion mode")

        print("[Main] Teleop running with native Cartesian velocity + cosine smoother")
        print("[Main] Press Ctrl+C to stop")
        print("")

        last_print = 0.0
        last_count = 0
        last_time = time.time()

        while True:
            time.sleep(0.05)

            if drain_status(status_q):
                raise RuntimeError("SpaceMouse process error")

            if franka is not None:
                th = getattr(franka, "velocity_thread", None)
                running = getattr(franka, "velocity_servo_running", None)

                if th is not None and not th.is_alive():
                    print("[Main] Franka velocity thread died.")
                    break

                if running is not None and not running.is_set():
                    print("[Main] Franka velocity servo stopped.")
                    break

            if args.debug and time.time() - last_print >= 1.0:
                now = time.time()
                last_print = now

                cmd = read_shared(shared)

                with read_count.get_lock():
                    c = int(read_count.value)

                dt = max(now - last_time, 1e-6)
                hz = (c - last_count) / dt
                last_time = now
                last_count = c

                age = now - cmd.stamp if cmd.stamp > 0 else 999.0

                print(
                    "[Main] "
                    f"age={age:.3f}, "
                    f"raw=[{cmd.raw_x:+.3f}, {cmd.raw_y:+.3f}, {cmd.raw_yaw:+.3f}], "
                    f"unit=[{cmd.x_unit:+.3f}, {cmd.y_unit:+.3f}, {cmd.yaw_unit:+.3f}], "
                    f"cmd=[{cmd.vx:+.4f}, {cmd.vy:+.4f}, {cmd.vz:+.4f}, {cmd.wz:+.4f}], "
                    f"buttons=[{cmd.b0}, {cmd.b1}], "
                    f"spm_hz={hz:.1f}"
                )

    except KeyboardInterrupt:
        print("\n[Main] Stopping...")

    finally:
        if franka is not None:
            try:
                franka.stop_ee_velocity_servo()
            except Exception as e:
                print(f"[Main] Franka stop failed: {e}")

        stop_event.set()

        if proc.is_alive():
            proc.join(timeout=2.0)

        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1.0)

        print("[Main] Stopped")


if __name__ == "__main__":
    main()
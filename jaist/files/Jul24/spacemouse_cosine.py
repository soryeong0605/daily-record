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

import numpy as np
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

RAW_Z_MIN = -0.80
RAW_Z_MAX = +0.80

RAW_YAW_MIN = -1.00
RAW_YAW_MAX = +1.00

MAX_VX_POS = +0.03
MAX_VX_NEG = -0.03

MAX_VY_POS = +0.03
MAX_VY_NEG = -0.03

MAX_VZ_POS = +0.03
MAX_VZ_NEG = -0.03

DEADBAND_Z = 0.12

# z button speed -- kept at the same 0.03 m/s as x/y/yaw for a uniform
# speed across all axes (per user's explicit intent).
BUTTON_Z_SPEED = 0.03

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


def unit_to_velocity(unit_value, max_positive_velocity, max_negative_velocity, mode="digital"):
    """
    mode="digital" (default): once unit_value passes the deadband in a
        direction, return a FIXED constant speed -- magnitude of the push
        is ignored, just like a keyboard key (held = constant speed,
        released = zero). Removes the need to finely modulate push
        strength by hand.
    mode="analog": original behavior, speed scales proportionally with
        how hard the SpaceMouse is pushed.
    """
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


def normalize_linear(vx, vy, vz, max_speed):
    """
    Scale the combined [vx, vy, vz] vector so its magnitude never exceeds
    max_speed, while preserving direction. Fixes the issue where pushing
    two axes at once (e.g. X and Y diagonally) in digital mode would add
    up to max_speed*sqrt(2) -- about 1.4x faster than a single-axis push.
    If the vector is already at/under max_speed (e.g. a light analog
    push), it is left unchanged.
    """
    import numpy as np
    v = np.array([vx, vy, vz], dtype=float)
    norm = float(np.linalg.norm(v))
    if norm > max_speed and norm > 1e-9:
        v = v / norm * max_speed
    return float(v[0]), float(v[1]), float(v[2])


def command_from_state(state, seq, mode="digital"):
    now = time.time()

    raw_x = float(state.x)
    raw_y = float(state.y)
    raw_z = float(state.z)
    raw_yaw = float(state.yaw)
    buttons = list(state.buttons)

    b0 = int(buttons[0]) if len(buttons) > 0 else 0
    # NOTE: the physical "+Z (up)" button was found (via test_buttons_only.py)
    # to actually report at index 14, not index 1 -- SpaceMouse devices with
    # many extra keys (this Universal Receiver-paired unit) don't always map
    # button 2 to index 1. Adjusted accordingly.
    b1 = int(buttons[14]) if len(buttons) > 14 else 0

    x_unit = map_raw_axis_to_unit(SIGN_X * raw_x, RAW_X_MIN, RAW_X_MAX, DEADBAND_X)
    y_unit = map_raw_axis_to_unit(SIGN_Y * raw_y, RAW_Y_MIN, RAW_Y_MAX, DEADBAND_Y)
    yaw_unit = map_raw_axis_to_unit(SIGN_YAW * raw_yaw, RAW_YAW_MIN, RAW_YAW_MAX, DEADBAND_YAW)

    vx = unit_to_velocity(x_unit, MAX_VX_POS, MAX_VX_NEG, mode)
    vy = unit_to_velocity(y_unit, MAX_VY_POS, MAX_VY_NEG, mode)
    wz = unit_to_velocity(yaw_unit, MAX_WZ_POS, MAX_WZ_NEG, mode)

    # Back to button-based z control (b0 = down, b1 = up) -- felt more
    # intuitive/reliable than driving z from the handle's own push/pull
    # motion (state.z), since that motion can get mixed up with x/y.
    if b0 == 1 and b1 == 0:
        vz = -SIGN_Z * BUTTON_Z_SPEED
    elif b1 == 1 and b0 == 0:
        vz = +SIGN_Z * BUTTON_Z_SPEED
    else:
        vz = 0.0

    # Normalize only the [vx, vy] planar vector so diagonal motion (e.g. X+Y
    # at once) isn't faster than single-axis motion. z is excluded from
    # this xy normalization since it's driven independently by a button
    # (BUTTON_Z_SPEED); with BUTTON_Z_SPEED now also set to 0.03 (same as
    # MAX_VX_POS/MAX_VY_POS), all axes move at a uniform 0.03 m/s.
    vx, vy, _ = normalize_linear(vx, vy, 0.0, MAX_VX_POS)

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


def spacemouse_process(shared, connected_event, stop_event, status_q, read_count, mode="digital"):
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
                    cmd = command_from_state(state, seq, mode)
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


def get_tcp_xyz(franka):
    """
    Extract [x, y, z] from FrankaController.get_ee(), which returns the
    16-element column-major O_T_EE matrix. Translation lives at indices
    12, 13, 14 (standard libfranka convention).
    """
    O_T_EE = franka.get_ee()
    return np.array([O_T_EE[12], O_T_EE[13], O_T_EE[14]], dtype=float)


class PoseWindow:
    """
    Small OpenCV window continuously showing the robot's current TCP
    [x, y, z], same black-panel style as the earlier velocity monitor.
    Read-only, never touches control. Press ESC (window focused) to close.
    """
    WINDOW_NAME = "Franka TCP Position"

    def __init__(self, refresh_hz=10.0):
        import cv2
        self.cv2 = cv2
        self.panel_w, self.panel_h = 620, 220
        self.min_dt = 1.0 / refresh_hz if refresh_hz > 0 else 0.0
        self._last_update = 0.0
        self._closed = False
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, self.panel_w, self.panel_h)

    def is_open(self):
        return not self._closed

    def maybe_update(self, franka, now):
        cv2 = self.cv2

        # Always pump the GUI event loop every tick, even on ticks where we
        # skip the actual redraw below (rate-limited by min_dt). On Linux
        # (X11/GTK) backends, cv2 windows can appear frozen/unresponsive if
        # waitKey() isn't called frequently and consistently, regardless of
        # how often the window's *content* actually changes.
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            self._closed = True
            return

        if self._closed or (now - self._last_update) < self.min_dt:
            return
        self._last_update = now

        x, y, z = get_tcp_xyz(franka)
        panel = np.zeros((self.panel_h, self.panel_w, 3), dtype=np.uint8)
        cv2.putText(panel, "Franka TCP Position (live)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.putText(panel, f"x: {x:+.4f} m   y: {y:+.4f} m   z: {z:+.4f} m",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (140, 255, 140), 2)
        cv2.putText(panel, "Press ESC (window focused) or Ctrl+C to stop.",
                    (20, self.panel_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (170, 170, 170), 1)
        cv2.imshow(self.WINDOW_NAME, panel)

    def close(self):
        if not self._closed:
            self.cv2.destroyWindow(self.WINDOW_NAME)
            self._closed = True


def move_to_target_closed_loop(franka, target_xyz, max_speed=0.03, kp=1.5,
                                 tol=0.003, timeout=15.0):
    """
    Simple closed-loop P-controller move: "go to this xyz position and stop
    once close enough". This is NOT franky's built-in point-to-point move --
    FrankaController only exposes Cartesian VELOCITY control, so instead we
    reuse the already-validated start_ee_velocity_servo() (hybrid cosine +
    slew-rate safety net) with a velocity_getter that computes
        v = clip(kp * (target - current), max_speed)
    every tick, and stop once the position error is under `tol`. Because
    this goes through the exact same safety-netted velocity servo used all
    day, it inherits the same acceleration-discontinuity protection.
    """
    print(f"[Move] Moving to target xyz={np.round(target_xyz, 4).tolist()} "
          f"(closed-loop, max_speed={max_speed} m/s) ...")

    state = {"done": False}

    def velocity_getter():
        current = get_tcp_xyz(franka)
        error = np.asarray(target_xyz) - current
        dist = float(np.linalg.norm(error))

        if dist < tol:
            state["done"] = True
            return 0.0, 0.0, 0.0, 0.0, time.time()

        v = error * kp
        speed = float(np.linalg.norm(v))
        if speed > max_speed:
            v = v / speed * max_speed

        return float(v[0]), float(v[1]), float(v[2]), 0.0, time.time()

    cfg = NativeVelocityConfig(
        max_linear_speed=max_speed,
        max_yaw_speed=0.05,
        transition_time_up=0.3,
        transition_time_down=0.2,
        min_retarget_dt=0.02,
        command_timeout=0.5,
        startup_zero_time=0.2,
    )

    franka.start_ee_velocity_servo(velocity_getter=velocity_getter, cfg=cfg)

    t0 = time.monotonic()
    while not state["done"] and (time.monotonic() - t0) < timeout:
        time.sleep(0.05)

    franka.stop_ee_velocity_servo()

    reached = get_tcp_xyz(franka)
    err = float(np.linalg.norm(np.asarray(target_xyz) - reached))
    if state["done"]:
        print(f"[Move] Reached target. xyz={np.round(reached, 4).tolist()} "
              f"(error={err*1000:.1f} mm)")
    else:
        print(f"[Move] Timed out before reaching target. "
              f"xyz={np.round(reached, 4).tolist()} (error={err*1000:.1f} mm)")
    return reached


def capture_home_position(franka):
    """
    One-time helper: hand-guide the robot (using its own physical guiding
    buttons on the arm -- this works independently of any code, as long as
    no velocity servo is currently running) to the posture you want as the
    fixed starting position, then read out its TCP xyz.
    """
    print("\n[Capture home] ================================================")
    print("[Capture home] Hand-guide the robot (physical guiding buttons on")
    print("[Capture home] the arm) to the position you want as home.")
    input("[Capture home] Press Enter once it's in position ...")

    xyz = get_tcp_xyz(franka)
    joints = franka.get_joint()
    print(f"[Capture home] TCP xyz: {np.round(xyz, 4).tolist()}")
    print(f"[Capture home] Joint angles (for reference only, not reusable "
          f"by this velocity-only controller): {np.round(joints, 4).tolist()}")
    print("[Capture home] Use this next time as:")
    print(f"[Capture home]   --home-xyz {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}")
    print("[Capture home] ================================================\n")


def manual_position_test(franka):
    """
    Verification procedure: hand-guide the robot to a new position, capture
    it, have the robot move back to that captured position automatically
    (via the closed-loop mover above), then return to the original start.
    """
    print("\n[Manual test] ================================================")
    start_xyz = get_tcp_xyz(franka)
    print(f"[Manual test] Initial xyz: {np.round(start_xyz, 4).tolist()}")

    print("[Manual test] Hand-guide the robot to a new position (physical")
    print("[Manual test] guiding buttons), then release.")
    input("[Manual test] Press Enter once moved ...")

    captured_xyz = get_tcp_xyz(franka)
    print(f"[Manual test] Captured xyz: {np.round(captured_xyz, 4).tolist()}")

    input("[Manual test] Press Enter to auto-move back to captured xyz ...")
    move_to_target_closed_loop(franka, captured_xyz, max_speed=0.02)

    input("[Manual test] Press Enter to return to the initial xyz ...")
    move_to_target_closed_loop(franka, start_xyz, max_speed=0.02)
    print("[Manual test] Passed -- starting teleop.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="172.16.0.2")
    parser.add_argument("--recover", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-franka", action="store_true")
    parser.add_argument("--no-robot-motion", action="store_true")
    parser.add_argument("--mode", choices=["analog", "digital"], default="digital",
                         help="'digital' (default): fixed constant speed once past the deadband, "
                              "like a keyboard key. 'analog': speed scales with how hard you push.")
    parser.add_argument("--capture-home", action="store_true",
                         help="One-time helper: hand-guide the robot to the posture you want as "
                              "the fixed starting position, then print its TCP xyz so you can "
                              "reuse it via --home-xyz on future runs. Exits after printing.")
    parser.add_argument("--home-xyz", type=float, nargs=3, default=None,
                         metavar=("X", "Y", "Z"),
                         help="TCP [x, y, z] (metres) to move to (closed-loop) before starting "
                              "teleop, so every session starts from the same fixed position.")
    parser.add_argument("--manual-test", action="store_true",
                         help="Interactive verification: hand-guide the robot to a new position, "
                              "capture it, auto-move back to it, then return to the start -- "
                              "verifying that reading and commanding positions agree.")
    parser.add_argument("--pose-window", action="store_true",
                         help="Open a live window continuously showing the current TCP [x,y,z].")
    parser.add_argument("--pose-window-hz", type=float, default=10.0,
                         help="Refresh rate (Hz) for --pose-window (default 10).")
    parser.add_argument("--print-pose-hz", type=float, default=5.0,
                         help="Continuously print the current TCP [x,y,z] to the console at "
                              "this rate (Hz), console-only (no GUI window). 0 (default) "
                              "disables this -- only the one-time startup pose is printed. "
                              "Try e.g. 2 for twice a second.")

    # Useful tuning knobs.
    parser.add_argument("--transition-up", type=float, default=0.60)
    parser.add_argument("--transition-down", type=float, default=0.05)

    args = parser.parse_args()
    print(f"[Main] Input mode: {args.mode}")

    ctx = mp.get_context("spawn")

    shared = ctx.Array("d", SHARED_SIZE, lock=False)
    connected_event = ctx.Event()
    stop_event = ctx.Event()
    status_q = ctx.Queue()
    read_count = ctx.Value("i", 0)

    write_shared(shared, zero_command())

    proc = ctx.Process(
        target=spacemouse_process,
        args=(shared, connected_event, stop_event, status_q, read_count, args.mode),
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
    pose_window = None

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

            if args.capture_home:
                capture_home_position(franka)
                return  # one-shot helper -- exit without starting teleop

            if args.home_xyz is not None:
                move_to_target_closed_loop(franka, np.array(args.home_xyz), max_speed=0.02)

            start_xyz = get_tcp_xyz(franka)
            print(f"[Main] Current TCP xyz at startup: {np.round(start_xyz, 4).tolist()}")

            if args.manual_test:
                manual_position_test(franka)

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

        pose_window = None
        if args.pose_window and franka is not None:
            try:
                pose_window = PoseWindow(refresh_hz=args.pose_window_hz)
            except Exception as e:
                print(f"[Main] Could not open pose window ({e}).")

        print("[Main] Teleop running with native Cartesian velocity + cosine smoother")
        print("[Main] Press Ctrl+C to stop")
        print("")

        last_print = 0.0
        last_count = 0
        last_time = time.time()

        last_pose_print_time = 0.0
        pose_print_dt = (1.0 / args.print_pose_hz) if args.print_pose_hz > 0 else None

        while True:
            time.sleep(0.05)

            now = time.monotonic()
            if pose_window is not None:
                if not pose_window.is_open():
                    print("[Main] Pose window closed (ESC pressed).")
                    break
                pose_window.maybe_update(franka, now)

            if pose_print_dt is not None and franka is not None and (now - last_pose_print_time) >= pose_print_dt:
                last_pose_print_time = now
                xyz = get_tcp_xyz(franka)
                print(f"[Pose] xyz={np.round(xyz, 4).tolist()}")

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

        if pose_window is not None:
            pose_window.close()

        stop_event.set()

        if proc.is_alive():
            proc.join(timeout=2.0)

        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1.0)

        print("[Main] Stopped")


if __name__ == "__main__":
    main()
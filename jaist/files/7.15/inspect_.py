#!/usr/bin/env python3
"""
test_native_cartesian_velocity_ramp.py

Native Cartesian velocity test with smooth cosine ramp.

Why:
    Sending CartesianVelocities as a step:
        0 -> speed
    can trigger:
        cartesian_motion_generator_acceleration_discontinuity

This test sends:
    zero -> smooth ramp up -> hold -> smooth ramp down -> zero

Run:
    python3 test_native_cartesian_velocity_ramp.py --ip 172.16.0.2 --recover --axis x --speed 0.005

Try:
    python3 test_native_cartesian_velocity_ramp.py --axis x --speed 0.005
    python3 test_native_cartesian_velocity_ramp.py --axis y --speed 0.005
    python3 test_native_cartesian_velocity_ramp.py --axis z --speed 0.003
    python3 test_native_cartesian_velocity_ramp.py --axis yaw --speed 0.005
"""

import argparse
import math
import time

from pylibfranka import (
    Robot,
    RealtimeConfig,
    ControllerMode,
    CartesianVelocities,
)


def smooth_ramp_01(t, T):
    """
    0 -> 1 with zero slope at both ends.
    """
    if t <= 0.0:
        return 0.0

    if t >= T:
        return 1.0

    return 0.5 - 0.5 * math.cos(math.pi * t / T)


def make_vel(axis, speed):
    vx = 0.0
    vy = 0.0
    vz = 0.0
    wz = 0.0

    if axis == "x":
        vx = speed
    elif axis == "y":
        vy = speed
    elif axis == "z":
        vz = speed
    elif axis == "yaw":
        wz = speed
    else:
        raise ValueError("axis must be x, y, z, or yaw")

    return [vx, vy, vz, 0.0, 0.0, wz]


def scale_vel(vel, s):
    return [float(v * s) for v in vel]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="172.16.0.2")
    parser.add_argument("--axis", default="x", choices=["x", "y", "z", "yaw"])
    parser.add_argument("--speed", type=float, default=0.005)

    # Timing
    parser.add_argument("--startup", type=float, default=1.0)
    parser.add_argument("--ramp", type=float, default=2.0)
    parser.add_argument("--hold", type=float, default=1.0)
    parser.add_argument("--stop-hold", type=float, default=1.0)

    parser.add_argument("--recover", action="store_true")
    args = parser.parse_args()

    robot = Robot(args.ip, RealtimeConfig.kIgnore)

    if args.recover:
        print("[Test] automatic_error_recovery...")
        robot.automatic_error_recovery()

    base_vel = make_vel(args.axis, args.speed)

    total_time = args.startup + args.ramp + args.hold + args.ramp + args.stop_hold

    print("[Test] Starting native Cartesian velocity control")
    print(f"[Test] axis={args.axis}, speed={args.speed}")
    print(f"[Test] startup={args.startup}, ramp={args.ramp}, hold={args.hold}")

    active = robot.start_cartesian_velocity_control(ControllerMode.JointImpedance)

    t0 = time.time()
    last_print = 0.0

    while True:
        state, duration = active.readOnce()
        t = time.time() - t0

        if t < args.startup:
            scale = 0.0
            phase = "startup zero"

        elif t < args.startup + args.ramp:
            tau = t - args.startup
            scale = smooth_ramp_01(tau, args.ramp)
            phase = "ramp up"

        elif t < args.startup + args.ramp + args.hold:
            scale = 1.0
            phase = "hold"

        elif t < args.startup + args.ramp + args.hold + args.ramp:
            tau = t - (args.startup + args.ramp + args.hold)
            scale = 1.0 - smooth_ramp_01(tau, args.ramp)
            phase = "ramp down"

        elif t < total_time:
            scale = 0.0
            phase = "stop hold"

        else:
            cmd = CartesianVelocities([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            cmd.motion_finished = True
            active.writeOnce(cmd)
            break

        vel = scale_vel(base_vel, scale)
        active.writeOnce(CartesianVelocities(vel))

        if t - last_print > 0.5:
            last_print = t
            print(f"[Test] t={t:.2f}, phase={phase}, scale={scale:.3f}, vel={vel}")

    print("[Test] Done")


if __name__ == "__main__":
    main()
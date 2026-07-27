#!/usr/bin/env python3
"""
Check if the Franka robot is connected and responsive via FrankaController
(pylibfranka) interface. Mirrors test_franka.py (which uses the franky-based
FrankaRobot) so the two backends can be compared directly on hardware.

Usage:
    python test_franka_control.py [--ip 172.16.0.2]

    # Also exercise move_tcp_pose / move_joints / set_ee_velocity (moves the
    # real arm a small amount and back). Off by default -- opt in explicitly:
    python test_franka_control.py --move
"""

import argparse
import sys
import time
import traceback

import numpy as np


def check_pylibfranka_import():
    print("[1/6] Checking pylibfranka installation ...")
    try:
        import pylibfranka
        print(f"      pylibfranka found: {pylibfranka.__file__}")
        return True
    except ImportError as e:
        print(f"      FAIL: {e}")
        print("      Install with: pip install pylibfranka")
        return False


def check_connection(robot_ip: str):
    print(f"[2/6] Connecting to Franka at {robot_ip} ...")
    try:
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "hardware"))
        from franka_control import FrankaController
        robot = FrankaController(robot_ip=robot_ip)
        print("      Connection OK")
        return robot
    except Exception as e:
        print(f"      FAIL: {e}")
        traceback.print_exc()
        return None


def check_state(robot):
    print("[3/6] Reading robot state ...")
    try:
        # joints = robot.get_joint_angles()
        # assert joints.shape == (7,), f"Expected (7,) joint angles, got {joints.shape}"
        # print(f"      Joint angles (rad): {np.round(joints, 4)}")

        tcp = robot.get_tcp_pose()
        assert tcp.shape == (6,), f"Expected (6,) TCP pose, got {tcp.shape}"
        print(f"      TCP pose [x,y,z,rx,ry,rz]: {np.round(tcp, 4)}")
        return True
    except Exception as e:
        print(f"      FAIL: {e}")
        traceback.print_exc()
        return False


def check_ft_sensor(robot):
    print("[4/6] Reading built-in force/torque estimate ...")
    try:
        wrench_base = robot.get_ee_wrench(frame="base")
        assert wrench_base.shape == (6,), f"Expected (6,) wrench, got {wrench_base.shape}"
        print(f"      EE wrench, base frame  [Fx,Fy,Fz,Tx,Ty,Tz]: {np.round(wrench_base, 3)}")

        # wrench_ee = robot.get_ee_wrench(frame="ee")
        # print(f"      EE wrench, EE frame    [Fx,Fy,Fz,Tx,Ty,Tz]: {np.round(wrench_ee, 3)}")

        # tau_ext = robot.get_joint_external_torques()
        # assert tau_ext.shape == (7,), f"Expected (7,) joint torques, got {tau_ext.shape}"
        # print(f"      Joint ext. torques (Nm): {np.round(tau_ext, 3)}")
        return True
    except Exception as e:
        print(f"      FAIL: {e}")
        traceback.print_exc()
        return False


def check_move_tcp_pose(robot, delta: float, target: np.ndarray, 
                        velocity: float, acceleration: float, reverse: bool = True):
    """Point-to-point Cartesian move (blocking) up by `delta` metres in Z, then back."""
    print("[5a/6] Testing move_tcp_pose (point-to-point) ...")
    try:
        start = robot.get_tcp_pose()
        print(f"force = {robot.get_ee_wrench(frame='base')}")
        if target is None:
            target = start.copy()
            target[2] += delta  # +Z, straight up
        else:
            assert target.shape == (6,), f"Expected (6,) target pose, got {target.shape}"
            

        robot.move_tcp_pose(target, velocity=velocity, acceleration=acceleration)
        reached = robot.get_tcp_pose()
        print(f"      Target Z {target[2]:.4f}  ->  reached Z {reached[2]:.4f}")
        if reverse:
            robot.move_tcp_pose(start, velocity=velocity, acceleration=acceleration)
            back = robot.get_tcp_pose()
            print(f"      Returned to Z {back[2]:.4f} (start was {start[2]:.4f})")
        return True
    except Exception as e:
        print(f"      FAIL: {e}")
        traceback.print_exc()
        return False


def check_set_ee_velocity(robot, speed: float, duration: float, hold_dt: float,
                           max_vel: float, max_ang_vel: float):
    """Hold a constant +Z EE velocity for `duration`s, then the reverse, then stop."""
    print("[5d/6] Testing set_ee_velocity (real-time velocity control) ...")
    try:
        start = robot.get_tcp_pose()
        steps = max(1, int(duration / hold_dt))

        for _ in range(steps):
            robot.set_ee_velocity([0.0, 0.0, speed], max_vel=max_vel, max_ang_vel=max_ang_vel)
            time.sleep(hold_dt)
        mid = robot.get_tcp_pose()
        print(f"      Held +Z velocity {speed:.3f} m/s for {duration:.2f}s: Z {start[2]:.4f} -> {mid[2]:.4f}")

        for _ in range(steps):
            robot.set_ee_velocity([0.0, 0.0, -speed], max_vel=max_vel, max_ang_vel=max_ang_vel)
            time.sleep(hold_dt)
        back = robot.get_tcp_pose()
        print(f"      Held -Z velocity for {duration:.2f}s: Z {mid[2]:.4f} -> {back[2]:.4f} (start was {start[2]:.4f})")
        return True
    except Exception as e:
        print(f"      FAIL: {e}")
        traceback.print_exc()
        return False
    finally:
        # Persistent velocity-servo session must be closed even on failure --
        # FrankaController raises on move_tcp_pose/move_joints if a session
        # is left open.
        try:
            robot.stop()
        except Exception:
            pass


def check_move_joints(robot, delta: float, velocity: float, acceleration: float):
    """Small move of the wrist joint (joint 7 -- safest to nudge), then back."""
    print("[5c/6] Testing move_joints ...")
    try:
        start = robot.get_joint_angles()
        target = start.copy()
        target[-1] += delta

        robot.move_joints(target, velocity=velocity, acceleration=acceleration)
        reached = robot.get_joint_angles()
        print(f"      Target joint7 {target[-1]:.4f}  ->  reached {reached[-1]:.4f}")

        robot.move_joints(start, velocity=velocity, acceleration=acceleration)
        back = robot.get_joint_angles()
        print(f"      Returned to joint7 {back[-1]:.4f} (start was {start[-1]:.4f})")
        return True
    except Exception as e:
        print(f"      FAIL: {e}")
        traceback.print_exc()
        return False


def check_error_recovery(robot):
    print("[6/6] Testing error recovery call ...")
    try:
        robot.recover()
        print("      recover_from_errors() OK")
        return True
    except Exception as e:
        print(f"      FAIL: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Check Franka robot connection (pylibfranka backend).")
    parser.add_argument("--ip", default="172.16.0.2", help="Franka FCI IP address")
    parser.add_argument("--move", action="store_true",
                         help="Also exercise move_tcp_pose/move_joints/set_ee_velocity "
                              "(moves the real arm a small amount and back). "
                              "Requires interactive confirmation unless --yes is given.")
    parser.add_argument("--yes", action="store_true",
                         help="Skip the interactive confirmation before moving (use with care).")
    parser.add_argument("--delta", type=float, default=0.02,
                         help="Cartesian test displacement in metres (default 0.04 = 4cm).")
    parser.add_argument("--target", type=float, nargs=6, default=None, 
                        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
                         help="Optional target pose for move_tcp_pose test (6 floats: x y z rx ry rz). If not given, will move +delta in Z from current pose.")
    parser.add_argument("--joint_delta", type=float, default=0.5,
                         help="Joint test displacement in radians (default 0.5).")
    parser.add_argument("--velocity", '-v', type=float, default=0.01, help="Test move velocity.")
    parser.add_argument("--acceleration", '-a', type=float, default=0.01, help="Test move acceleration.")
    parser.add_argument("--ee_velocity", type=float, default=0.01,
                         help="set_ee_velocity test speed in m/s.")
    parser.add_argument("--ee_velocity_duration", type=float, default=1.0,
                         help="How long to hold each velocity direction (s).")
    parser.add_argument("--ee_velocity_dt", type=float, default=0.05,
                         help="Interval between set_ee_velocity calls (s); "
                              "must stay under the 0.5s watchdog timeout.")
    parser.add_argument("--reverse", action="store_true",
                         help="If set, move_tcp_pose will move back to start after reaching target.")
    args = parser.parse_args()

    print("=" * 50)
    print(" Franka Connection Check (pylibfranka / FrankaController)")
    print("=" * 50)

    results = {}

    results["pylibfranka"] = check_pylibfranka_import()
    if not results["pylibfranka"]:
        print("\nAborting: pylibfranka not available.")
        sys.exit(1)

    robot = check_connection(args.ip)
    results["connection"] = robot is not None
    if robot is None:
        print("\nAborting: could not connect to robot.")
        sys.exit(1)
    stop_flag = {"stop": False}
    results["ft_sensor"] = check_ft_sensor(robot)
    try:
        while True:
            results["state"] = check_state(robot)
            
            if not (results["state"] and results["ft_sensor"]):
                print("\nAborting: failed to read robot state or FT sensor.")
                stop_flag["stop"] = True
                break
            time.sleep(1.0)  # Polling interval; adjust as needed
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Stopping the loop.")
        stop_flag["stop"] = True
            
    results["state"] = check_state(robot)
    results["ft_sensor"] = check_ft_sensor(robot)
    if args.move:
        if not args.yes:
            print(f"\nAbout to move the real Franka arm at {args.ip}:")
            # print(f"  - Cartesian +{args.delta*1000:.0f}mm in Z and back (move_tcp_pose)")
            # print(f"  - Joint 7 +{args.joint_delta:.3f} rad and back (move_joints)")
            # print(f"  - Hold {args.ee_velocity:.3f} m/s in +Z then -Z for {args.ee_velocity_duration:.2f}s each (set_ee_velocity)")
            confirm = input("Proceed? [y/N] ").strip().lower()
            if confirm != "y":
                print("Aborted by user.")
                robot.disconnect()
                sys.exit(1)

        target = np.array(args.target) if args.target is not None else None
        results["move_tcp_pose"] = check_move_tcp_pose(
            robot, args.delta, target, args.velocity, args.acceleration, reverse=args.reverse)
        # results["move_joints"] = check_move_joints(
        #     robot, args.joint_delta, args.velocity, args.acceleration)
        # results["set_ee_velocity"] = check_set_ee_velocity(
        #     robot, args.ee_velocity, args.ee_velocity_duration, args.ee_velocity_dt,
        #     args.velocity, args.acceleration)
    else:
        print("\n[5/6] Skipping motion tests (pass --move to exercise "
              "move_tcp_pose/move_joints/set_ee_velocity).")

    results["recovery"] = check_error_recovery(robot)

    robot.disconnect()

    print("\n" + "=" * 50)
    all_ok = all(results.values())
    for check, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {check:<16} : {status}")
    print("=" * 50)
    print("Overall:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

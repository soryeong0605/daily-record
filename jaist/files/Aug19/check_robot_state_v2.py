"""
check_robot_state.py - 로봇을 전혀 움직이지 않고 연결/상태/에러만 확인.

FCI가 꺼져있거나, 로봇이 reflex 에러 상태로 남아있거나, 브레이크가
잠겨있는 등의 문제를 실제 움직임 없이 미리 확인하는 용도.
"""

import sys

from franky import Robot

FRANKA_IP = "172.16.0.2"

print(f"=== {FRANKA_IP} 연결 시도 ===")
try:
    robot = Robot(FRANKA_IP)
    print("연결 성공.")
except Exception as e:
    print(f"연결 실패: {e}")
    print()
    print("확인할 것:")
    print("  1. Franka Desk 웹 화면에서 FCI(Franka Control Interface)가")
    print("     'Activate' 상태인지 확인")
    print("  2. 조인트 브레이크가 풀려있는지 확인")
    print("  3. 네트워크 연결 (ping 172.16.0.2) 확인")
    sys.exit(1)

print()
print("=== Robot State ===")
try:
    state = robot.state
    print("robot_mode:", state.robot_mode)
    print("current_errors:", state.current_errors)
    print("last_motion_errors:", state.last_motion_errors)
except Exception as e:
    print("state read failed:", e)

print()
print("=== Cartesian State (위치/속도) ===")
try:
    cart_state = robot.current_cartesian_state
    pose = cart_state.pose.end_effector_pose.translation
    vel = cart_state.velocity.end_effector_twist.linear
    print("pose (xyz):", pose)
    print("velocity (linear, m/s):", vel)
except Exception as e:
    print("cartesian state read failed:", e)

print()
print("=== recover_from_errors() 시도 ===")
try:
    robot.recover_from_errors()
    print("성공 -- 로봇이 명령을 받을 준비가 됨.")
except Exception as e:
    print("실패:", e)

print()
print("=== 결과 요약 ===")
try:
    state = robot.state
    if len(state.current_errors) == 0 and len(state.last_motion_errors) == 0:
        print("에러 없음. 로봇은 정상 상태로 보임.")
    else:
        print("에러 있음! 위 current_errors / last_motion_errors 내용을 확인하세요.")
except Exception:
    print("상태를 다시 못 읽어서 요약 불가 -- 위 로그를 직접 확인하세요.")

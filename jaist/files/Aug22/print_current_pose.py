"""
print_current_pose.py
======================
지금 프랑카가 있는 자세의 관절 각도(7개)를 출력함.

사용법:
1. Desk에서 조인트 잠금 해제 (Unlock joints), FCI ON 확인
2. 손으로 로봇을 원하는 "에피소드 시작 자세"로 직접 움직여서 위치잡기
   (조인트가 unlock 상태면 손으로 잡고 움직일 수 있음)
3. 이 스크립트 실행 -> 출력된 관절 각도 7개를 복사해서
   go_home.py의 HOME_JOINTS에 붙여넣기
"""

from franky import Robot

FRANKA_IP = "172.16.0.2"

robot = Robot(FRANKA_IP)

state = robot.current_joint_state
print("현재 관절 각도 (radian):")
print(list(state.position))

print("\ngo_home.py에 붙여넣을 형태:")
print("HOME_JOINTS = " + str(list(state.position)))

pose = robot.current_cartesian_state.pose
print("\n(참고) 현재 EE Cartesian 위치:", pose.end_effector_pose.translation)

"""
go_home.py
==========
매 에피소드 시작 전에 실행 -- 프랑카를 저장해둔 관절 각도로,
그리퍼를 PWM (0,0,0) 원점으로 되돌림.

사용법:
1. 아래 HOME_JOINTS를 print_current_pose.py로 뽑은 값으로 채워넣기
   (한 번만 하면 됨 -- 원하는 시작 자세를 손으로 잡아서 위치잡고 기록)
2. 매 에피소드 시작 전: python go_home.py
3. "Done. 원점 도착." 뜨면 그때부터 녹화 시작
"""

import os
import sys
import time

from franky import Robot, JointMotion

FRANKA_IP = "172.16.0.2"

# TODO: print_current_pose.py로 뽑은 값으로 바꾸세요 (지금은 예시 숫자임)
HOME_JOINTS = [-0.10143285989761353, -0.2382602095603943, 0.12125885486602783,
               -2.0079030990600586, 0.015823200345039368, 1.7655394077301025,
               0.8588652610778809]

FLOWBOT_PROJECT_DIR = os.path.expanduser("~/Desktop/franka_soryeong/learning")
FLOWBOT_SERIAL_PORT = "/dev/ttyACM0"
sys.path.insert(0, FLOWBOT_PROJECT_DIR)


def go_home_franka():
    print(f"[Franka] Connecting to {FRANKA_IP} ...")
    robot = Robot(FRANKA_IP)
    robot.relative_dynamics_factor = 0.1   # 원점 복귀는 안전하게 느린 속도로
    robot.recover_from_errors()

    print("[Franka] 홈 위치로 이동 중...")
    motion = JointMotion(HOME_JOINTS)
    robot.move(motion)   # 동기(synchronous) 호출 -- 도착할 때까지 여기서 대기
    print("[Franka] 홈 위치 도착.")


def go_home_gripper():
    from learning.hardware import flowbot as flowbot_module

    print(f"[Gripper] Opening flowbot on {FLOWBOT_SERIAL_PORT} ...")
    fb = flowbot_module.flowbot(
        serial_port=FLOWBOT_SERIAL_PORT,
        pwm_min=0,
        pwm_max=26,
        enable_plot=False,
        frequency=30.0,
        max_pos_speed=50,
        pressure_model="linear",
    )
    fb.start()
    fb.reset()
    print("[Gripper] 원점 복귀 (PWM 0,0,0).")
    time.sleep(0.5)
    fb.stop()


if __name__ == "__main__":
    go_home_franka()
    go_home_gripper()
    print("\nDone. 원점 도착. 이제 에피소드를 시작하세요.")

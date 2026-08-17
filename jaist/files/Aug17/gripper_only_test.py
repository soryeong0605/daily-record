"""
gripper_only_test.py
=====================
프랑카, 스페이스마우스, OptiTrack 전부 빼고 그리퍼(flowbot)만 단독으로
움직여보는 스크립트.
"""

import os
import sys
import time

import numpy as np

PROJECT_DIR = os.path.expanduser(
    "~/woonjoo/GitHub-RobotArm/learning_8.11"
)
SERIAL_PORT = "/dev/ttyACM0"

sys.path.insert(0, PROJECT_DIR)

from learning.hardware import flowbot as flowbot_module

print(f"[Test] Opening flowbot on {SERIAL_PORT} ...")
fb = flowbot_module.flowbot(
    serial_port=SERIAL_PORT,
    pwm_min=1,
    pwm_max=25,
    enable_plot=False,
    frequency=30.0,
    max_pos_speed=50.0,
    pressure_model="linear",
)
fb.start()
print("[Test] flowbot connected. 3초 후 움직임 시작...")

inside = fb.ws.is_inside_workspace(fb.pc, fb.tri)
print(f"[진단] 시작 위치 {fb.pc} 가 워크스페이스 안쪽인가? -> {inside}")
if not inside:
    print("[진단] False -> 이게 움직임이 고정되는 원인입니다.")

time.sleep(3)

try:
    axes = [
        ("+X", [1.0, 0.0, 0.0]),
        ("-X", [-1.0, 0.0, 0.0]),
        ("+Y", [0.0, 1.0, 0.0]),
        ("-Y", [0.0, -1.0, 0.0]),
    ]

    for label, direction in axes:
        direction = np.array(direction, dtype=float)
        print(f"\n[Test] {label} 방향으로 1초간 이동...")
        t_end = time.time() + 1.0
        while time.time() < t_end:
            pwm = fb.step(direction)
            print(f"  pc={np.round(fb.pc, 1)}  pwm={pwm}")
        print(f"[Test] {label} 완료. pc={fb.pc}")
        time.sleep(1.0)

finally:
    print("\n[Test] 안전 종료 중...")
    fb.stop()
    print("[Test] 종료 완료.")

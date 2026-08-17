"""
gripper_only_test2.py
======================
그리퍼 단독 테스트 v2 -- 워크스페이스 경계 판정을 "현재위치에서 탐색"이 아니라
"워크스페이스 중심에서 탐색"으로 바꿔서, 시작점이 얇은/경계 영역에 있을 때
움직임이 고정되는 문제를 우회함.
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
print("[Test] flowbot connected.")

inside = fb.ws.is_inside_workspace(fb.pc, fb.tri)
print(f"[진단] 시작 위치 {fb.pc} 가 워크스페이스 안쪽인가? -> {inside}")

center = np.mean(fb.ws.P, axis=0)
print(f"[진단] 워크스페이스 중심(추정): {np.round(center, 1)}")

def constrain_from_center(pc, pc_proposed, policy="backtrack"):
    if fb.ws.is_inside_workspace(pc_proposed, fb.tri):
        return pc_proposed
    d = pc_proposed - center
    norm = float(np.linalg.norm(d))
    if norm < 1e-9:
        return pc
    steps = 30
    for a in np.linspace(1.0, 0.0, steps):
        cand = center + a * d
        if fb.ws.is_inside_workspace(cand, fb.tri):
            return cand
    return pc

fb.apply_workspace_constraint = constrain_from_center

print("[Test] 3초 후 움직임 시작 (수정된 워크스페이스 판정 적용됨)...")
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

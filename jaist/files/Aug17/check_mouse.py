"""
check_mouse.py - SpaceMouse raw 입력값을 콘솔에 실시간으로 찍어주는
아주 짧은 테스트 스크립트. 배터리/연결 문제인지 확인용.

버튼을 안 눌러도 손을 살짝 얹기만 해도 x/y/z/yaw 값이 미세하게라도
바뀌면 정상. 계속 정확히 0.000, 0.000, 0.000, 0.000만 나오면
배터리 방전이나 연결 문제일 가능성이 높음.

Ctrl+C로 종료.
"""

import time
import pyspacemouse

with pyspacemouse.open() as device:
    print("SpaceMouse connected. 흔들거나 눌러보세요. (Ctrl+C 종료)")
    try:
        while True:
            state = device.read()
            if state is None:
                print("state: None (읽기 실패)")
            else:
                print(f"x={state.x:+.3f} y={state.y:+.3f} z={state.z:+.3f} "
                      f"yaw={state.yaw:+.3f} buttons={list(state.buttons)}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n종료")

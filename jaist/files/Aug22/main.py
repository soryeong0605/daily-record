"""
main.py - franka_soryeong/combined_teleop/main.py

franka_teleop과 learning(flowbot)을 SpaceMouse 1번 버튼으로 전환하며
번갈아 구동. 두 쪽 다 라이브러리를 통일하지 않고(franka=pyspacemouse,
flowbot=libspnav 기반), 한 번에 한쪽만 디바이스를 열어서 씀.

- franka는 백그라운드 스레드(tasks/franka_task.py)에서 돌고, 자체적으로
  active_mode에 따라 SpaceMouse를 열고 닫음. 1번 버튼 누르면 정지 후
  (실측 속도 확인) flowbot으로 전환.
- flowbot은 matplotlib 그래프(2x2 패널)를 써서 메인 스레드에서 돌아야
  하므로, 이 main.py의 메인 루프가 flowbot 쪽을 직접 담당함. flowbot
  쪽 SpaceMouse도 active_mode=="flowbot"일 때만 열림. 1번 버튼 누르면
  즉시(정지 대기 없이) franka로 전환 -- flowbot은 마지막 PWM을 유지한
  채로 넘어가면 되니까 별도 정지 절차 필요 없음.

8/19 추가: 그래프 창에 포커스를 준 상태에서 'r' 키를 누르면 흡착이
순간(약 0.1초) 풀렸다가 자동으로 다시 흡착 상태로 복귀함 (아두이노
펌웨어의 "release" 명령 -- fb.release() 그대로 사용). 물건을 선반
위에서 내려놓을 때 이 키를 누르면 됨.

8/22 추가: 모방학습용 데이터수집(zarr) 연동.
터미널에 포커스를 준 상태에서:
    'c' -> 에피소드 레코딩 시작
    's' -> 에피소드 레코딩 종료 + zarr에 저장
    'q' -> 프로그램 종료
franka 모드/flowbot 모드 어느 쪽이든 is_recording이 켜져있으면 각자 스레드/루프에서
알아서 스텝을 기록함 (franka_task.py, 이 파일의 flowbot 분기 양쪽 다 참여).
"""

import os
import select
import sys
import termios
import time
import tty

import numpy as np
import matplotlib.pyplot as plt

from shared import SharedData, RECORD_DT
from tasks.franka_task import FrankaTask
from data_recorder import EpisodeBuffer, create_zarr_dataset, save_episode, save_episode_csv
from realsense_camera import MultiRealSenseCamera

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(FILE_DIR)
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, os.path.join(PARENT_DIR, "learning", "flowbot"))
from learning.hardware import flowbot as flowbot_module
from learning.hardware.spacemouse import _build_spacemouse

FRANKA_IP = "172.16.0.2"

DEADZONE = 0.1
IDLE_REFRESH_DT = 0.1   # franka 활성 중, 그래프 창이 안 얼어붙게 가끔 갱신

# ===== 데이터수집 설정 =====
OUTPUT_DIR = os.path.join(FILE_DIR, "data", "demo_data")
# TODO: realsense-viewer 켜고 두 카메라 시리얼 넘버 확인해서 여기 순서대로 넣기
# (첫 번째 = camera_0, 두 번째 = camera_1)
CAMERA_SERIALS = [
    "051222061185",  # camera_0 (D415)
    "827112072398",  # camera_1 (D435)
]
CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS = 640, 480, 30


def _read_key_nonblocking():
    """터미널에서 입력 대기 없이 키 하나 읽기. 없으면 None."""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def main():
    shared = SharedData()

    # =========================
    # 카메라 시작 (2대, 시리얼넘버로 고정 지정)
    # =========================
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    camera_names = [f"camera_{i}" for i in range(len(CAMERA_SERIALS))]
    print("\n카메라 초기화 중...")
    camera_manager = MultiRealSenseCamera(
        serial_numbers=CAMERA_SERIALS,
        width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS,
    )
    camera_manager.start()
    print("카메라 준비 완료.\n")

    # =========================
    # zarr 데이터셋 준비 + 이번 세션의 에피소드 버퍼
    # =========================
    zarr_root = create_zarr_dataset(
        OUTPUT_DIR, camera_names=camera_names,
        image_shape=(CAMERA_HEIGHT, CAMERA_WIDTH, 3),
    )
    shared.episode_buffer = EpisodeBuffer(camera_names=camera_names)

    # =========================
    # flowbot 준비 (그래프는 항상 켜둠 -- franka 활성 중엔 갱신만 멈춤)
    # =========================
    fb = flowbot_module.flowbot(
        serial_port="/dev/ttyACM0",
        pwm_min=0,
        pwm_max=26,
        enable_plot=True,
        frequency=30.0,
        max_pos_speed=50,
        pressure_model="linear",
    )
    fb.start()
    fb.pl.highlight_pwm_corners(fb.axes, fb.flowbot, pwm_max=26)

    # =========================
    # franka 백그라운드 스레드 시작 (기본 모드: franka)
    # =========================
    franka_task = FrankaTask(
        shared, ip=FRANKA_IP, dynamics_factor=0.05, camera_manager=camera_manager,
    )
    franka_task.start()

    sm_fb = None          # flowbot용 SpaceMouse, active일 때만 open
    last_button0 = False
    last_release_button = False

    # 버튼1(모드전환)을 누른 순간 바로 전환하지 않고, COMBO_WINDOW 동안
    # 버튼14가 같이 눌리는지 관찰함. 그 사이 버튼14가 같이 눌리면
    # "굽은 상태 유지 + 흡착만 해제"(release)로 처리하고, 안 눌리면
    # 원래대로 franka 모드로 전환함.
    COMBO_WINDOW = 0.15  # sec
    button0_press_time = None
    switch_done_this_press = False
    combo_active = False  # 이번 "두 버튼 동시 눌림" 구간에서 이미 release를 보냈는지

    print("[Main] franka 모드로 시작. SpaceMouse 1번 버튼으로 전환. "
          "그리퍼 모드에서 14번 버튼 -> (0,0,0)으로 놓기, "
          "1번+14번 동시 -> 굽은 상태 유지한 채 흡착만 해제(release).")
    print("[Main] 터미널에 포커스 준 상태에서 'c'=레코딩 시작, 's'=레코딩 종료+저장, 'q'=종료")

    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    try:
        while shared.running and plt.fignum_exists(fb.fig.number):

            # ---- 키보드: 레코딩 시작/종료/종료 ----
            key = _read_key_nonblocking()
            if key in ("c", "C"):
                if not shared.is_recording:
                    shared.episode_buffer.reset()
                    with shared.lock:
                        shared.is_recording = True
                    print("\n>>> RECORDING STARTED <<<\n")
            elif key in ("s", "S"):
                if shared.is_recording:
                    with shared.lock:
                        shared.is_recording = False
                    n_steps = len(shared.episode_buffer)
                    if n_steps > 0:
                        ep_data = shared.episode_buffer.to_dict()
                        ep_id = save_episode(zarr_root, ep_data)
                        shared.episode_count += 1

                        csv_dir = os.path.join(OUTPUT_DIR, "episode_csv")
                        os.makedirs(csv_dir, exist_ok=True)
                        csv_path = os.path.join(csv_dir, f"episode_{ep_id:04d}.csv")
                        save_episode_csv(ep_data, csv_path)

                        print(f"\n>>> Episode {ep_id} SAVED ({n_steps} steps) "
                              f"-- total episodes: {shared.episode_count}\n")
                    else:
                        print("\n⚠️  기록된 스텝이 없어서 저장 안 함\n")
            elif key in ("q", "Q"):
                print("\n[Main] 종료 요청 -> 정리 중...")
                with shared.lock:
                    shared.running = False
                break

            with shared.lock:
                mode = shared.active_mode

            if mode == "flowbot":
                if sm_fb is None:
                    sm_fb = _build_spacemouse(os_name="linux")
                    sm_fb.start()
                    print("[Main] flowbot active -> SpaceMouse 여는 중...")

                xyz = sm_fb.get_latest_xyz()
                buttons = sm_fb.get_button_status()
                button0 = bool(buttons[0]) if len(buttons) > 0 else False   # 물리버튼 "1"
                release_button = bool(buttons[1]) if len(buttons) > 1 else False  # 물리버튼 "14"

                now = time.monotonic()
                both_down = button0 and release_button

                # ---- 조합(1+14): 굽은 상태 유지 + 흡착만 해제 ----
                if both_down and not combo_active:
                    print("[Main] 1+14 동시 감지 -> release (굽은 상태 유지, 흡착만 해제)")
                    fb.release()
                    combo_active = True
                    # 이번 눌림에서는 모드전환도, (0,0,0) 놓기도 발생시키지 않음
                    button0_press_time = None
                    switch_done_this_press = True
                if not both_down:
                    combo_active = False

                # ---- 버튼1 단독: 짧은 유예시간(COMBO_WINDOW) 동안 버튼14와
                #      같이 눌리는지 지켜본 뒤에만 franka 모드로 전환 ----
                if button0 and not last_button0:
                    button0_press_time = now
                    switch_done_this_press = False

                if (
                    button0
                    and not both_down
                    and not switch_done_this_press
                    and button0_press_time is not None
                    and (now - button0_press_time) >= COMBO_WINDOW
                ):
                    print("[Main] 1번 버튼(단독) 감지 -> franka로 전환 (flowbot 상태 유지)")
                    switch_done_this_press = True
                    sm_fb.stop()
                    sm_fb = None
                    with shared.lock:
                        shared.active_mode = "franka"
                    last_button0 = False
                    last_release_button = False
                    time.sleep(0.05)
                    continue

                if not button0:
                    button0_press_time = None
                    switch_done_this_press = False

                # ---- 버튼14 단독: 기존 (0,0,0)으로 놓기 ----
                release_edge_on = release_button and not last_release_button
                if release_edge_on and not button0:
                    print("[Main] 14번 버튼(단독) 감지 -> PWM (0,0,0)으로 압력 해제")
                    fb.reset()

                last_button0 = button0
                last_release_button = release_button

                if not release_button and not button0:
                    xyz = xyz.copy()
                    xyz[2] = -xyz[2]
                    xyz = np.where(np.abs(xyz) < DEADZONE, 0.0, xyz)
                    fb.step(xyz)   # 여기서 PWM 값이 계속 [PYTHON] Sent: ... 로 출력됨

                fb.update_plot()

                # ---- franka 스레드도 최신 PWM을 참조할 수 있게 broadcast ----
                with shared.lock:
                    shared.latest_pwm = np.asarray(fb.last_pwm, dtype=float)

                # ---- 데이터수집: is_recording이면서 RECORD_DT 이상 지났을 때만 기록 ----
                now2 = time.time()
                with shared.lock:
                    recording = shared.is_recording
                    franka_pose = shared.latest_franka_eef_pose
                    time_to_record = (now2 - shared.last_record_time) >= RECORD_DT
                    if recording and time_to_record:
                        shared.last_record_time = now2
                if recording and time_to_record and franka_pose is not None:
                    op_mode = np.array([1, 1] if both_down else [0, 1], dtype=np.uint8)
                    # franka_task.py와 동일한 고정 shape로 통일 --
                    # action(7,) = [vx,vy,vz,wz, pwm1,pwm2,pwm3]. 그리퍼 구간이므로
                    # velocity 부분은 0(안 움직임), pwm 부분이 실제 명령값.
                    unified_action = np.concatenate(
                        [np.zeros(4, dtype=float), np.asarray(fb.last_pwm, dtype=float)]
                    )
                    # sm_raw(6,) = [x,y,z,yaw,btn1,btn14] -- 이 라이브러리는 yaw가 없어서 0으로,
                    # 버튼도 2개뿐이라 buttons[0]=물리버튼1, buttons[1]=물리버튼14로 매핑.
                    btn1 = float(buttons[0]) if len(buttons) > 0 else 0.0
                    btn14 = float(buttons[1]) if len(buttons) > 1 else 0.0
                    unified_sm_raw = np.array(
                        [xyz[0], xyz[1], xyz[2], 0.0, btn1, btn14], dtype=float
                    )
                    shared.episode_buffer.add(
                        timestamp=now2,
                        franka_eef_pose=franka_pose,
                        gripper_tip_pose=franka_pose,  # TODO: PCC 순기구학 나오면 교체
                        pwm_signals=np.asarray(fb.last_pwm, dtype=float),
                        action=unified_action,
                        sm_raw=unified_sm_raw,
                        operation_mode=op_mode,
                        camera_frames=camera_manager.get_latest_frames(),
                    )

            else:
                # franka 활성 중 -- flowbot 쪽은 아무것도 안 하고 그래프만
                # 가끔 갱신해서 창이 "응답 없음" 상태로 안 보이게만 함.
                if sm_fb is not None:
                    sm_fb.stop()
                    sm_fb = None
                fb.fig.canvas.flush_events()
                time.sleep(IDLE_REFRESH_DT)

    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        with shared.lock:
            shared.running = False
        if sm_fb is not None:
            sm_fb.stop()
        franka_task.join()
        fb.stop()
        camera_manager.stop()
        time.sleep(0.5)
        print(f"\n✅ Done! Collected {shared.episode_count} episodes")
        print(f"Data: {os.path.join(OUTPUT_DIR, 'dataset.zarr')}\n")


if __name__ == "__main__":
    main()
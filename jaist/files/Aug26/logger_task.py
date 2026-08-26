"""
tasks/logger_task.py - franka_soryeong/combined_teleop/tasks/logger_task.py

franka_task.py / main.py의 기존 구동 로직은 건드리지 않고, shared에
노출된 값들(franka_pose, franka_velocity, flowbot_pwm, sm_raw,
active_mode, latest_frames)을 20Hz로 읽어서 CSV + 카메라 이미지(jpg)로
저장하는 독립 스레드.

터미널에 포커스 준 상태에서:
    'c' -> 카메라 스트리밍 요청(camera_request=True) -> camera_task.py가
           초기화 완료(camera_ready=True)할 때까지 대기 -> 그 다음에야
           비로소 에피소드 시작(CSV + 이미지 폴더 생성, 20Hz 기록 시작)
    's' -> 레코딩 종료 + 저장, camera_request=False로 카메라도 같이 정지

동기화 정책 (A안): 카메라가 준비되기 전에는 pose/velocity/pwm 등
아무 것도 기록하지 않음 -- 모든 필드가 "카메라 첫 프레임부터" 함께
시작하도록 해서, 이미지-상태-액션의 시간 정합성을 우선시함.

주의:
- franka 모드 중엔 flowbot_pwm이 "마지막 값" 그대로 찍히고,
  flowbot 모드 중엔 franka_pose/franka_velocity가 "마지막 값" 그대로
  찍힘. 각 경우 실제로 해당 쪽이 정지해있는 상태라 문제 없음
  (의도된 설계 -- shared.py 주석 참고).
- shared.py에 이미 있는 franka/flowbot 제어 로직에는 전혀 관여하지
  않고, 값을 읽기만(read-only) 함. 카메라 제어는 camera_task.py가
  전담하고, 여기선 camera_request 플래그만 세팅/해제함.
"""

import csv
import os
import re
import threading
import time
from datetime import datetime

import cv2

from keypress_listener import KeypressListener

LOG_HZ = 20.0
LOG_DT = 1.0 / LOG_HZ
JPEG_QUALITY = 92


class LoggerTask(threading.Thread):

    def __init__(self, shared, output_dir="episodes"):
        super().__init__()
        self.shared = shared
        self.output_dir = output_dir
        self.kp = KeypressListener()
        self.episode_id = self._find_last_episode_id()
        self.csv_file = None
        self.writer = None
        self.image_dir = None
        self.frame_idx = 0

    def _find_last_episode_id(self):
        """재시작해도 이어서 에피소드 번호를 매기기 위해 기존 파일들을 스캔."""
        if not os.path.isdir(self.output_dir):
            return 0
        max_id = 0
        pattern = re.compile(r"episode_(\d+)_")
        for name in os.listdir(self.output_dir):
            m = pattern.match(name)
            if m:
                max_id = max(max_id, int(m.group(1)))
        return max_id

    def _start_episode(self):
        """카메라 준비가 끝난 뒤 호출됨 -- CSV + 카메라별 이미지 폴더 생성."""
        self.episode_id += 1
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"episode_{self.episode_id:04d}_{timestamp}"

        csv_path = os.path.join(self.output_dir, base_name + ".csv")
        self.csv_file = open(csv_path, "w", newline="")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow([
            "timestamp", "frame_idx", "mode",
            "franka_x", "franka_y", "franka_z",
            "vx", "vy", "vz", "wz",
            "pwm1", "pwm2", "pwm3",
            "sm_x", "sm_y", "sm_z", "sm_yaw", "btn0", "btn_other",
        ])

        self.image_dir = os.path.join(self.output_dir, base_name)
        with self.shared.lock:
            camera_names = list(self.shared.latest_frames.keys())
        for cam_name in camera_names:
            os.makedirs(os.path.join(self.image_dir, cam_name), exist_ok=True)

        self.frame_idx = 0
        print(f"\n>>> RECORDING STARTED -- episode {self.episode_id} -> {csv_path} "
              f"(+ {self.image_dir}/) <<<\n")

    def _stop_episode(self):
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.writer = None
        print(f"\n>>> RECORDING STOPPED -- episode {self.episode_id} saved "
              f"({self.frame_idx} frames) <<<\n")
        self.image_dir = None

    def run(self):
        self.kp.start()
        print("[Logger] 'c'=레코딩 시작(카메라 준비 후), 's'=레코딩 종료+저장 (20Hz)")

        state = "idle"   # "idle" -> "waiting_camera" -> "recording"
        last_log_time = 0.0

        while self.shared.running:

            # =========================
            # 키 입력: c=시작 요청, s=종료
            # =========================
            key = self.kp.get_key()
            if key in ("c", "C") and state == "idle":
                with self.shared.lock:
                    self.shared.camera_request = True
                state = "waiting_camera"
                print("[Logger] 카메라 초기화 대기 중... (준비되면 자동으로 기록 시작)")

            elif key in ("s", "S") and state in ("waiting_camera", "recording"):
                if state == "recording":
                    self._stop_episode()
                with self.shared.lock:
                    self.shared.camera_request = False
                    self.shared.recording = False
                state = "idle"

            # =========================
            # 상태별 처리
            # =========================
            if state == "idle":
                time.sleep(0.02)
                continue

            if state == "waiting_camera":
                with self.shared.lock:
                    ready = self.shared.camera_ready
                if ready:
                    self._start_episode()
                    with self.shared.lock:
                        self.shared.recording = True
                    state = "recording"
                    last_log_time = 0.0
                else:
                    time.sleep(0.05)
                continue

            # state == "recording"
            now = time.time()
            if now - last_log_time < LOG_DT:
                time.sleep(0.005)
                continue
            last_log_time = now

            with self.shared.lock:
                mode = self.shared.active_mode
                pose = self.shared.franka_pose.copy()
                vel = self.shared.franka_velocity.copy()
                pwm = self.shared.flowbot_pwm.copy()
                sm = self.shared.sm_raw.copy()
                frames = {k: v.copy() for k, v in self.shared.latest_frames.items()}

            # ---- 이미지 저장 (카메라별 폴더에 frame_idx.jpg) ----
            for cam_name, frame in frames.items():
                img_path = os.path.join(
                    self.image_dir, cam_name, f"{self.frame_idx:06d}.jpg"
                )
                cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

            # ---- CSV 한 줄 ----
            self.writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                self.frame_idx,
                mode,
                f"{pose[0]:.5f}", f"{pose[1]:.5f}", f"{pose[2]:.5f}",
                f"{vel[0]:.5f}", f"{vel[1]:.5f}", f"{vel[2]:.5f}", f"{vel[3]:.5f}",
                f"{pwm[0]:.2f}", f"{pwm[1]:.2f}", f"{pwm[2]:.2f}",
                f"{sm[0]:.4f}", f"{sm[1]:.4f}", f"{sm[2]:.4f}", f"{sm[3]:.4f}",
                f"{sm[4]:.0f}", f"{sm[5]:.0f}",
            ])

            self.frame_idx += 1

        # =========================
        # 전체 종료 시, 레코딩 중이었으면 안전하게 닫기
        # =========================
        if state == "recording":
            self._stop_episode()
        with self.shared.lock:
            self.shared.camera_request = False
            self.shared.recording = False

        self.kp.stop()

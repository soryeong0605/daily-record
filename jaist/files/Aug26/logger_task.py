import threading
import csv
import os
import re
import time
from datetime import datetime


class LoggerTask(threading.Thread):

    def __init__(self, shared):
        super().__init__()
        self.shared = shared
        self.episode_id = self._find_last_episode_id()
        self.csv_file = None
        self.writer = None

    def _find_last_episode_id(self):
        """
        프로그램을 재시작해도 episodes/ 폴더에 이미 있는 파일들을 보고
        마지막 번호 다음부터 이어서 매기도록 함 (재시작할 때마다
        1번으로 덮어쓰는 것 방지).
        """
        if not os.path.isdir("episodes"):
            return 0

        max_id = 0
        pattern = re.compile(r"episode_(\d+)_")

        for name in os.listdir("episodes"):
            m = pattern.match(name)
            if m:
                max_id = max(max_id, int(m.group(1)))

        return max_id

    def start_episode(self):
        self.episode_id += 1

        os.makedirs("episodes", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = f"episodes/episode_{self.episode_id:04d}_{timestamp}.csv"

        self.csv_file = open(filepath, "w", newline="")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow([
            "timestamp",
            "x", "y", "z",
            "vx", "vy", "vz", "wz"
        ])
        print(f"Episode {self.episode_id} 시작! -> {filepath}")

    def stop_episode(self):
        if self.csv_file:
            self.csv_file.close()
        print(f"Episode {self.episode_id} 종료!")

    def run(self):
        while self.shared.running:

            with self.shared.lock:
                changed = self.shared.recording_changed
                recording = self.shared.recording

            if changed:
                with self.shared.lock:
                    self.shared.recording_changed = False
                if recording:
                    self.start_episode()
                else:
                    self.stop_episode()

            if not recording:
                time.sleep(0.05)
                continue

            with self.shared.lock:
                pose = self.shared.robot_pose
                vel = list(self.shared.sent_velocity)

            if pose is None:
                time.sleep(0.05)
                continue

            # =========================
            # CSV 저장 (TCP 좌표 + 명령 속도)
            # =========================

            self.writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                pose[0], pose[1], pose[2],
                vel[0], vel[1], vel[2], vel[3]
            ])

            time.sleep(0.05)  # 20Hz

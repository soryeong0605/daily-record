import threading
import csv
import os
import time
import cv2
from datetime import datetime


class LoggerTask(threading.Thread):

    def __init__(self, shared):
        super().__init__()
        self.shared = shared
        self.episode_id = 0
        self.frame_id = 0
        self.csv_file = None
        self.writer = None
        self.image_dir = None

    def start_episode(self):
        self.episode_id += 1
        self.frame_id = 0

        episode_dir = f"dataset/episode_{self.episode_id:03d}"
        self.image_dir = f"{episode_dir}/images"
        os.makedirs(self.image_dir, exist_ok=True)

        self.csv_file = open(
            f"{episode_dir}/data.csv", "w", newline=""
        )
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow([
            "timestamp",
            "ef_x", "ef_y", "ef_z",
            "robot_x", "robot_y", "robot_z",
            "l1", "l2", "l3",
            "dl1", "dl2", "dl3",
            "sm_x", "sm_y", "sm_z",
            "sm_roll", "sm_pitch", "sm_yaw"
        ])
        print(f"Episode {self.episode_id} 시작!")

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
                frame = self.shared.frame
                ef = self.shared.end_effector
                pose = self.shared.robot_pose
                length = self.shared.current_length
                delta_l = self.shared.delta_l
                sm = dict(self.shared.spacemouse)

            if frame is None or ef is None or delta_l is None or length is None:
                continue

            # 이미지 저장
            cv2.imwrite(
                f"{self.image_dir}/frame_{self.frame_id:04d}.jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 80]
            )

            # CSV 저장
            self.writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                ef[0], ef[1], ef[2],
                pose[0], pose[1], pose[2],
                length[0], length[1], length[2],
                delta_l[0], delta_l[1], delta_l[2],
                sm["x"], sm["y"], sm["z"],
                sm["roll"], sm["pitch"], sm["yaw"]
            ])

            self.frame_id += 1
            time.sleep(0.2)
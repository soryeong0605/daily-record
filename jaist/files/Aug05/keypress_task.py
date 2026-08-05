import threading
import time

from keypress_listener import KeypressListener


class KeypressTask(threading.Thread):
    """
    터미널 키 입력 담당. display_task.py의 'r' 키 로직(녹화 토글)을
    GUI(cv2 창) 없이 콘솔로 옮긴 버전.

    'r' -> 녹화 시작/종료 토글 (LoggerTask가 이 플래그를 보고 episode 시작/종료)
    's' -> 저속모드 토글 (0.01m/s <-> 원래 속도, FrankaTask가 이 플래그를 봄)
    'q' -> 전체 종료
    """

    def __init__(self, shared):
        super().__init__()
        self.shared = shared
        self.kp = KeypressListener()

    def run(self):
        self.kp.start()
        print("[Keypress] 'r' 녹화 시작/종료, 's' 저속모드, 'q' 종료")

        while self.shared.running:

            key = self.kp.get_key()

            if key in ("r", "R"):
                with self.shared.lock:
                    self.shared.recording = not self.shared.recording
                    self.shared.recording_changed = True
                    recording = self.shared.recording

                if recording:
                    print("[Keypress] 녹화를 시작합니다...")
                else:
                    print("[Keypress] 녹화를 종료하고 저장합니다...")

            elif key in ("s", "S"):
                with self.shared.lock:
                    self.shared.slow_mode = not self.shared.slow_mode
                    slow_mode = self.shared.slow_mode

                if slow_mode:
                    print("[Keypress] 저속모드 ON (0.01m/s)")
                else:
                    print("[Keypress] 저속모드 OFF (원래 속도)")

            elif key in ("q", "Q"):
                print("[Keypress] 종료합니다...")
                with self.shared.lock:
                    self.shared.running = False
                break

            time.sleep(0.02)

        self.kp.stop()

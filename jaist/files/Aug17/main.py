import time

from shared import SharedData

from tasks.franka_task import FrankaTask       # 프랑카만 쓸 때 (기존)
from tasks.combined_task import CombinedTeleopTask  # 프랑카+그리퍼 같이 쓸 때 (신규)
from tasks.logger_task import LoggerTask
from tasks.keypress_task import KeypressTask


# =========================
# ROBOT IP
# =========================

FRANKA_IP = "172.16.0.2"


# =========================
# MAIN
# =========================

def main():

    # =========================
    # SHARED
    # =========================

    shared = SharedData()

    # =========================
    # TASKS
    # =========================
    # SpaceMouse 읽기 + 로봇 명령이 franka_task.py 한 곳에서 같이 돎
    # (franky_teleop.py 원본 구조와 동일 -- 별도 스레드로 안 쪼갬)

    # franka = FrankaTask(shared, ip=FRANKA_IP, mode="analog", dynamics_factor=0.05)
    franka = CombinedTeleopTask(shared, franka_ip=FRANKA_IP, dynamics_factor=0.02)
   # franka = FrankaTask(...)
    # franka = CombinedTeleopTask(shared, franka_ip=FRANKA_IP, dynamics_factor=0.05)
    logger = LoggerTask(shared)
    keypress = KeypressTask(shared)

    tasks = [
        franka,
        logger,
        keypress,
    ]

    # =========================
    # START
    # =========================

    for t in tasks:
        t.start()

    # =========================
    # MAIN LOOP
    # =========================

    try:

        while shared.running:
            time.sleep(0.1)

    except KeyboardInterrupt:

        with shared.lock:
            shared.running = False

    # =========================
    # JOIN
    # =========================

    for t in tasks:
        t.join()


if __name__ == "__main__":
    main()

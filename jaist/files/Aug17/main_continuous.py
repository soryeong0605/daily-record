import time

from shared import SharedData

from tasks.franka_task import FrankaTask
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
    # SpaceMouse 읽기 + 로봇 명령이 franka_task.py 한 곳에서 같이 돎.
    # 그래프는 실시간으로 그리지 않고, franka_task.py가 sessions/ 폴더에
    # 세션 CSV로 저장해두면 이후 plot_session.py로 따로 열어서 봄.

    franka = FrankaTask(shared, ip=FRANKA_IP, dynamics_factor=0.05)
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

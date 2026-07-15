import time

import cv2
import numpy as np

from shared import SharedData

from tasks.capture_task import CaptureTask
from tasks.vision_task import VisionTask
from tasks.tracking_task import TrackingTask
from tasks.display_task import DisplayTask
from tasks.logger_task import LoggerTask
from tasks.transform_task import TransformTask
from tasks.three_continum_solving_task import RobotTask

from tasks.spacemouse_task import SpaceMouseTask
from tasks.robot_arm_task import RobotArmTask
from tasks.peripheralControl_task import SerialTask


# =========================
# TRACKBAR CALLBACK
# =========================


def nothing(x):
    pass


# =========================
# CONTROL PANEL
# =========================

def create_control_panel():

    cv2.namedWindow("Control", cv2.WINDOW_NORMAL)

    cv2.resizeWindow("Control", 400, 420)

    # ===== HSV =====

    cv2.createTrackbar(
        "H min",
        "Control",
        18,
        179,
        nothing
    )

    cv2.createTrackbar(
        "H max",
        "Control",
        80,
        179,
        nothing
    )

    cv2.createTrackbar(
        "S min",
        "Control",
        100,
        255,
        nothing
    )

    cv2.createTrackbar(
        "S max",
        "Control",
        192,
        255,
        nothing
    )

    cv2.createTrackbar(
        "V min",
        "Control",
        121,
        255,
        nothing
    )

    cv2.createTrackbar(
        "V max",
        "Control",
        215,
        255,
        nothing
    )

    # ===== AREA =====

    cv2.createTrackbar(
        "Min Area",
        "Control",
        170,
        200,
        nothing
    )

    cv2.createTrackbar(
        "Max Area",
        "Control",
        1000,
        1500,
        nothing
    )

    # ===== DOT COUNT =====

    cv2.createTrackbar(
        "Dot Count",
        "Control",
        1,
        50,
        nothing
    )

# =========================
# Transformation Matrix
# =========================

camera_yaw_deg = 7.5
camera_tilt_deg = 65.0


# T = np.array([

#     [0.9950, -0.0998, 0,  -32.5],
#     [0.0499, 0.4975, -0.8660,   246],
#     [0.0865, 0.8617, 0.5,  29.6],
#     [0,     0, 0,    1]

# ], dtype=np.float64)

# =========================
# TARGET POINT (WORLD)
# =========================
SharedData.target_point = np.array([
    0,
    15.0,
    185.0
], dtype=np.float64)

# =========================
# MAIN
# =========================

def main():

    # =========================
    # GUI MUST BE MAIN THREAD
    # =========================

    create_control_panel()

    # =========================
    # SHARED
    # =========================

    shared = SharedData()

    # shared.offset_distance = 90.0

    # =========================
    # TASKS
    # =========================

    capture = CaptureTask(shared)
    robot = RobotTask(shared)
    vision = VisionTask(shared)
    tracking = TrackingTask(
        shared,
        capture.intrinsics
    )
    display = DisplayTask(shared)
    logger = LoggerTask(shared)
    transform = TransformTask(shared, camera_yaw_deg, camera_tilt_deg)

    spacemouse = SpaceMouseTask(shared)
    arm = RobotArmTask(shared)
    serial = SerialTask(shared)

    tasks = [
        capture,
        vision,
        tracking,
        transform,
        display,
        logger,
        robot,
        spacemouse,
        arm,
        serial,
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

            key = cv2.waitKey(1)

            if key == 27:
                shared.running = False

            time.sleep(0.001)

    except KeyboardInterrupt:

        shared.running = False

    # =========================
    # JOIN
    # =========================

    for t in tasks:
        t.join()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

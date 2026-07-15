import threading
import time

from dobot_api import (
    DobotApiDashboard,
    DobotApiFeedBack
)

class RobotArmTask(threading.Thread):
    def __init__(self, shared):
        super().__init__()
        self.shared = shared
        ROBOT_IP = "192.168.5.1"
        self.dashboard = DobotApiDashboard(
            ROBOT_IP,
            29999
        )

        self.feedback = DobotApiFeedBack(
            ROBOT_IP,
            30004
        )

        # enable robot
        print(self.dashboard.DisableRobot())
        print(self.dashboard.ClearError())
        print(self.dashboard.Stop())
        print(self.dashboard.ClearError())
        print(self.dashboard.EnableRobot())

        time.sleep(1)

    # ============================================
    # RUN
    # ============================================

    def run(self):
        scale = 0.01
        while self.shared.running:

            # ============================================
            # FEEDBACK
            # ============================================

            data = self.feedback.feedBackData()
            if data is not None:
                try:
                    with self.shared.lock:
                        self.shared.robot_mode = data['RobotMode'][0]
                        self.shared.robot_pose = [
                            float(x)
                            for x in data['ToolVectorActual'][0]
                        ]
                except Exception as e:
                    print(e)

            # ============================================
            # GET SPACEMOUSE
            # ============================================

            with self.shared.lock:
                pose = self.shared.robot_pose
                sm = dict(self.shared.spacemouse)
            x,y,z,rx,ry,rz = pose

            delta_x = 0
            delta_y = 0
            delta_z = 0

            if abs(sm["x"]) > 100:
                delta_x = sm["x"] * scale

            if abs(sm["y"]) > 100:
                delta_y = sm["y"] * scale

            if abs(sm["z"]) > 100:
                delta_z = sm["z"] * scale

            target_x = x + delta_x
            target_y = y - delta_y
            target_z = z - delta_z

            # ============================================
            # SERVO
            # ============================================

            try:
                self.dashboard.ServoP(
                    target_x,
                    target_y,
                    target_z,
                    rx,
                    ry,
                    rz
                )

            except Exception as e:
                print(e)
            time.sleep(0.05)
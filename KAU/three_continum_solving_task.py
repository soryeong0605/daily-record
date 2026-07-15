import threading
import numpy as np
import time


class RobotTask(threading.Thread):

    def __init__(self, shared):
        super().__init__()
        self.shared = shared

        # numerical jacobian step
        self.delta = 1e-5

        # control gain
        self.alpha = 0.2

        # chamber distance from center (mm)
        self.de = 11.66

    # =====================================================
    # POSITION -> POSE
    # =====================================================

    def cal_pose_from_point(self, X, Y, Z):
        r = np.sqrt(X**2 + Y**2)
        den = X**2 + Y**2 + Z**2
        if den < 1e-6:
            return np.zeros(3)
        curv = 2*r / den
        v = 1 - curv*r
        v = np.clip(v, -1.0, 1.0)
        theta = np.arccos(v)
        phi = np.arctan2(Y, X)
        return np.array([
            phi,
            curv,
            theta
        ])

    # =====================================================
    # POSE -> CHAMBER LENGTH
    # =====================================================

    def pose_to_length(self, phi, curv, theta):
        if abs(curv) < 1e-6:
            l = theta
        else:
            l = theta / curv
        l1 = (l - curv*l*self.de*np.cos(-phi + np.pi/2))
        l2 = (l - curv*l*self.de*np.cos(-phi + 4*np.pi/3 + np.pi/2))
        l3 = (l - curv*l*self.de*np.cos(-phi + 2*np.pi/3 + np.pi/2))
        return np.array([
            l1,
            l2,
            l3
        ])

    # =====================================================
    # FORWARD KINEMATIC
    # =====================================================

    def forward(self, l):
        l1, l2, l3 = l
        phi = np.arctan2(l2 - 2*l1 + l3, -np.sqrt(3)*(l2 - l3))
        temp = (l1**2 - l1*l2 - l1*l3 + l2**2 - l2*l3 + l3**2)
        temp = max(temp, 0.0)
        root = np.sqrt(temp)
        curv = (100.0*root/(583.0*(l1 + l2 + l3)))
        theta = (100.0*((l1 + l2 + l3)/3.0)*root/(583.0*(l1 + l2 + l3)))
        return np.array([
            phi,
            curv,
            theta
        ])

    # =====================================================
    # NUMERICAL JACOBIAN
    # =====================================================

    def jacobian(self, l):
        J = np.zeros((3,3))
        f0 = self.forward(l)
        for i in range(3):
            dl = np.zeros(3)
            dl[i] = self.delta
            fi = self.forward(l + dl)
            J[:,i] = (fi - f0)/self.delta
        return J

    # =====================================================
    # RUN
    # =====================================================

    def run(self):
        scale = 0.01
        while self.shared.running:
            # =====================================================
            # GET DATA
            # =====================================================
            

            with self.shared.lock:
                target = self.shared.target_point
                endEffector = self.shared.end_effector
                sm = dict(self.shared.spacemouse)

            if target is None or endEffector is None:
                continue
            
            target = target.copy()

            delta_x = 0
            delta_y = 0
            delta_z = 0

            if abs(sm["roll"]) > 100:
                delta_x = sm["roll"] * scale
            if abs(sm["pitch"]) > 100:
                delta_y = sm["pitch"] * scale
            if abs(sm["yaw"]) > 100:
                delta_z = sm["yaw"] * scale

            target[0] += delta_x
            target[1] += delta_y
            target[2] += delta_z

            with self.shared.lock:
                self.shared.target_point = target

            

            # # =====================================================
            # # TARGET CENTER
            # # =====================================================

            # target_center = 0.5*(target - offset_distance * normal) + 0.5*center

            # =====================================================
            # DESIRED POSE
            # =====================================================

            desired_pose = self.cal_pose_from_point(
                target[0],
                target[1],
                target[2]
            )

            # =====================================================
            # CURRENT POSE
            # =====================================================

            current_pose = self.cal_pose_from_point(
                endEffector[0],
                endEffector[1],
                endEffector[2]
            )

            # =====================================================
            # CURRENT LENGTH
            # =====================================================

            current_length = self.pose_to_length(
                current_pose[0],
                current_pose[1],
                current_pose[2]
            )

            # =====================================================
            # ERROR
            # =====================================================

            error = desired_pose - current_pose

            # =====================================================
            # JACOBIAN
            # =====================================================

            J = self.jacobian(current_length)

            # =====================================================
            # DAMPED LEAST SQUARE
            # =====================================================

            try:
                JT = J.T
                # lam = 1e-3
                # delta_l = (self.alpha*JT@np.linalg.inv(J@JT + lam*lam*np.eye(3))@error)
                delta_l = self.alpha*np.linalg.solve(JT @ J,JT @ error)
            except Exception as e:
                print(e)
                # time.sleep(0.01)
                continue

            # =====================================================
            # LIMIT
            # =====================================================

            delta_l = np.clip(
                delta_l,
                -3.0,
                3.0
            )

            # =====================================================
            # SAVE
            # =====================================================

            with self.shared.lock:
                # self.shared.target_center = target_center
                self.shared.desired_pose = desired_pose
                self.shared.current_pose = current_pose
                self.shared.current_length = current_length
                self.shared.delta_l = delta_l
            time.sleep(0.2)
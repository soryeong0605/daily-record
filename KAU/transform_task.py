import threading
import numpy as np
import time


class TransformTask(threading.Thread):

    def __init__(self, shared, yaw_deg, tilt_deg):

        super().__init__()
        self.shared = shared
        # # 4x4 transform matrix
        # self.T = T
        # DEG → RAD
        yaw = np.deg2rad(yaw_deg)
        tilt = np.deg2rad(tilt_deg)

        Rz = np.array([
            [ np.cos(yaw), -np.sin(yaw), 0 ],
            [ np.sin(yaw),  np.cos(yaw), 0 ],
            [ 0,            0,           1 ]
        ])

        Rx = np.array([
            [1, 0,              0             ],
            [0, np.cos(tilt),  -np.sin(tilt)],
            [0, np.sin(tilt),   np.cos(tilt)]
        ])
        # R = Rz @ Rx
        R = Rx
        t = np.array([
            -35.0,
            247.5,
            41.7
        ])
        self.T = np.eye(4)
        self.T[:3, :3] = R
        self.T[:3, 3] = t

    def run(self):

        while self.shared.running:

            # =========================
            # GET POINTS
            # =========================

            with self.shared.lock:

                points_3d = list(self.shared.points_3d)

            transformed_points = []

            # =========================
            # TRANSFORM
            # =========================

            for p in points_3d:
                x, y, X, Y, Z = p
                # no depth
                if X is None:

                    transformed_points.append(
                        (x, y, None, None, None)
                    )

                    continue

                # =========================
                # HOMOGENEOUS VECTOR
                # =========================

                P_camera = np.array([
                    X,
                    Y,
                    Z,
                    1.0
                ])

                # =========================
                # MATRIX MULTIPLY
                # =========================

                P_world = self.T @ P_camera

                Xw = round(P_world[0], 1)
                Yw = round(P_world[1], 1)
                Zw = round(P_world[2], 1)

                transformed_points.append(
                    (
                        x,
                        y,
                        Xw,
                        Yw,
                        Zw
                    )
                )

            # =========================
            # SAVE
            # =========================

            with self.shared.lock:
                self.shared.points_world = transformed_points

            valid_points = []
            for p in transformed_points:
                x, y, X, Y, Z = p
                if X is None:
                    continue
                valid_points.append(p)

            if len(valid_points) > 0:
                endEffector = max(
                    valid_points,
                    key=lambda p: p[4]
                )
                with self.shared.lock:
                    self.shared.end_effector = np.array([endEffector[2], endEffector[3], endEffector[4]])
            else:
                endEffector = None
                    
            # # ============================================
            # # GEOMETRY
            # # ============================================

            # valid_points = []

            # for p in transformed_points:
            #     x, y, X, Y, Z = p
            #     if X is None:
            #         continue
            #     valid_points.append([X, Y, Z])

            # # need at least 4 points
            # if len(valid_points) >= 4:
            #     P = np.array(valid_points[:4], dtype=np.float64)

            #     # ============================================
            #     # PLANE FIT
            #     # ============================================

            #     centroid = np.mean(P, axis=0)
            #     centered = P - centroid

            #     _, _, vh = np.linalg.svd(centered)

            #     # normal vector
            #     n = vh[-1]

            #     # normalize
            #     n = n / np.linalg.norm(n)
            #     if n[2] < 0:
            #         n = -n
            #     # ============================================
            #     # CIRCLE CENTER
            #     # use first 3 points
            #     # ============================================

            #     A = P[0]
            #     B = P[1]
            #     C = P[2]

            #     AB = B - A
            #     AC = C - A

            #     ABxAC = np.cross(AB, AC)
            #     denom = 2 * np.linalg.norm(ABxAC)**2
            #     if denom > 1e-6:

            #         term1 = np.cross(ABxAC, AB) * np.dot(AC, AC)
            #         term2 = np.cross(AC, ABxAC) * np.dot(AB, AB)
            #         I = A + (term1 + term2) / denom

            #         # ============================================
            #         # END EFFECTOR POINT
            #         # ============================================

            #         P_end_effector = I + self.shared.offset_distance * n

            #         # ============================================
            #         # SAVE
            #         # ============================================

            #         with self.shared.lock:
            #             self.shared.circle_center = I
            #             self.shared.normal_vector = n
            #             self.shared.end_effector = P_end_effector

            time.sleep(0.1)
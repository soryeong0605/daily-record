import threading
import pyrealsense2 as rs
import numpy as np
import time

class TrackingTask(threading.Thread):

    def __init__(self, shared, intrinsics):
        super().__init__()
        self.shared = shared
        self.intrinsics = intrinsics

    def run(self):

        while self.shared.running:

            with self.shared.lock:

                if self.shared.depth_frame is None:
                    continue

                depth_frame = self.shared.depth_frame
                frame = self.shared.frame
                points_2d = list(self.shared.points_2d)

            height, width = frame.shape[:2]

            points_3d = []

            for x, y in points_2d:

                depths = []

                # =========================
                # DEPTH AVERAGE
                # =========================

                for dx in range(-8, 9):
                    for dy in range(-8, 9):

                        xx = x + dx
                        yy = y + dy

                        if xx < 0 or yy < 0:
                            continue

                        if xx >= width or yy >= height:
                            continue

                        d = depth_frame.get_distance(xx, yy)

                        if 0.05 < d < 3.0:
                            depths.append(d)

                if len(depths) == 0:

                    points_3d.append((x, y, None, None, None))
                    continue

                depth = np.median(depths)

                # =========================
                # DEPROJECT
                # =========================

                point_3d = rs.rs2_deproject_pixel_to_point(
                    self.intrinsics,
                    [x, y],
                    depth
                )

                X = round(point_3d[0] * 1000.0, 1)
                Y = round(point_3d[1] * 1000.0, 1)
                Z = round(point_3d[2] * 1000.0, 1)

                points_3d.append((x, y, X, Y, Z))

            with self.shared.lock:

                self.shared.points_3d = points_3d

            time.sleep(0.1)
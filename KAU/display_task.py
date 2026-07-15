import threading
import cv2
import numpy as np
import time

class DisplayTask(threading.Thread):

    def __init__(self, shared):
        super().__init__()
        self.shared = shared
    
    def run(self):
        while self.shared.running:
            with self.shared.lock:

                if self.shared.frame is None:
                    continue

                frame = self.shared.frame.copy()
                mask = self.shared.mask
                # transformed/world points
                points = list(self.shared.points_3d)

                # geometry
                # circle_center = self.shared.circle_center
                # normal_vector = self.shared.normal_vector
                end_effector_point = self.shared.end_effector

                # target point in world coordinate
                target_point = self.shared.target_point

                # control signal
                delta_l = self.shared.delta_l

            height, width = frame.shape[:2]

            panel_width = 520

            display = np.zeros(
                (height, width + panel_width, 3),
                dtype=np.uint8
            )

            display[:, :width] = frame
            display[:, width:] = (40, 40, 40)

            cv2.putText(
                display,
                "Tracked Points",
                (width + 20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            # =========================
            # DRAW
            # =========================

            for i, (x, y, X, Y, Z) in enumerate(points):

                cv2.circle(display, (x, y), 10, (0, 255, 0), 2)
                cv2.circle(display, (x, y), 3, (0, 0, 255), -1)

                cv2.putText(
                    display,
                    str(i + 1),
                    (x + 12, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                geom_y = 60 + i * 30

                if X is None:
                    text = f"{i+1:02d} | NO DEPTH"
                else:
                    text = (
                        f"{i+1:02d} | "
                        f"X:{X:8.1f}   "
                        f"Y:{Y:8.1f}   "
                        f"Z:{Z:8.1f} mm"
                    )

                cv2.putText(
                    display,
                    text,
                    (width + 20, geom_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    1
                )

            # ============================================
            # GEOMETRY INFO
            # ============================================

            geom_y = 120

            cv2.putText(
                display,
                "Geometry",
                (width + 20, geom_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            geom_y += 30

            # ============================================
            # CIRCLE CENTER
            # ============================================

            # if circle_center is not None:

            #     Ix, Iy, Iz = circle_center

            #     text = (
            #         f"Center I: "
            #         f"{Ix:7.1f}, "
            #         f"{Iy:7.1f}, "
            #         f"{Iz:7.1f}"
            #     )

            #     cv2.putText(
            #         display,
            #         text,
            #         (width + 20, geom_y),
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         0.55,
            #         (255, 255, 255),
            #         1
            #     )

            #     geom_y += 30

            # ============================================
            # NORMAL VECTOR
            # ============================================

            # if normal_vector is not None:

                # nx, ny, nz = normal_vector

                # text = (
                #     f"normal vector: "
                #     f"{nx:6.3f}, "
                #     f"{ny:6.3f}, "
                #     f"{nz:6.3f}"
                # )

                # cv2.putText(
                #     display,
                #     text,
                #     (width + 20, geom_y),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.55,
                #     (255, 255, 255),
                #     1
                # )

                # geom_y += 30

            # ============================================
            # OFFSET POINT
            # ============================================

            if end_effector_point is not None:

                Px, Py, Pz = end_effector_point

                text = (
                    f"End Effector Point: "
                    f"{Px:7.1f}, "
                    f"{Py:7.1f}, "
                    f"{Pz:7.1f}"
                )

                cv2.putText(
                    display,
                    text,
                    (width + 20, geom_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1
                )

            # geom_y = 380

            # cv2.putText(
            #     display,
            #     "Control Signal",
            #     (width + 20, geom_y),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.7,
            #     (0, 255, 255),
            #     2
            # )



            # ============================================
            # TARGET POINT
            # ============================================

            geom_y += 40

            cv2.putText(
                display,
                "Target Point",
                (width + 20, geom_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            geom_y += 30

            if target_point is not None:

                tx, ty, tz = target_point

                text = (
                    f"X:{tx:7.1f}   "
                    f"Y:{ty:7.1f}   "
                    f"Z:{tz:7.1f}"
                )

                cv2.putText(
                    display,
                    text,
                    (width + 20, geom_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1
                )


            # ============================================
            # DELTA L
            # ============================================

            geom_y += 40

            cv2.putText(
                display,
                "Delta L",
                (width + 20, geom_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            geom_y += 30

            if delta_l is not None:

                dl1, dl2, dl3 = delta_l

                text = (
                    f"dL1:{dl1:7.3f}   "
                    f"dL2:{dl2:7.3f}   "
                    f"dL3:{dl3:7.3f}"
                )

                cv2.putText(
                    display,
                    text,
                    (width + 20, geom_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1
                )

            if mask is not None:
                cv2.imshow("Mask", mask)

            cv2.imshow("Result", display)
            key = cv2.waitKey(1)

            if key == 27:
                self.shared.running = False
            
            #권소령이 함

            if key == ord('r'):
                with self.shared.lock:
                    self.shared.recording = not self.shared.recording
                    self.shared.recording_changed = True

            time.sleep(0.1)
        cv2.destroyAllWindows()
import threading
import cv2
import numpy as np
import time


class VisionTask(threading.Thread):
    def __init__(self, shared):
        super().__init__()
        self.shared = shared

    def run(self):
        while self.shared.running:
            # =========================
            # GET FRAME
            # =========================
            with self.shared.lock:
                if self.shared.frame is None:
                    continue
                frame = self.shared.frame.copy()
            # =========================
            # HSV
            # =========================
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # =========================
            # TRACKBAR
            # =========================
            hmin = cv2.getTrackbarPos("H min", "Control")
            hmax = cv2.getTrackbarPos("H max", "Control")

            smin = cv2.getTrackbarPos("S min", "Control")
            smax = cv2.getTrackbarPos("S max", "Control")

            vmin = cv2.getTrackbarPos("V min", "Control")
            vmax = cv2.getTrackbarPos("V max", "Control")

            min_area = cv2.getTrackbarPos("Min Area", "Control")
            max_area = cv2.getTrackbarPos("Max Area", "Control")

            dot_count = cv2.getTrackbarPos("Dot Count", "Control")
            # =========================
            # MASK
            # =========================

            lower = np.array([hmin, smin, vmin])
            upper = np.array([hmax, smax, vmax])
            mask = cv2.inRange(hsv, lower, upper)

            # =========================
            # FILTER
            # =========================

            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            _, mask = cv2.threshold(
                mask,
                127,
                255,
                cv2.THRESH_BINARY
            )

            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                kernel
            )

            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                kernel
            )

            # =========================
            # CONTOURS
            # =========================

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            points = []

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area or area > max_area:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                x = int(M["m10"] / M["m00"])
                y = int(M["m01"] / M["m00"])
                points.append((x, y))

            # =========================
            # SORT
            # =========================

            points.sort(key=lambda p: p[1])
            points = points[:dot_count]

            # =========================
            # SAVE
            # =========================

            with self.shared.lock:

                self.shared.mask = mask
                self.shared.points_2d = points

            time.sleep(0.1)
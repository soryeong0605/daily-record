# daily-record
기록용용용
안녕
function v_ref = smooth_accel(v_target)
  
   
    persistent v_prev;     
    dt = 0.002;           
    max_accel = 30;       
    % ---------------------------------

    if isempty(v_prev)
        v_prev = 0;
    end

   
    diff = v_target - v_prev;
    
    max_step = max_accel * dt;

    if abs(diff) < max_step
      
        v_ref = v_target;
    else
      
        v_ref = v_prev + sign(diff) * max_step;
    end

    v_prev = v_ref;
end






0.0089, +0.0000, +0.0000], safety_net_hits=132
[FrankaController] age=0.015, target=[+0.0000, +0.0000, +0.0000, +0.0195], send=[+0.0000, +0.0000, +0.0000, +0.0032], safety_net_hits=137
[FrankaController] age=0.016, target=[+0.0141, +0.0592, +0.0000, +0.0519], send=[+0.0002, +0.0003, +0.0000, +0.0010], safety_net_hits=141
[FrankaController] age=0.001, target=[+0.0000, -0.0242, +0.0000, +0.0097], send=[+0.0000, -0.0070, +0.0000, +0.0097], safety_net_hits=146
[FrankaController] Native velocity loop exception: libfranka: Move command aborted: motion aborted by reflex! ["cartesian_motion_generator_joint_acceleration_discontinuity"]
[FrankaController] Force stop requested
Exception in thread Thread-1 (_velocity_loop):
Traceback (most recent call last):
  File "/home/holabrb/miniforge3/envs/franka12/lib/python3.12/threading.py", line 1075, in _bootstrap_inner
    self.run()
  File "/home/holabrb/miniforge3/envs/franka12/lib/python3.12/threading.py", line 1012, in run
    self._target(*self._args, **self._kwargs)
  File "/home/holabrb/woonjoo/Jul16/cosine_visual/franka_cosine.py", line 447, in _velocity_loop
    state, duration = active.readOnce()
                      ^^^^^^^^^^^^^^^^^
pylibfranka._pylibfranka.ControlException: libfranka: Move command aborted: motion aborted by reflex! ["cartesian_motion_generator_joint_acceleration_discontinuity"]
[Main] Live monitor window opened. Close it or Ctrl+C to stop.
[Main] Teleop running with hybrid cosine/slew-rate smoother
[Main] Press Ctrl+C to stop
[Main] Franka velocity thread died.
[FrankaController] Native Cartesian velocity servo stopped
[Main] Stopped
(franka12) holabrb@franka:~/woonjoo/Jul16/cosine_visual$

#!/usr/bin/env python3
"""
preview_monitor_only.py

Standalone preview -- no SpaceMouse, no Franka, no franka_cosine.py needed.
Shows what the CV2Monitor window looks like, animating fake sine-wave
values (in / out / actual) so you can see the three bar segments move.

Run:
    python3 preview_monitor_only.py

Press ESC (with the window focused) or Ctrl+C in the terminal to stop.
"""

import time
import numpy as np
import cv2


class CV2Monitor:
    WINDOW_NAME = "SpaceMouse -> Franka Monitor (PREVIEW)"

    def __init__(self, v_range=0.08, w_range=0.6):
        self.v_range = v_range
        self.w_range = w_range
        self.labels = ["vx", "vy", "vz", "wz"]

        self.panel_w = 760
        self.panel_h = 460
        self.bar_x0 = 160
        self.bar_w = 420
        self.bar_h = 48
        self.row_gap = 88
        self.y0 = 90

        self._closed = False

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, self.panel_w, self.panel_h)

    def is_open(self):
        return not self._closed

    def _draw_bar(self, panel, row_idx, val_in, val_out, val_actual, rng):
        y = self.y0 + row_idx * self.row_gap
        cx = self.bar_x0 + self.bar_w // 2
        half = self.bar_w // 2
        third = self.bar_h // 3

        cv2.rectangle(panel, (self.bar_x0, y), (self.bar_x0 + self.bar_w, y + self.bar_h),
                      (70, 70, 70), 1)
        cv2.line(panel, (cx, y - 6), (cx, y + self.bar_h + 6), (200, 200, 200), 1)

        def seg(val, y_top, y_bot, color):
            frac = float(np.clip(val / rng, -1.0, 1.0))
            w = int(frac * half)
            if w >= 0:
                cv2.rectangle(panel, (cx, y_top), (cx + w, y_bot), color, -1)
            else:
                cv2.rectangle(panel, (cx + w, y_top), (cx, y_bot), color, -1)

        seg(val_in,     y + 1,             y + third - 1,       (255, 140, 30))   # blue
        seg(val_out,    y + third + 1,     y + 2 * third - 1,   (30, 140, 255))   # orange
        seg(val_actual, y + 2 * third + 1, y + self.bar_h - 1,  (60, 220, 60))    # green

        cv2.putText(panel, self.labels[row_idx], (20, y + self.bar_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        tx = self.bar_x0 + self.bar_w + 15
        cv2.putText(panel, f"in:     {val_in:+.4f}",    (tx, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 180, 120), 1)
        cv2.putText(panel, f"out:    {val_out:+.4f}",   (tx, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (150, 210, 255), 1)
        cv2.putText(panel, f"actual: {val_actual:+.4f}", (tx, y + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (140, 255, 140), 1)

    def update(self, target_in, out, actual, hits):
        panel = np.zeros((self.panel_h, self.panel_w, 3), dtype=np.uint8)

        cv2.putText(panel, "SpaceMouse -> Franka: live command monitor",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(panel, "blue=input (target)  orange=output (commanded)  green=actual (measured)",
                    (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1)

        ranges = [self.v_range, self.v_range, self.v_range, self.w_range]
        for i in range(4):
            self._draw_bar(panel, i, float(target_in[i]), float(out[i]), float(actual[i]), ranges[i])

        counter_y = self.y0 + 4 * self.row_gap + 10
        color = (0, 255, 0) if hits == 0 else (0, 165, 255)
        cv2.putText(panel, f"safety_net_hits: {hits}",
                    (20, counter_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        cv2.putText(panel, "Press ESC (with this window focused) or Ctrl+C in terminal to stop.",
                    (20, self.panel_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1)

        cv2.imshow(self.WINDOW_NAME, panel)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            self._closed = True

    def close(self):
        if not self._closed:
            cv2.destroyWindow(self.WINDOW_NAME)
            self._closed = True


def main():
    monitor = CV2Monitor()
    print("Preview window opened. Press ESC (window focused) or Ctrl+C to stop.")

    t0 = time.time()
    hits = 0

    try:
        while monitor.is_open():
            t = time.time() - t0

            # in: fake SpaceMouse target
            target_in = np.array([
                0.06 * np.sin(0.7 * t),
                0.06 * np.sin(0.5 * t + 1.0),
                0.04 * np.sin(0.3 * t + 2.0),
                0.4 * np.sin(0.9 * t + 0.5),
            ])
            # out: what gets commanded, lagging slightly (like the smoother)
            out = np.array([
                0.06 * np.sin(0.7 * (t - 0.2)),
                0.06 * np.sin(0.5 * (t - 0.2) + 1.0),
                0.04 * np.sin(0.3 * (t - 0.2) + 2.0),
                0.4 * np.sin(0.9 * (t - 0.2) + 0.5),
            ])
            # actual: what the "robot" really does, lagging a bit more + small noise
            actual = np.array([
                0.06 * np.sin(0.7 * (t - 0.35)),
                0.06 * np.sin(0.5 * (t - 0.35) + 1.0),
                0.04 * np.sin(0.3 * (t - 0.35) + 2.0),
                0.4 * np.sin(0.9 * (t - 0.35) + 0.5),
            ]) + np.random.normal(0, 0.002, size=4)

            if int(t * 10) % 15 == 0:
                hits += 1

            monitor.update(target_in, out, actual, hits)
            time.sleep(0.03)

    except KeyboardInterrupt:
        pass
    finally:
        monitor.close()
        print("Preview stopped.")


if __name__ == "__main__":
    main()

  

  
  

  
  
https://github.com/frankarobotics/franka_ros2  

https://beomjoonkim.github.io/


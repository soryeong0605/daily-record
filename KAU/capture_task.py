import threading
from time import time
import numpy as np
import pyrealsense2 as rs

class CaptureTask(threading.Thread):
    def __init__(self, shared):

        super().__init__()

        self.shared = shared
        # =========================
        # REALSENSE
        # =========================

        self.pipeline = rs.pipeline()

        self.config = rs.config()

        self.config.enable_stream(
            rs.stream.depth,
            640,
            480,
            rs.format.z16,
            30
        )

        self.config.enable_stream(
            rs.stream.color,
            640,
            480,
            rs.format.bgr8,
            30
        )

        self.profile = self.pipeline.start(self.config)

        # =========================
        # DEPTH SENSOR
        # =========================

        depth_sensor = self.profile.get_device().first_depth_sensor()

        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 1)

        if depth_sensor.supports(rs.option.laser_power):
            depth_sensor.set_option(rs.option.laser_power, 360)

        # =========================
        # ALIGN
        # =========================

        self.align = rs.align(rs.stream.color)

        # =========================
        # FILTER
        # =========================

        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()

        # =========================
        # INTRINSICS
        # =========================

        color_profile = rs.video_stream_profile(
            self.profile.get_stream(rs.stream.color)
        )

        self.intrinsics = color_profile.get_intrinsics()

    def run(self):

        while self.shared.running:

            frames = self.pipeline.wait_for_frames()

            aligned_frames = self.align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # =========================
            # DEPTH FILTER
            # =========================

            depth_frame = self.spatial.process(depth_frame)
            depth_frame = self.temporal.process(depth_frame)

            depth_frame = depth_frame.as_depth_frame()

            # =========================
            # IMAGE
            # =========================

            frame = np.asanyarray(color_frame.get_data())

            with self.shared.lock:

                self.shared.frame = frame
                self.shared.depth_frame = depth_frame
        self.pipeline.stop()
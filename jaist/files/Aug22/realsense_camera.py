"""
realsense_camera.py

RealSense 카메라 1~N대를 시리얼 넘버로 지정해서 열고, 백그라운드 스레드에서
계속 최신 컬러 프레임을 갱신해두는 wrapper.

사용 전 확인:
    import pyrealsense2 as rs
    ctx = rs.context()
    for dev in ctx.query_devices():
        print(dev.get_info(rs.camera_info.name), dev.get_info(rs.camera_info.serial_number))
위처럼 시리얼 넘버 2개를 먼저 뽑아서 아래 CAMERA_SERIALS 자리에 넣어주세요.
"""

import threading
import time

import numpy as np
import pyrealsense2 as rs


class RealSenseCamera:
    """카메라 1대. serial_number를 지정하면 항상 같은 물리 카메라를 엶."""

    def __init__(self, serial_number, width=640, height=480, fps=30, name="cam"):
        self.serial_number = serial_number
        self.name = name
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial_number)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.profile = self.pipeline.start(config)

        self._latest_frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while self._running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                img = np.asanyarray(color_frame.get_data())
                # RealSense는 BGR로 나옴 -- 저장은 RGB로 통일
                img_rgb = img[:, :, ::-1].copy()
                with self._lock:
                    self._latest_frame = img_rgb
            except Exception as e:
                print(f"[{self.name}] frame read failed: {e}")
                time.sleep(0.01)

    def get_latest_frame(self):
        """(H, W, 3) uint8 RGB, 아직 프레임이 없으면 None."""
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self.pipeline.stop()
        except Exception:
            pass


class MultiRealSenseCamera:
    """카메라 여러 대를 한번에 관리. camera_0, camera_1 ... 이름으로 접근."""

    def __init__(self, serial_numbers, width=640, height=480, fps=30):
        """
        serial_numbers: ["1234...", "5678..."] 처럼 순서대로 camera_0, camera_1로 매핑됨.
        """
        self.cameras = {}
        for i, serial in enumerate(serial_numbers):
            name = f"camera_{i}"
            print(f"[MultiRealSenseCamera] {name} <- serial {serial}")
            self.cameras[name] = RealSenseCamera(
                serial_number=serial, width=width, height=height, fps=fps, name=name
            )

    def start(self):
        for cam in self.cameras.values():
            cam.start()
        # 첫 프레임 들어올 때까지 잠깐 대기 (초기 recording에서 None 안 나오게)
        time.sleep(0.5)
        return self

    def get_latest_frames(self):
        """{'camera_0': (H,W,3) or None, 'camera_1': ...} 딕셔너리 반환."""
        return {name: cam.get_latest_frame() for name, cam in self.cameras.items()}

    def stop(self):
        for cam in self.cameras.values():
            cam.stop()

import threading
import pywinusb.hid as hid
import time


class SpaceMouseTask(threading.Thread):
    def __init__(self, shared):
        super().__init__()
        self.shared = shared

    # ============================================
    # INT16
    # ============================================

    def to_int16(self, low, high):
        value = low | (high << 8)
        if value >= 32768:
            value -= 65536
        return value

    # ============================================
    # CALLBACK
    # ============================================

    def handler(self, data):
        packet_id = data[0]
        with self.shared.lock:

            # translation
            if packet_id == 1:
                self.shared.spacemouse["x"] = self.to_int16(data[1], data[2])
                self.shared.spacemouse["y"] = self.to_int16(data[3], data[4])
                self.shared.spacemouse["z"] = self.to_int16(data[5], data[6])

            # rotation
            elif packet_id == 2:
                self.shared.spacemouse["roll"] = self.to_int16(data[1], data[2])
                self.shared.spacemouse["pitch"] = self.to_int16(data[3], data[4])
                self.shared.spacemouse["yaw"] = self.to_int16(data[5], data[6])

    # ============================================
    # RUN
    # ============================================

    def run(self):
        filter = hid.HidDeviceFilter(
            product_name='SpaceMouse Compact'
        )

        devices = filter.get_devices()

        if not devices:
            print("SpaceMouse not found")
            return
        
        device = devices[0]
        device.open()
        device.set_raw_data_handler(
            self.handler
        )

        print("SpaceMouse connected")
        while self.shared.running:
            time.sleep(0.05)
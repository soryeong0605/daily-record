import threading
import time
import serial


class SerialTask(threading.Thread):
    def __init__(self, shared):
        super().__init__()
        self.shared = shared

        # ============================================
        # SERIAL
        # ============================================

        self.ser = serial.Serial(
            port='COM8',
            stopbits=serial.STOPBITS_ONE,
            parity=serial.PARITY_NONE,
            baudrate=115200,
            timeout=1
        )

        time.sleep(1)
        print("Serial connected")

    # ============================================
    # RUN
    # ============================================

    def run(self):
        while self.shared.running:

            # ============================================
            # GET DELTA_L
            # ============================================
            with self.shared.lock:
                delta_l = self.shared.delta_l
            if delta_l is None:
                # time.sleep(0.01)
                continue

            # ============================================
            # SERIAL MESSAGE
            # ============================================

            dl1 = int(delta_l[0] * 20)
            dl2 = int(delta_l[1] * 20)
            dl3 = int(delta_l[2] * 20)

            msg = f"@{dl1},{dl2},{dl3}#"

            try:
                self.ser.write(
                    msg.encode()
                )
                # print(msg.strip())
            except Exception as e:
                print(e)
            time.sleep(0.5)
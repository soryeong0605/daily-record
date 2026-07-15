import time
import pyspacemouse


def main():
    print("Opening SpaceMouse...")

    with pyspacemouse.open() as device:
        print("SpaceMouse connected.")
        print("Move the SpaceMouse strongly. Press Ctrl+C to stop.")

        while True:
            state = device.read()

            if state is not None:
                if (
                    abs(state.x) > 0.02 or
                    abs(state.y) > 0.02 or
                    abs(state.z) > 0.02 or
                    abs(state.roll) > 0.02 or
                    abs(state.pitch) > 0.02 or
                    abs(state.yaw) > 0.02 or
                    state.buttons != [0, 0]
                ):
                    print(
                        f"x={state.x:.3f}, y={state.y:.3f}, z={state.z:.3f}, "
                        f"roll={state.roll:.3f}, pitch={state.pitch:.3f}, yaw={state.yaw:.3f}, "
                        f"buttons={state.buttons}"
                    )

            time.sleep(0.02)


if __name__ == "__main__":
    main()
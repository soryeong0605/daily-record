#!/usr/bin/env python3
"""
keypress_listener.py - Non-blocking single-key terminal input, no GUI.

Puts the controlling terminal into cbreak mode (key presses are delivered
immediately, no Enter needed, no local echo) and lets the caller poll for
a key without blocking. Restores the original terminal settings on stop()
or on exit, even if the program crashes.

Avoids any GUI toolkit (cv2/Qt, tkinter, etc.) -- this script is
console-only, so a GUI window is unnecessary and, on some Wayland setups,
actively broken (missing Qt font dirs can crash the process).
"""

import atexit
import sys
import termios
import tty


class KeypressListener:
    """
    Usage:
        kp = KeypressListener()
        kp.start()
        ...
        key = kp.get_key()   # returns a 1-char string, or None if nothing pressed
        ...
        kp.stop()
    """

    def __init__(self):
        self._fd = sys.stdin.fileno()
        self._original_settings = None
        self._active = False

    def start(self):
        if not sys.stdin.isatty():
            print("[KeypressListener] stdin is not a terminal -- key capture disabled.")
            return
        self._original_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._active = True
        atexit.register(self.stop)

    def get_key(self):
        """Non-blocking: returns the pressed key as a 1-char string, or None."""
        if not self._active:
            return None

        import select
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if r:
            return sys.stdin.read(1)
        return None

    def stop(self):
        if self._active and self._original_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._original_settings)
            self._active = False

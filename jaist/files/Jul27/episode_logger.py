#!/usr/bin/env python3
"""
episode_logger.py - Episode-based CSV logger for imitation-learning data.

Toggle-driven: call start_episode() once to begin logging, call
stop_episode() (or start_episode() again -- it's a toggle) to close the
file. A background thread writes rows at a fixed rate (drift-corrected
scheduling, same technique as PosePrinter in spacemouse_cosine.py) for as
long as an episode is active.

Each row = one timestep. Default columns are the TCP xyz position, but a
`state_extra_getter` / `action_getter` hook is provided so you can log more
(full 6D pose, commanded velocity, buttons, etc.) without changing the
core loop.
"""

import csv
import os
import threading
import time
from typing import Callable, Optional, Sequence


class EpisodeLogger:
    """
    Background CSV logger toggled on/off per episode.

    Parameters
    ----------
    state_getter : Callable[[], Sequence[float]]
        Returns the current state to log, e.g. lambda: get_tcp_xyz(franka).
        Length must match `state_columns`.
    state_columns : Sequence[str]
        Column names for state_getter's output, e.g. ["x", "y", "z"].
    action_getter : Optional[Callable[[], Sequence[float]]]
        Optional -- returns the current commanded action (e.g. the
        post-smoother velocity actually sent to the robot). If you skip
        this, only state gets logged.
    action_columns : Optional[Sequence[str]]
        Column names for action_getter's output, required if action_getter
        is given.
    log_dir : str
        Directory episodes are saved into (created if missing).
    hz : float
        Logging rate in Hz.
    """

    def __init__(
        self,
        state_getter: Callable[[], Sequence[float]],
        state_columns: Sequence[str] = ("x", "y", "z"),
        action_getter: Optional[Callable[[], Sequence[float]]] = None,
        action_columns: Optional[Sequence[str]] = None,
        log_dir: str = "episodes",
        hz: float = 20.0,
    ):
        if action_getter is not None and not action_columns:
            raise ValueError("action_columns is required when action_getter is given")

        self.state_getter = state_getter
        self.state_columns = list(state_columns)
        self.action_getter = action_getter
        self.action_columns = list(action_columns) if action_columns else []

        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.dt = 1.0 / hz

        self._active = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self._file = None
        self._writer = None
        self._episode_idx = 0
        self._episode_start_time = None
        self._row_count = 0

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # -------------------------
    # Public control
    # -------------------------

    @property
    def is_active(self) -> bool:
        return self._active.is_set()

    @property
    def episode_count(self) -> int:
        return self._episode_idx

    def toggle(self):
        """Start an episode if idle, stop it if currently active."""
        if self.is_active:
            self.stop_episode()
        else:
            self.start_episode()

    def start_episode(self):
        with self._lock:
            if self._active.is_set():
                print("[EpisodeLogger] Episode already active, ignoring start.")
                return

            self._episode_idx += 1
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"episode_{self._episode_idx:04d}_{timestamp}.csv"
            filepath = os.path.join(self.log_dir, filename)

            self._file = open(filepath, "w", newline="")
            header = ["timestamp", "episode_time"] + self.state_columns + self.action_columns
            self._writer = csv.writer(self._file)
            self._writer.writerow(header)

            self._episode_start_time = time.time()
            self._row_count = 0
            self._active.set()

            print(f"[EpisodeLogger] Episode {self._episode_idx} started -> {filepath}")

    def stop_episode(self):
        with self._lock:
            if not self._active.is_set():
                print("[EpisodeLogger] No active episode to stop.")
                return

            self._active.clear()
            if self._file is not None:
                self._file.close()

            print(f"[EpisodeLogger] Episode {self._episode_idx} stopped "
                  f"({self._row_count} rows).")

            self._file = None
            self._writer = None

    def shutdown(self):
        """Stop the background thread entirely (call on program exit)."""
        if self.is_active:
            self.stop_episode()
        self._stop.set()
        self._thread.join(timeout=2.0)

    # -------------------------
    # Background loop
    # -------------------------

    def _run(self):
        next_tick = time.monotonic()
        while not self._stop.is_set():
            if self._active.is_set():
                self._write_row()

            next_tick += self.dt
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Fell behind -- reset schedule instead of bursting rows.
                next_tick = time.monotonic()

    def _write_row(self):
        try:
            state = list(self.state_getter())
        except Exception as e:
            print(f"[EpisodeLogger] state_getter failed: {e}")
            return

        action = []
        if self.action_getter is not None:
            try:
                action = list(self.action_getter())
            except Exception as e:
                print(f"[EpisodeLogger] action_getter failed: {e}")
                action = [0.0] * len(self.action_columns)

        now = time.time()
        episode_time = now - self._episode_start_time

        with self._lock:
            if self._writer is None:
                return  # stopped between the .is_active check and here
            self._writer.writerow([now, episode_time] + state + action)
            self._row_count += 1

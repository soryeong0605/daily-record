

from __future__ import annotations

import time
import threading
import argparse
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

import os
import sys
FILE_DIR = os.path.dirname(os.path.abspath(os.path.abspath(__file__)))
PARENT_DIR = os.path.dirname(FILE_DIR)
sys.path.insert(0, PARENT_DIR)
import serial
from typing import Callable, Optional, Sequence

# -------------------------
# Serial helpers
# -------------------------
def drain_serial(ser: serial.Serial, stop_flag: dict):
    """Continuously read from serial so input buffer doesn't grow (optional)."""
    while not stop_flag["stop"]:
        try:
            ser.readline()
        except Exception:
            pass

# Expected ACK format from Arduino: "ACK p1 p2 p3\n"
# e.g. after receiving "0 5 5\n", Arduino replies "ACK 0 5 5\n"
ACK_PREFIX = "ACK"
def clamp_pwm(pwm: np.ndarray) -> np.ndarray:
    pwm = np.asarray(pwm, dtype=int).reshape(3,)
    return pwm.astype(np.int32)

class flowbot:
    from flowbot.kinematic_modeling import Flow_driven_bellow
    from flowbot.kinematic_modeling_linear import Flow_driven_bellow_linear
    from flowbot.workspace import workspace_using_fwdmodel
    from flowbot.plot_helper import plot_helper
    # online_optitrack.MotiveNatNetReader는 어디서도 실제로 안 쓰여서
    # (opti_handles 등은 그냥 그래프 핸들 변수 이름일 뿐) 제거함 --
    # 순수 feedforward 구동에서 online_optitrack.py 파일이 없어도
    # flowbot 클래스 정의 자체가 깨지지 않게 하기 위함.
    def __init__(self,serial_port: str = "/dev/ttyACM0",
                 baud: int = 115200,
                 pwm_min: int = 1,
                 pwm_max: int = 25,
                 enable_plot: bool = False,
                frequency: float = 30.0,
                max_pos_speed: float = 150.0,
                initial_pwm: Optional[Sequence[float]] = None,
                pressure_model: str = "linear",   # "learned" or "linear"
                draw_hull: bool = False,
                ):
        # # --- Serial ---
        self.serial_port = serial_port
        self.baud = baud
        self.ser = serial.Serial(self.serial_port, self.baud, timeout=0.1, write_timeout=0.1)
        
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        time.sleep(1.0)
        print("Opened serial:", self.serial_port, self.baud)
        self.stop_flag = {"stop": False}

        # ---- PWM 전송 주파수 실측용 ----
        self._send_count = 0
        self._last_send_time = None
        self._send_dt_window = []   # 최근 30개 dt(초) 보관 -- 평균 Hz 계산용

        # ACK protocol: Arduino replies "ACK p1 p2 p3\n" after applying each command.
        self._ack_event = threading.Event()   # set when a new ACK arrives
        self._ack_pwm   = None                # last ACK'd PWM (np.ndarray (3,) int)
        self._ack_lock  = threading.Lock()

        # ACK-aware reader replaces the old drain_serial thread.
        t_reader = threading.Thread(target=self._serial_reader_thread, daemon=True)
        t_reader.start()
        self.pwm_min = pwm_min
        self.pwm_max = pwm_max
        self.frequency = float(frequency)
        self.dt = 1.0 / self.frequency
        self.max_pos_speed = float(max_pos_speed)
        self._running = False
        self._lock = threading.Lock()
        self.pressure_model = pressure_model   # "learned" or "linear"
        # Init robot
        self.flowbot = self.flowbot_init()
        
        # Load workspace
        self.ws, self.tri, self.bbox = self.load_workspace(self.flowbot)
        
        if initial_pwm is None:
            # Start pc at workspace "origin": pwm = [1,1,1] if pwm_min=0, which is just above the minimum to avoid numerical issues in the model
            pwm = np.array([self.pwm_min, self.pwm_min , self.pwm_min], dtype=int)  # [1,1,1] if pwm_min=0
            pb0 = self.flowbot.pwm_to_pressure(pwm)
            fk0 = self.flowbot.forward_kinematics_from_pressures(pb0)
            self.pc_init = np.asarray(fk0["pc"], dtype=float).reshape(3,)
            self.pc =  np.asarray(fk0["pc"], dtype=float).reshape(3,)
            print("Init pwm:", pwm, "Init pc:", self.pc)
            self.serial_sending(pwm)
            self.last_pwm = pwm
        else:
            pwm = np.asarray(initial_pwm, dtype=float).copy()
            pb0 = self.flowbot.pwm_to_pressure(pwm)
            fk0 = self.flowbot.forward_kinematics_from_pressures(pb0)
            self.pc =  np.asarray(fk0["pc"], dtype=float).reshape(3,)
            print("Init pwm:", pwm, "Init pc:", self.pc)
            self.serial_sending(pwm)
            self.last_pwm = pwm
        if enable_plot:
            plt.ion()
            self.pl = self.plot_helper()
            self.fig, self.axes, self.pc_handles, self.opti_handles, self.trail_handles = self.pl.setup_plot(self.ws.P, draw_hull=draw_hull)

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.show(block=False)
            self.pl.update_point_handle(self.pc_handles, self.pc)
        else:
            self.fig = None
            self.pc_handle = None
            self.opti_handle = None
            self.opti_trail = None
            self.pl = None 

    def flowbot_init(self):
        
        _k_model = lambda deltal: 0.18417922367667078 + 0.1511268093994831 * (1.0 - np.exp(-0.18801952663756039 * deltal))
        _common = dict(D_in=5, D_out=16.5, l0=82, d=28.17, lb=0.0, lu=13.5,
                       k_model=_k_model, a_delta=0, b_delta=0)

        if self.pressure_model == "linear":
            print("[flowbot] Using LINEAR pressure model (a=0.004227, b=0.012059)")
            robot = self.Flow_driven_bellow_linear(
                **_common,
                a_pwm2press=0.004227,
                b_pwm2press=0.012059,
            )
        else:
            from pathlib import Path
            from flowbot.pwm2flow import Pwm2FlowModel
            from flowbot.pressure_flow_model import Flow2PressModel, Press2FlowModel

            _pkl_dir = Path(__file__).parent.parent.parent / "flowbot"
            pwm2flow   = Pwm2FlowModel.load(  _pkl_dir / "pwm2flow.pkl")
            flow2press = Flow2PressModel.load( _pkl_dir / "flow2press.pkl")
            press2flow = Press2FlowModel.load( _pkl_dir / "press2flow.pkl")
            print("[flowbot] Using LEARNED pressure model (pwm2flow + flow2press + press2flow)")
            robot = self.Flow_driven_bellow(
                **_common,
                pwm2flow_model   = pwm2flow,
                flow2press_model = flow2press,
                press2flow_model = press2flow,
            )
        return robot
    
    def _serial_reader_thread(self) -> None:
        """
        Background serial reader.

        Captures ACK lines from Arduino (format: "ACK p1 p2 p3\\n").
        All other lines are silently drained so the input buffer never fills up.
        """
        while not self.stop_flag["stop"]:
            try:
                raw = self.ser.readline()
                if not raw:
                    continue
                line = raw.decode("ascii", errors="ignore").strip()
                if line.startswith(ACK_PREFIX):
                    parts = line.split()          # ["ACK", "p1", "p2", "p3"]
                    if len(parts) == 4:
                        ack_pwm = np.array([int(parts[1]), int(parts[2]), int(parts[3])],
                                           dtype=int)
                        with self._ack_lock:
                            self._ack_pwm = ack_pwm
                        self._ack_event.set()
                # non-ACK lines are discarded (drain behaviour)
            except Exception:
                pass

    def serial_sending(self, pwm, wait_ack: bool = False, ack_timeout: float = 0.5) -> bool:
        """
        Send a PWM command over serial.

        Args:
            pwm         : array-like (3,) — PWM values to send.
            wait_ack    : If True, block until Arduino sends back "ACK p1 p2 p3"
                          or until ack_timeout expires.
                          Requires Arduino firmware to echo "ACK p1 p2 p3\\n".
                          Default False → fire-and-forget (original behaviour).
            ack_timeout : Maximum seconds to wait for ACK (default 0.5 s).

        Returns:
            True  — command sent (and ACK received if wait_ack=True).
            False — serial write failed, or ACK timed out.
        """
        cmd0 = f"{int(pwm[0])} {int(pwm[1])} {int(pwm[2])}\n"
        # Clear event BEFORE writing so a fast ACK is never missed.
        if wait_ack:
            self._ack_event.clear()
        try:
            self.ser.write(cmd0.encode("ascii"))

            # ---- 실제 전송 간격(dt)/Hz 실측 ----
            now = time.time()
            if self._last_send_time is not None:
                dt = now - self._last_send_time
                hz = 1.0 / dt if dt > 0 else float("inf")
                self._send_dt_window.append(dt)
                if len(self._send_dt_window) > 30:
                    self._send_dt_window.pop(0)
                avg_hz = (
                    len(self._send_dt_window) / sum(self._send_dt_window)
                    if sum(self._send_dt_window) > 0 else float("inf")
                )
                self._send_count += 1
                extra = f"  dt={dt*1000:.1f}ms  inst={hz:.1f}Hz"
                if self._send_count % 30 == 0:
                    extra += f"  avg(30)={avg_hz:.1f}Hz"
            else:
                extra = ""
            self._last_send_time = now

            print("[PYTHON] Sent:", cmd0.strip(), extra)
        except Exception as e:
            print("Serial write failed:", e)
            return False
        if wait_ack:
            got_ack = self._ack_event.wait(timeout=ack_timeout)
            if not got_ack:
                print(f"[WARN] No ACK for PWM {pwm} within {ack_timeout:.2f}s")
            return got_ack
        return True
    # -------------------------
    # Workspace 
    # -------------------------
    def load_workspace(self,robot: Flow_driven_bellow):
        ws = self.workspace_using_fwdmodel(robot=robot, pwm_min=self.pwm_min, pwm_max=self.pwm_max)
        hull = ws.build_workspace_hull_checker(ws.P)
        tri = hull["tri"]
        bbox = hull["bbox"]
        return ws, tri, bbox
    
    def apply_workspace_constraint(self,
        pc: np.ndarray,
        pc_proposed: np.ndarray,
        policy: str = "backtrack",
    ) -> np.ndarray:
        if self.ws.is_inside_workspace(pc_proposed, self.tri):
            return pc_proposed

        if policy == "hold":
            return pc

        if policy == "backtrack":
            d = pc_proposed - pc
            norm = float(np.linalg.norm(d))
            if norm < 1e-12:
                return pc
            direction = d / norm

            # Try stepping from proposed back toward current until inside
            steps = 30
            for a in np.linspace(1.0, 0.0, steps):
                cand = pc + a * d
                if self.ws.is_inside_workspace(cand,self.tri):
                    return cand
            return pc

        raise ValueError(f"Unknown outside policy: {policy}")

    def start(self) -> None:
        self._running = True


    def _solve_pwm(self, p_task: np.ndarray) -> np.ndarray:
        if self.kin is not None:
            if not hasattr(self.kin, "solve"):
                raise AttributeError("kin must have method solve(p_task)->pwm")
            pwm = self.kin.solve(p_task)  # type: ignore[attr-defined]
        else:
            pwm = self.ik_fn(p_task)  # type: ignore[misc]
        return np.asarray(pwm, dtype=float)
    
    def stop(self) -> None:
        # Stop the robot
        try:
            cmd = "0 0 0\n"
            self.ser.write(cmd.encode("ascii"))
            print("[PYTHON] Sent:", cmd.strip())
        except Exception:
            pass
        try:
            self.ser.close()
        except Exception:
            pass

        print("Closed serial. Exited.")
        self._running = False
        
        if self.pl is not None:
            try:
                self.pl.close()
            except Exception:
                pass
        self.stop_flag["stop"] = True
        try:
            self.ser.close()
        except Exception:
            pass

    def reset(self):
        # (26,26,26)으로 변경 함 (이전에는 20,20,20였음). pwm이 바뀌면
        # pc도 __init__과 똑같이 순기구학으로 다시 계산해줘야
        # IK가 안 꼬임 (실제 보낸 pwm과 self.pc가 일치해야 함).
        pwm = np.array([26, 26, 26], dtype=np.int32)
        pb = self.flowbot.pwm_to_pressure(pwm)
        fk = self.flowbot.forward_kinematics_from_pressures(pb)
        self.pc = np.asarray(fk["pc"], dtype=float).reshape(3,)
        self.last_pwm = pwm
        self.serial_sending(pwm)

    def set_compensator(self, compensator) -> None:
        """
        Attach an ErrorCompensator to be applied inside every step() call.

        method="simple" : recomputes IK at (pc - correction) so the robot aims
                          at a pre-corrected position every tick.
        method="mpc"    : adds deltaU to the nominal PWM every tick.

        Pass None to disable.
        """
        self._compensator = compensator
        if compensator is not None:
            compensator.reset()
            print(f"[flowbot] Compensator attached  method={compensator.method}")
        else:
            print("[flowbot] Compensator detached.")

    def step(self, dpc) -> np.ndarray:
        if not self._running:
            raise RuntimeError("Call start() before step().")

        t0 = time.time()
        dpc = dpc[:3] * self.max_pos_speed * self.dt

        with self._lock:
            self.pc = self.apply_workspace_constraint(self.pc, self.pc + dpc, "backtrack")

        # ── Nominal IK ────────────────────────────────────────────
        try:
            ik  = self.flowbot.inverse_pressures_from_position(self.pc)
            pwm = np.asarray(ik["pwm"], dtype=int).reshape(3,)
        except Exception as e:
            print("IK failed:", e)
            pwm = np.array([0, 0, 0], dtype=np.int32)

        # ── Compensation ──────────────────────────────────────────
        comp = getattr(self, "_compensator", None)
        if comp is not None:
            features = [*self.pc, *pwm]

            if comp.method == "mpc":
                # Optimise deltaU in PWM space, add to nominal
                delta_u = comp.step(features)
                if delta_u is not None:
                    pwm = np.clip(pwm + delta_u, 0, 180).astype(int)
                    print(f"[comp/mpc] deltaU={np.round(delta_u,2)}  pwm→{pwm}")

            elif comp.method == "simple":
                # Predict position error, recompute IK at corrected target
                correction = comp.step(features)
                if correction is not None:
                    pc_corr = self.apply_workspace_constraint(
                        self.pc, self.pc - correction, "backtrack"
                    )
                    try:
                        ik2 = self.flowbot.inverse_pressures_from_position(pc_corr)
                        pwm = np.asarray(ik2["pwm"], dtype=int).reshape(3,)
                        print(f"[comp/simple] corr={np.round(correction,2)}  pwm→{pwm}")
                    except Exception:
                        pass   # fallback to nominal pwm
        offset = np.asarray([0, 0, 0], dtype=np.int32)
        sent_pwm = pwm + offset if pwm[1] > 0 else pwm
        self.serial_sending(sent_pwm)
        self.last_pwm = sent_pwm

        elapsed = time.time() - t0
        if elapsed < self.dt:
            time.sleep(self.dt - elapsed)

        return pwm

    def update_plot(self):
        if self.pl is not None:
            self.pl.update_point_handle(self.pc_handles,self.pc)
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            # plt.pause(0.001)
            
    def set_position(self, pc: Sequence[float]) -> None:
        p = self.apply_workspace_constraint(self.pc,np.asarray(pc, dtype=float))
        with self._lock:
            self.pc = p
        if self.pl is not None:
            self.pl.update_point_handle(self.pc_handles,p)

    @property
    def pwm_cur(self) -> np.ndarray:
        """Current PWM command (alias for last_pwm)."""
        return self.last_pwm

    def apply_delta_pwm(self, delta_u: np.ndarray) -> np.ndarray:
        """
        Add a PWM correction delta_u to the current command and send it.

        Used by the MPC compensator to apply deltaU without recomputing IK.

        Parameters
        ----------
        delta_u : (3,) array — PWM correction in raw counts (can be float).

        Returns
        -------
        pwm_new : (3,) int array — the clamped, applied PWM.
        """
        delta_u  = np.asarray(delta_u, dtype=float).reshape(3,)
        pwm_new  = np.clip(
            self.last_pwm + delta_u,
            0, 180,  # 180 is a safe upper bound; actual max is usually lower (e.g. 25) depending on the robot and pressure model
        ).astype(int)
        print(f"Applying delta PWM: {delta_u} → New PWM: {pwm_new}")
        self.last_pwm = pwm_new
        self.serial_sending(pwm_new)
        return pwm_new

    def release(self):

        cmd = "r\n"
        try:
            self.ser.write(cmd.encode("ascii"))
        except Exception as e:
            print("Serial write failed:", e)
        time.sleep(0.1)



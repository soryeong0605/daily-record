from dataclasses import dataclass, field
from threading import Lock


@dataclass
class SharedData:

    # =========================
    # SPACEMOUSE COMMAND
    # =========================
    # deadband/속도 매핑까지 끝난 "목표 속도" (digital 모드, 고정 속력)
    # 실제 가속도 스무딩은 franka_task.py 쪽 HybridVelocitySmoother가 담당

    spacemouse = {
        "vx": 0.0,
        "vy": 0.0,
        "vz": 0.0,
        "wz": 0.0,
        "stamp": 0.0
    }

    # =========================
    # FRANKA ROBOT
    # =========================

    robot_pose = None                      # TCP xyz (np.ndarray, (3,))
    sent_velocity = [0.0, 0.0, 0.0, 0.0]    # 스무더 통과 후 실제로 로봇에 나간 [vx,vy,vz,wz]

    # =========================
    # RECORDING
    # =========================

    recording: bool = False
    recording_changed: bool = False

    # =========================
    # SLOW MODE
    # =========================
    # True면 franka_task가 MAX_LINEAR 대신 SLOW_LINEAR(0.01m/s)를 씀.
    # 's' 키로 토글 (keypress_task.py 참고)

    slow_mode: bool = False

    running: bool = True

    lock: Lock = field(default_factory=Lock)

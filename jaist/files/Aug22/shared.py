from dataclasses import dataclass, field
from threading import Lock
from typing import Optional

import numpy as np

# 레코딩 전용 고정 주기 -- franka 스레드(20Hz)든 flowbot 루프(가변)든
# 상관없이 이 주기로만 스텝을 기록해서, 저장되는 데이터의 시간 간격을 통일함.
RECORD_HZ = 20.0
RECORD_DT = 1.0 / RECORD_HZ


@dataclass
class SharedData:
    # "franka" 또는 "flowbot". SpaceMouse는 둘 중 이 값에 해당하는 쪽만
    # 열어서 독점 사용 -- 라이브러리(pyspacemouse vs libspnav)를 통일하지
    # 않고, 대신 한 번에 한쪽만 디바이스를 열게 해서 충돌을 피함.
    active_mode: str = "franka"

    running: bool = True
    lock: Lock = field(default_factory=Lock)

    # ===== 데이터수집(레코딩)용으로 추가된 부분 =====
    is_recording: bool = False
    episode_buffer: Optional[object] = None   # data_recorder.EpisodeBuffer 인스턴스 (main에서 생성)
    episode_count: int = 0
    last_record_time: float = 0.0             # RECORD_HZ 게이팅용 -- 마지막으로 기록한 시각

    # franka 스레드가 주기적으로 갱신 -- flowbot 모드에서 기록할 때도
    # "최근 franka pose"를 쓸 수 있게 함 (로봇 연결은 모드 전환과 무관하게 유지되므로
    # franka가 안 움직이는 중에도 상태 조회는 계속 가능)
    latest_franka_eef_pose: Optional[np.ndarray] = None
    latest_franka_velocity: Optional[np.ndarray] = None

    # main(flowbot 루프)이 매 스텝 갱신 -- franka 스레드에서 기록할 때도
    # "최근 PWM"을 참조할 수 있게 함
    latest_pwm: Optional[np.ndarray] = None
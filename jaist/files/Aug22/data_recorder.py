"""
data_recorder.py

교수님의 demo_collect.py 구조(DataBuffer + zarr save)를 그대로 가져오되,
필드명을 프랑카+소프트그리퍼 구조에 맞게 조정한 버전.

저장 필드:
- timestamp          : (T,)          각 스텝 시각
- franka_eef_pose     : (T, 6 or 7)   프랑카 플랜지(EE) pose (xyz + rotvec 또는 quat, 기존 코드 표현 방식에 맞추면 됨)
- gripper_tip_pose    : (T, 3 or 6)   소프트그리퍼 끝단 pose (PCC 순기구학으로 계산한 값, 아직 없으면 franka_eef_pose로 대체 후 나중에 채워도 됨)
- pwm_signals         : (T, 3)        아두이노로 보낸 PWM 명령 (현재 프레임에 대응하는 값 -- 이전 스텝 값 스냅샷 권장)
- action              : (T, action_dim) 그 순간 실제로 로봇/그리퍼에 내린 명령 (state와 구분!)
- sm_raw              : (T, N)        스페이스마우스 raw 값 (xyz+yaw+버튼 상태 등, 그대로 이어붙여 저장)
- operation_mode      : (T, 2) uint8  [franka_active, gripper_active] 형태 (release는 둘 다 1로 표현 가능)
- camera_0, camera_1, ...  : (T, H, W, 3)  RGB 프레임 (카메라 대수만큼, camera_names로 지정)

새 필드가 필요하면 EpisodeBuffer.add()의 kwargs만 늘리면 되고,
save_episode/create_zarr_dataset은 필드명에 의존하지 않는 범용 구조라 그대로 재사용 가능.
"""

from pathlib import Path
import numpy as np
import zarr
from zarr.codecs.numcodecs import Blosc


class EpisodeBuffer:
    """한 에피소드(demonstration) 동안의 모든 스텝을 모아두는 버퍼."""

    def __init__(self, camera_names=None):
        """camera_names: ["camera_0", "camera_1"] 처럼 사용할 카메라 키 목록. None/[]이면 카메라 없이 기록."""
        self.camera_names = list(camera_names) if camera_names else []
        self.reset()

    def reset(self):
        self._data = {
            "timestamp": [],
            "franka_eef_pose": [],
            "gripper_tip_pose": [],
            "pwm_signals": [],
            "action": [],
            "sm_raw": [],
            "operation_mode": [],
        }
        for cam_name in self.camera_names:
            self._data[cam_name] = []
        self._expected_shapes = {}  # 필드별로 첫 add()에서 본 shape -- 이후 불일치시 즉시 에러

    def _check_shape(self, key, arr):
        """이 필드가 이전 스텝들과 shape이 같은지 확인. 다르면 바로 명확한 에러로 알림
        (그대로 두면 to_dict()의 np.array()에서 훨씬 알아보기 힘든 에러가 남)."""
        shape = arr.shape
        if key in self._expected_shapes:
            if self._expected_shapes[key] != shape:
                raise ValueError(
                    f"[EpisodeBuffer] '{key}' 필드의 shape이 이전 스텝과 다릅니다: "
                    f"기존 {self._expected_shapes[key]} vs 이번 {shape}. "
                    f"franka/그리퍼 두 모드에서 서로 다른 크기로 이 필드를 채우고 있지 않은지 "
                    f"호출부(franka_task.py / main.py)를 확인하세요."
                )
        else:
            self._expected_shapes[key] = shape

    def add(
        self,
        timestamp,
        franka_eef_pose,
        gripper_tip_pose,
        pwm_signals,
        action,
        sm_raw,
        operation_mode,
        camera_frames=None,
    ):
        """camera_frames: {"camera_0": (H,W,3) array, "camera_1": ...} -- camera_names와 키가 일치해야 함."""
        franka_eef_pose = np.asarray(franka_eef_pose)
        gripper_tip_pose = np.asarray(gripper_tip_pose)
        pwm_signals = np.asarray(pwm_signals)
        action = np.asarray(action)
        sm_raw = np.asarray(sm_raw)

        for key, arr in [
            ("franka_eef_pose", franka_eef_pose),
            ("gripper_tip_pose", gripper_tip_pose),
            ("pwm_signals", pwm_signals),
            ("action", action),
            ("sm_raw", sm_raw),
        ]:
            self._check_shape(key, arr)

        self._data["timestamp"].append(timestamp)
        self._data["franka_eef_pose"].append(franka_eef_pose.copy())
        self._data["gripper_tip_pose"].append(gripper_tip_pose.copy())
        self._data["pwm_signals"].append(pwm_signals.copy())
        self._data["action"].append(action.copy())
        self._data["sm_raw"].append(sm_raw.copy())
        self._data["operation_mode"].append(
            np.asarray(operation_mode, dtype=np.uint8).copy()
        )
        if self.camera_names:
            if camera_frames is None:
                raise ValueError("camera_names가 지정됐는데 camera_frames가 없습니다.")
            for cam_name in self.camera_names:
                frame = camera_frames.get(cam_name)
                if frame is None:
                    raise ValueError(f"camera_frames에 '{cam_name}' 프레임이 없습니다.")
                self._data[cam_name].append(np.asarray(frame).copy())

    def __len__(self):
        return len(self._data["timestamp"])

    def to_dict(self):
        """zarr에 넣을 수 있는 numpy dict로 변환."""
        out = {}
        for key, values in self._data.items():
            out[key] = np.array(values)
        return out


def create_zarr_dataset(output_dir, camera_names=None, image_shape=None):
    """dataset.zarr 생성 (없으면 새로, 있으면 append 모드로 열기)."""
    zarr_path = Path(output_dir) / "dataset.zarr"
    root = zarr.open(str(zarr_path), mode="a")

    if "data" not in root:
        root.create_group("data")
    if "meta" not in root:
        meta = root.create_group("meta")
        meta.create_array("episode_ends", shape=(0,), dtype=np.int64, chunks=(100,))

    if camera_names and "camera_info" not in root:
        camera_info = root.create_group("camera_info")
        camera_info.attrs["camera_names"] = list(camera_names)
        if image_shape:
            camera_info.attrs["image_shape"] = image_shape
            camera_info.attrs["format"] = "RGB"

    return root


def save_episode(zarr_root, episode_data):
    """한 에피소드 데이터를 zarr에 이어붙여 저장. 반환값은 새로 저장된 에피소드 index."""
    data_group = zarr_root["data"]
    meta_group = zarr_root["meta"]

    episode_ends = meta_group["episode_ends"]
    n_eps = episode_ends.shape[0]
    current_len = int(episode_ends[n_eps - 1]) if n_eps > 0 else 0
    episode_len = episode_data["timestamp"].shape[0]
    new_len = current_len + episode_len

    for key, value in episode_data.items():
        if key not in data_group:
            if key.startswith("camera_"):
                data_group.create_array(
                    key,
                    shape=(new_len,) + value.shape[1:],
                    dtype=value.dtype,
                    chunks=(1,) + value.shape[1:],
                    compressors=Blosc(cname="lz4", clevel=3),
                )
            else:
                data_group.create_array(
                    key,
                    shape=(new_len,) + value.shape[1:],
                    dtype=value.dtype,
                    chunks=(100,) + value.shape[1:],
                )
        else:
            dataset = data_group[key]
            dataset.resize((new_len,) + value.shape[1:])

        data_group[key][current_len:new_len] = value

    episode_ends.resize(n_eps + 1)
    episode_ends[-1] = new_len

    return n_eps


def save_episode_csv(episode_data, csv_path):
    """카메라(camera_*) 배열은 제외하고, 스칼라/벡터 필드만 사람이 훑어볼 수 있게 CSV로 저장.
    각 필드가 다차원이면 col0, col1, ... 식으로 펼쳐서 한 줄에 다 넣음."""
    import csv as csv_module

    scalar_keys = [k for k in episode_data.keys() if not k.startswith("camera_")]
    n_steps = episode_data["timestamp"].shape[0]

    header = []
    for key in scalar_keys:
        arr = episode_data[key]
        if arr.ndim == 1:
            header.append(key)
        else:
            dim = arr.shape[1]
            header.extend([f"{key}_{i}" for i in range(dim)])

    with open(csv_path, "w", newline="") as f:
        writer = csv_module.writer(f)
        writer.writerow(header)
        for t in range(n_steps):
            row = []
            for key in scalar_keys:
                arr = episode_data[key]
                if arr.ndim == 1:
                    row.append(arr[t])
                else:
                    row.extend(arr[t].tolist())
            writer.writerow(row)

    print(f"[data_recorder] CSV -> {csv_path} ({n_steps} rows)")

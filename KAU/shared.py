from dataclasses import dataclass, field
from threading import Lock


@dataclass
class SharedData:

    frame = None
    depth_frame = None
    mask = None

    points_2d: list = field(default_factory=list)

    # capture points in camera coordinate
    points_3d: list = field(default_factory=list)

    # transformed points in world coordinate
    points_world: list = field(default_factory=list)

    # interpolated geometry in world coordinate
    # circle_center = None
    # normal_vector = None
    end_effector = None

    # target point in world coordinate
    target_point = None

    # robot parameter
    offset_distance: float = None

    # three-continuum state in world coordinate
    # target_center = None
    desired_pose = None
    current_pose = None
    current_length = None

    # control command
    delta_l = None

    # spacemouse
    spacemouse = {
        "x": 0,
        "y": 0,
        "z": 0,
        "roll": 0,
        "pitch": 0,
        "yaw": 0
    }

    # CR5 robot
    robot_pose = [0,0,0,0,0,0]
    robot_mode = 0
    
    #권소령이함

    recording: bool = False
    recording_changed: bool = False


    running: bool = True

    lock: Lock = field(default_factory=Lock)
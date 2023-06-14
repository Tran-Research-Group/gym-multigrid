from typing import TypeVar
from constants import *

WorldT = TypeVar("WorldT", bound="World")


class World:
    encode_dim = 6
    normalize_obs = 1

    COLOR_TO_IDX = COLOR_TO_IDX
    IDX_TO_COLOR = IDX_TO_COLOR

    # Map of object type to integers
    OBJECT_TO_IDX = {
        "unseen": 0,
        "empty": 1,
        "wall": 2,
        "floor": 3,
        "door": 4,
        "key": 5,
        "ball": 6,
        "box": 7,
        "goal": 8,
        "lava": 9,
        "agent": 10,
        "objgoal": 11,
        "switch": 12,
    }
    IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))


class CollectWorld:
    encode_dim = 3
    normalize_obs = 1
    
    COLOR_TO_IDX = COLOR_TO_IDX
    IDX_TO_COLOR = IDX_TO_COLOR

    OBJECT_TO_IDX = {
        "empty": 0,
        "wall": 1,
        "ball": 2,
        "agent": 3,
    }
    IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))
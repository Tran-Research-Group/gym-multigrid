from typing import TypeVar
import numpy as np
from .rendering import *

# Map of color names to RGB values
COLORS: dict[str, NDArray] = {
    "red": np.array([228, 3, 3]),
    "orange": np.array([255, 140, 0]),
    "yellow": np.array([255, 237, 0]),
    "green": np.array([0, 128, 38]),
    "blue": np.array([0, 77, 255]),
    "purple": np.array([117, 7, 135]),
    "brown": np.array([120, 79, 23]),
    "grey": np.array([100, 100, 100]),
    "light_red": np.array([234, 153, 153]),
    "light_blue": np.array([90, 170, 223]),
}

COLOR_NAMES: list[str] = sorted(list(COLORS.keys()))

# Map of state names to integers
STATE_TO_IDX: dict[str, int] = {
    "open": 0,
    "closed": 1,
    "locked": 2,
}

WorldT = TypeVar("WorldT", bound="World")


class World:
    encode_dim = 6
    normalize_obs = 1

    # Used to map colors to integers
    COLOR_TO_IDX: dict[str, int] = {key: i for i, key in enumerate(COLORS.keys())}
    IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

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
    IDX_TO_OBJECT: dict[int, str] = dict(
        zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys())
    )


class CollectWorld:
    encode_dim = 3
    normalize_obs = 1

    COLOR_TO_IDX: "dict[str, int]" = {key: i for i, key in enumerate(COLORS.keys())}
    IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

    OBJECT_TO_IDX = {
        "empty": 0,
        "wall": 1,
        "ball": 2,
        "agent": 3,
    }
    IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))


class SmallWorld:
    encode_dim = 3
    normalize_obs = 1 / 3

    COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "grey": 3}
    IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

    OBJECT_TO_IDX = {"unseen": 0, "empty": 1, "wall": 2, "agent": 3}
    IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

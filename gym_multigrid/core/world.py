from typing import TypeVar
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .constants import COLORS

WorldT = TypeVar("WorldT", bound="World")


@dataclass
class World:
    encode_dim: int
    normalize_obs: int
    OBJECT_TO_IDX: dict[str, int]  # Map of object type to integers
    COLORS: dict[str, NDArray]  # Map of color names to RGB values
    COLOR_TO_IDX: dict[str, int] = field(init=False)
    IDX_TO_COLOR: dict[int, str] = field(init=False)
    IDX_TO_OBJECT: dict[int, str] = field(init=False)

    def __post_init__(self):
        self.COLOR_TO_IDX = dict(
            zip(self.COLORS.keys(), range(len(self.COLORS.keys())))
        )
        self.IDX_TO_COLOR = dict(
            zip(self.COLOR_TO_IDX.values(), self.COLOR_TO_IDX.keys())
        )
        self.IDX_TO_OBJECT = dict(
            zip(self.OBJECT_TO_IDX.values(), self.OBJECT_TO_IDX.keys())
        )


DefaultWorld = World(
    encode_dim=6,
    normalize_obs=1,
    COLORS=COLORS,
    OBJECT_TO_IDX={
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
    },
)

CollectWorld = World(
    encode_dim=3,
    normalize_obs=1,
    COLORS=COLORS,
    OBJECT_TO_IDX={
        "empty": 0,
        "wall": 1,
        "ball": 2,
        "agent": 3,
    },
)

CtfColors: dict[str, NDArray] = {
    "red": np.array([228, 3, 3]),
    "orange": np.array([255, 140, 0]),
    "yellow": np.array([255, 237, 0]),
    "green": np.array([0, 128, 38]),
    "blue": np.array([0, 77, 255]),
    "purple": np.array([117, 7, 135]),
    "brown": np.array([120, 79, 23]),
    "grey": np.array([100, 100, 100]),
    "light_red": np.array([255, 228, 225]),
    "light_blue": np.array([240, 248, 255]),
    "white": np.array([255, 250, 250]),
    "red_grey": np.array([170, 152, 169]),
    "blue_grey": np.array([140, 146, 172]),
}

CtfWorld = World(
    encode_dim=3,
    normalize_obs=1,
    COLORS=CtfColors,
    OBJECT_TO_IDX={
        "blue_territory": 0,
        "red_territory": 1,
        "blue_agent": 2,
        "red_agent": 3,
        "blue_flag": 4,
        "red_flag": 5,
        "obstacle": 6,
    },
)

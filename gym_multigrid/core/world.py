from typing import TypeVar
from dataclasses import dataclass, field

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

WildfireWorld = World(
    encode_dim=3,
    normalize_obs=1,
    COLORS=COLORS,
    OBJECT_TO_IDX={
        "empty": 0,
        "tree": 1,
        "agent": 2,
        "wall": 3,
    },
)

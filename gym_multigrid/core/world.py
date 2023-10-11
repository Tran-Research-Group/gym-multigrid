from typing import TypeVar
from dataclasses import dataclass, field
from .constants import COLOR_TO_IDX

WorldT = TypeVar("WorldT", bound="World")


@dataclass
class World:
    encode_dim: int
    normalize_obs: int
    COLOR_TO_IDX: dict[str, int]
    OBJECT_TO_IDX: dict[str, int]  # Map of object type to integers
    IDX_TO_COLOR: dict[int, str] = field(init=False)
    IDX_TO_OBJECT: dict[int, str] = field(init=False)

    def __post_init__(self):
        self.IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))
        self.IDX_TO_OBJECT = dict(
            zip(self.OBJECT_TO_IDX.values(), self.OBJECT_TO_IDX.keys())
        )


DefaultWorld = World(
    encode_dim=6,
    normalize_obs=1,
    COLOR_TO_IDX=COLOR_TO_IDX,
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
    COLOR_TO_IDX=COLOR_TO_IDX,
    OBJECT_TO_IDX={
        "empty": 0,
        "wall": 1,
        "ball": 2,
        "agent": 3,
    },
)

import numpy as np
from gym_multigrid.core.constants import (
    ACCESSIBLE_COLORS,
    COLORS,
    CTF_COLORS,
    LABYRINTH_COLORS,
)
from numpy.typing import NDArray
from pydantic import Field, dataclasses


@dataclasses.dataclass(config={"arbitrary_types_allowed": True})
class World:
    """This class defines the world within which grid is situated."""

    encode_dim: int
    normalize_obs: int
    OBJECT_TO_IDX: dict[str, int]  # Map of object type to integers
    COLORS: dict[str, NDArray[np.uint8]] = Field(default=COLORS)
    COLOR_TO_IDX: dict[str, int] = Field(init=False)
    IDX_TO_COLOR: dict[int, str] = Field(init=False)
    IDX_TO_OBJECT: dict[int, str] = Field(init=False)

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

FRWorld = World(
    encode_dim=3,
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
    },
)

LBFWorld = World(
    encode_dim=3,
    normalize_obs=1,
    COLORS=ACCESSIBLE_COLORS,
    OBJECT_TO_IDX={
        "empty": 0,
        "wall": 1,
        "fruit": 2,
        "goal": 3,
        "agent": 4,
    },
)


TeamNavigationWorld = World(
    encode_dim=3,
    normalize_obs=1,
    COLORS=ACCESSIBLE_COLORS,
    OBJECT_TO_IDX={
        "empty": 0,
        "wall": 1,
        "goal": 2,
        "agent": 3,
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

CtfWorld = World(
    encode_dim=3,
    normalize_obs=1,
    COLORS=CTF_COLORS,
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

MazeWorld = World(
    encode_dim=3,
    normalize_obs=1,
    COLORS=ACCESSIBLE_COLORS,
    OBJECT_TO_IDX={
        "background": 0,
        "agent": 1,
        "flag": 2,
        "wall": 3,
    },
)

LabyrinthWorld = World(
    encode_dim=3,
    normalize_obs=1,
    COLORS=LABYRINTH_COLORS,
    OBJECT_TO_IDX={
        "unseen": 0,
        "empty": 1,
        "wall": 2,
        "agent": 3,
        "door": 4,
        "goal": 5,
        "button": 6,
        "box": 7,
        "zone": 8,
        "blue_zone": 9,
        "red_zone": 10,
        "purple_zone": 11,
    },
)

GridWorld = World(
    encode_dim=1,
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
        "obstacle": 13,
    },
)

RoomsWorld = World(
    encode_dim=1,
    normalize_obs=1,
    COLORS=ACCESSIBLE_COLORS,
    OBJECT_TO_IDX={
        "empty": 0,
        "wall": 1,
        "agent": 2,
        "goal": 3,
        "lava": 4,
        "hole": 5,
        "floor": 6,
        "door": 7,
        "key": 8,
        "ball": 9,
        "box": 10,
        "objgoal": 11,
    },
)

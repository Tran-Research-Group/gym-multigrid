from typing import TypeVar
from dataclasses import dataclass, field
from numpy.typing import NDArray

from gym_multigrid.core.constants import (
    COLORS,
    CTF_COLORS,
    MAZE_COLORS,
    LABYRINTH_COLORS,
)

WorldT = TypeVar("WorldT", bound="World")


@dataclass
class World:
    """This class defines the world within which grid is situated."""

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

SaveTheCityWorld = World(
    encode_dim=4,  # (Object Type, Color Index, Construction Progress, Fire Progress)
    normalize_obs=1,  # Normalize observations
    COLORS={  # Define colors used in the environment
        "red": [255, 0, 0],  # Fire
        "gray": [128, 128, 128],  # Buildings
        "blue": [0, 0, 255],  # Firefighter
        "green": [0, 255, 0],  # Builder
        "yellow": [255, 255, 0],  # Generalist
    },
    OBJECT_TO_IDX={  # Define objects in the environment
        "unseen": 0,  # Not visible in agent's observation
        "empty": 1,  # Empty cell
        "wall": 2,  # Boundary walls
        "fire": 3,  # Fire spreading on buildings
        "fast_burning_building": 4,  # Burns quickly
        "slow_burning_building": 5,  # Burns slowly
        "firefighter": 6,  # Firefighter agent
        "builder": 7,  # Builder agent
        "generalist": 8,  # Generalist agent
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
    COLORS=MAZE_COLORS,
    OBJECT_TO_IDX={
        "background": 0,
        "agent": 1,
        "flag": 2,
        "obstacle": 3,
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
        "block": 4,
        "goal": 5,
        "button": 6,
        "box": 7,
        "zone": 8,
        "blue_zone": 9,
        "red_zone": 10,
        "purple_zone": 11,
    },
)

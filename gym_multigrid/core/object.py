from typing import TypeVar
import numpy as np
from numpy.typing import NDArray

from gym_multigrid.core.world import WorldT
from gym_multigrid.typing import Position
from ..utils.rendering import *

WorldObjT = TypeVar("WorldObjT", bound="WorldObj")


class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, world: WorldT, type: str, color: str):
        assert type in world.OBJECT_TO_IDX, type
        assert color in world.COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.contains = None
        self.world = world

        # Initial position of the object
        self.init_pos: Position | None = None

        # Current position of the object
        self.pos: Position | None = None

    def can_overlap(self) -> bool:
        """Can the agent overlap with this?"""
        return False

    def can_pickup(self) -> bool:
        """Can the agent pick this up?"""
        return False

    def can_contain(self) -> bool:
        """Can this contain another object?"""
        return False

    def see_behind(self) -> bool:
        """Can the agent see behind this object?"""
        return True

    def toggle(self, env, pos):
        """Method to trigger/toggle an action this object performs"""
        return False

    def encode(self, current_agent: bool = False) -> tuple[int, ...]:
        """Encode the a description of this object as a 3-tuple of integers"""
        if self.world.encode_dim == 3:
            return (
                self.world.OBJECT_TO_IDX[self.type],
                self.world.COLOR_TO_IDX[self.color],
                0,
            )
        else:
            return (
                self.world.OBJECT_TO_IDX[self.type],
                self.world.COLOR_TO_IDX[self.color],
                0,
                0,
                0,
                0,
            )

    @staticmethod
    def decode(type_idx, color_idx, state):
        assert False, "not implemented"

    def render(self, r: NDArray) -> None:
        """Draw this object with the given renderer"""
        raise NotImplementedError


class ObjectGoal(WorldObj):
    def __init__(
        self, world: WorldT, index: int, target_type: str = "ball", reward=1, color=None
    ):
        if color is None:
            super().__init__(world, "objgoal", world.IDX_TO_COLOR[index])
        else:
            super().__init__(world, "objgoal", world.IDX_TO_COLOR[color])
        self.target_type = target_type
        self.index = index
        self.reward = reward

    def can_overlap(self):
        return False

    def render(self, img: NDArray):
        fill_coords(img, point_in_rect(0, 1, 0, 1), self.world.COLORS[self.color])


class Goal(WorldObj):
    def __init__(self, world: WorldT, index: int, reward=1, color=None):
        if color is None:
            super().__init__(world, "goal", world.IDX_TO_COLOR[index])
        else:
            super().__init__(world, "goal", world.IDX_TO_COLOR[color])
        self.index = index
        self.reward = reward

    def can_overlap(self):
        return True

    def render(self, img: NDArray):
        fill_coords(img, point_in_rect(0, 1, 0, 1), self.world.COLORS[self.color])


class Switch(WorldObj):
    def __init__(self, world: WorldT):
        super().__init__(world, "switch", world.IDX_TO_COLOR[0])

    def can_overlap(self):
        return True

    def render(self, img: NDArray):
        fill_coords(img, point_in_rect(0, 1, 0, 1), self.world.COLORS[self.color])


class Floor(WorldObj):
    """
    Colored floor tile the agent can walk over
    """

    def __init__(self, world: WorldT, color: str = "blue"):
        super().__init__(world, "floor", color)

    def can_overlap(self) -> bool:
        return True

    def render(self, img: NDArray):
        fill_coords(img, point_in_rect(0, 1, 0, 1), self.world.COLORS[self.color])


class Lava(WorldObj):
    def __init__(self, world):
        super().__init__(world, "lava", "red")

    def can_overlap(self):
        return True

    def render(self, img):
        c = (255, 128, 0)

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))


class Wall(WorldObj):
    def __init__(self, world, color="grey"):
        super().__init__(world, "wall", color)

    def see_behind(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), self.world.COLORS[self.color])


class Obstacle(WorldObj):
    def __init__(
        self,
        world: WorldT,
        penalty: float = 0,
        can_see_through: bool = True,
        color: str = "grey",
    ):
        super().__init__(world, "obstacle", color)
        self.penalty = penalty
        self.can_see_through = can_see_through

    def see_behind(self):
        return self.can_see_through

    def can_overlap(self):
        return True if self.penalty != 0 else False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), self.world.COLORS[self.color])


class Door(WorldObj):
    def __init__(self, world, color, is_open=False, is_locked=False):
        super().__init__(world, "door", color)
        self.is_open = is_open
        self.is_locked = is_locked

    def can_overlap(self):
        """The agent can only walk over this cell when the door is open"""
        return self.is_open

    def see_behind(self):
        return self.is_open

    def toggle(self, env, pos):
        # If the player has the right key to open the door
        if self.is_locked:
            if isinstance(env.carrying, Key) and env.carrying.color == self.color:
                self.is_locked = False
                self.is_open = True
                return True
            return False

        self.is_open = not self.is_open
        return True

    def encode(self, current_agent: bool = False):
        """Encode the a description of this object as a 3-tuple of integers"""

        # State, 0: open, 1: closed, 2: locked
        if self.is_open:
            state = 0
        elif self.is_locked:
            state = 2
        elif not self.is_open:
            state = 1
        else:
            raise ValueError("Invalid door state")

        return (
            self.world.OBJECT_TO_IDX[self.type],
            self.world.COLOR_TO_IDX[self.color],
            state,
            0,
            0,
            0,
        )

    def render(self, img):
        c = self.world.COLORS[self.color]

        if self.is_open:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))
            return

        # Door frame and door
        if self.is_locked:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * np.array(c))

            # Draw key slot
            fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), c)
        else:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))

            # Draw door handle
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)


class Key(WorldObj):
    def __init__(self, world, color="blue"):
        super(Key, self).__init__(world, "key", color)

    def can_pickup(self):
        return True

    def render(self, img):
        c = self.world.COLORS[self.color]

        # Vertical quad
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)

        # Teeth
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)

        # Ring
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0, 0, 0))


class Ball(WorldObj):
    def __init__(self, world, index=0, reward=2):
        super(Ball, self).__init__(world, "ball", self.world.IDX_TO_COLOR[index])
        self.index = index
        self.reward = reward

    def can_pickup(self):
        return True

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), self.world.COLORS[self.color])


class Box(WorldObj):
    def __init__(self, world, color, contains=None):
        super(Box, self).__init__(world, "box", color)
        self.contains = contains

    def can_pickup(self):
        return True

    def render(self, img):
        c = self.world.COLORS[self.color]

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0, 0, 0))

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)

    def toggle(self, env, pos):
        # Replace the box by its contents
        env.grid.set(*pos, self.contains)
        return True


class Flag(WorldObj):
    def __init__(
        self,
        world: WorldT,
        index: int,
        type: str = "flag",
        color: str = "blue",
    ):
        super().__init__(world, type, color)
        self.index: int = index

    def can_pickup(self):
        return True

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), self.world.COLORS[self.color])

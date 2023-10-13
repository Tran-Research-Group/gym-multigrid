import enum
import math
from typing import Type, TypeVar

import numpy as np
from numpy.typing import NDArray

from gym_multigrid.core.world import WorldT
from gym_multigrid.policy.base import AgentPolicyT
from gym_multigrid.typing import Position
from ..utils.rendering import point_in_triangle, rotate_fn, fill_coords
from .object import WorldObj
from .constants import COLORS, DIR_TO_VEC

ActionsT = TypeVar("ActionsT", bound=enum.IntEnum)


class DefaultActions(enum.IntEnum):
    still = 0
    left = 1
    right = 2
    forward = 3
    pickup = 4
    drop = 5
    toggle = 6
    done = 7


class CollectActions(enum.IntEnum):
    north = 0
    east = 1
    south = 2
    west = 3


class SmallActions(enum.IntEnum):
    still = 0
    left = 1
    right = 2
    forward = 3


class MineActions(enum.IntEnum):
    still = 0
    left = 1
    right = 2
    forward = 3
    build = 4


AgentT = TypeVar("AgentT", bound="Agent")


class Agent(WorldObj):
    def __init__(
        self,
        world: WorldT,
        index: int = 0,
        view_size: int = 7,
        actions: Type[ActionsT] = DefaultActions,
        dir_to_vec: list[NDArray] = DIR_TO_VEC,
        color: str | None = None,
    ):
        if color is None:
            color = world.IDX_TO_COLOR[index]
        else:
            pass

        super(Agent, self).__init__(world, "agent", color)
        self.pos: Position | None = None
        self.dir: int | None = None
        self.init_dir: int | None = None
        self.index = index
        self.view_size = view_size
        self.carrying = None
        self.terminated = False
        self.started = True
        self.paused = False
        self.actions = actions
        self.world = world
        self.dir_to_vec = dir_to_vec

    def render(self, img):
        c = COLORS[self.color]
        tri_fn = point_in_triangle(
            (0.12, 0.19),
            (0.87, 0.50),
            (0.12, 0.81),
        )
        # Rotate the agent based on its direction
        assert self.dir is not None
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * self.dir)
        fill_coords(img, tri_fn, c)

    def encode(self, current_agent=False):
        """Encode the a description of this object as a 3-tuple of integers"""
        if self.world.encode_dim == 3:
            return (
                self.world.OBJECT_TO_IDX[self.type],
                self.world.COLOR_TO_IDX[self.color],
                self.dir,
            )
        elif self.carrying:
            if current_agent:
                return (
                    self.world.OBJECT_TO_IDX[self.type],
                    self.world.COLOR_TO_IDX[self.color],
                    self.world.OBJECT_TO_IDX[self.carrying.type],
                    self.world.COLOR_TO_IDX[self.carrying.color],
                    self.dir,
                    1,
                )
            else:
                return (
                    self.world.OBJECT_TO_IDX[self.type],
                    self.world.COLOR_TO_IDX[self.color],
                    self.world.OBJECT_TO_IDX[self.carrying.type],
                    self.world.COLOR_TO_IDX[self.carrying.color],
                    self.dir,
                    0,
                )

        else:
            if current_agent:
                return (
                    self.world.OBJECT_TO_IDX[self.type],
                    self.world.COLOR_TO_IDX[self.color],
                    0,
                    0,
                    self.dir,
                    1,
                )
            else:
                return (
                    self.world.OBJECT_TO_IDX[self.type],
                    self.world.COLOR_TO_IDX[self.color],
                    0,
                    0,
                    self.dir,
                    0,
                )

    @property
    def dir_vec(self):
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """

        assert self.dir is not None
        assert self.dir >= 0 and self.dir < len(self.dir_to_vec)
        return self.dir_to_vec[self.dir]

    @property
    def right_vec(self):
        """
        Get the vector pointing to the right of the agent.
        """

        dx, dy = self.dir_vec
        return np.array((-dy, dx))

    @property
    def front_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """

        return self.pos + self.dir_vec

    def west_pos(self):
        """
        Get the position of the cell to the left of the agent
        """
        if self.pos is None:
            raise ValueError("Agent position is not set")
        else:
            return self.pos + np.array([-1, 0])

    def east_pos(self):
        """
        Get the position of the cell to the right of the agent
        """
        if self.pos is None:
            raise ValueError("Agent position is not set")
        else:
            return self.pos + np.array([1, 0])

    def north_pos(self):
        """
        Get the position of the cell above the agent
        """
        if self.pos is None:
            raise ValueError("Agent position is not set")
        else:
            return self.pos + np.array([0, -1])

    def south_pos(self):
        """
        Get the position of the cell below the agent
        """
        if self.pos is None:
            raise ValueError("Agent position is not set")
        else:
            return self.pos + np.array([0, 1])

    def get_view_coords(self, i, j):
        """
        Translate and rotate absolute grid coordinates (i, j) into the
        agent's partially observable view (sub-grid). Note that the resulting
        coordinates may be negative or outside of the agent's view size.
        """

        assert self.pos is not None
        ax, ay = self.pos
        dx, dy = self.dir_vec
        rx, ry = self.right_vec

        # Compute the absolute coordinates of the top-left view corner
        sz = self.view_size
        hs = self.view_size // 2
        tx = ax + (dx * (sz - 1)) - (rx * hs)
        ty = ay + (dy * (sz - 1)) - (ry * hs)

        lx = i - tx
        ly = j - ty

        # Project the coordinates of the object relative to the top-left
        # corner onto the agent's own coordinate system
        vx = rx * lx + ry * ly
        vy = -(dx * lx + dy * ly)

        return vx, vy

    def get_view_exts(self):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        """

        assert self.pos is not None

        # Facing right
        if self.dir == 0:
            topX = self.pos[0]
            topY = self.pos[1] - self.view_size // 2
        # Facing down
        elif self.dir == 1:
            topX = self.pos[0] - self.view_size // 2
            topY = self.pos[1]
        # Facing left
        elif self.dir == 2:
            topX = self.pos[0] - self.view_size + 1
            topY = self.pos[1] - self.view_size // 2
        # Facing up
        elif self.dir == 3:
            topX = self.pos[0] - self.view_size // 2
            topY = self.pos[1] - self.view_size + 1
        else:
            assert False, "invalid agent direction"

        botX = topX + self.view_size
        botY = topY + self.view_size

        return (topX, topY, botX, botY)

    def relative_coords(self, x, y):
        """
        Check if a grid position belongs to the agent's field of view, and returns the corresponding coordinates
        """

        vx, vy = self.get_view_coords(x, y)

        if vx < 0 or vy < 0 or vx >= self.view_size or vy >= self.view_size:
            return None

        return vx, vy

    def in_view(self, x, y):
        """
        check if a grid position is visible to the agent
        """

        return self.relative_coords(x, y) is not None


class PolicyAgent(Agent):
    """
    Agent with a policy that determines its actions
    """

    def __init__(
        self,
        policy: Type[AgentPolicyT],
        world: WorldT,
        index: int = 0,
        view_size: int = 7,
        actions: type[ActionsT] = DefaultActions,
        dir_to_vec: list[NDArray] = DIR_TO_VEC,
        color: str | None = None,
    ):
        super().__init__(world, index, view_size, actions, dir_to_vec, color)
        self.policy: AgentPolicyT = policy()

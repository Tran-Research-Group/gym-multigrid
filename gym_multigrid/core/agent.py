import enum
import math
from typing import Type, TypeVar
import numpy as np
from numpy.typing import NDArray
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.world import WorldT
from gym_multigrid.policy.base import AgentPolicyT
from gym_multigrid.typing import Position
from gym_multigrid.utils.rendering import point_in_triangle, rotate_fn, fill_coords
from gym_multigrid.core.object import WorldObj
from gym_multigrid.core.constants import DIR_TO_VEC

ActionsT = TypeVar("ActionsT", bound=enum.IntEnum)


class DefaultActions(enum.IntEnum):
    """Set of DefaultActions

    Parameters
    ----------
    enum : IntEnum
        Base class for creating enumerated constants that are also subclasses of int.
    """

    STILL = 0
    LEFT = 1
    RIGHT = 2
    FORWARD = 3
    PICKUP = 4
    DROP = 5
    TOGGLE = 6
    DONE = 7


class CollectActions(enum.IntEnum):
    """Set of actions available for the agents in Collect Game environment

    Parameters
    ----------
    enum : IntEnum
        Base class for creating enumerated constants that are also subclasses of int.
    """

    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


class WildfireActions(enum.IntEnum):
    """Set of actions available for the agents in Wildfire environment

    Parameters
    ----------
    enum : IntEnum
        Base class for creating enumerated constants that are also subclasses of int.
    """

    STILL = 0
    NORTH = 1
    EAST = 2
    SOUTH = 3
    WEST = 4


AgentT = TypeVar("AgentT", bound="Agent")


class Agent(WorldObj):
    """Defines the class for an agent in the environment"""

    def __init__(
        self,
        world: WorldT,
        index: int = 0,
        view_size: int = 7,
        actions: Type[ActionsT] = DefaultActions,
        dir_to_vec: list[NDArray] = DIR_TO_VEC,
        color: str | None = None,
        bg_color: str | None = None,
        type: str = "agent",
    ):
        """Initialize the agent object

        Parameters
        ----------
        world : WorldT
            the world within which grid is situated
        index : int, optional
            a number useful to identify the instantiated agent, by default 0
        view_size : int, optional
            the size of agent view, if partial observability holds, by default 7
        actions : Type[ActionsT], optional
            set of actions available to the agent, by default DefaultActions
        dir_to_vec : list[NDArray], optional
            map of agent direction indices to vectors, by default DIR_TO_VEC
        color : str | None, optional
            color of the agent, by default None
        bg_color : str | None, optional
            background color of the tile containing agent, by default None
        type : str, optional
            type of the world object, by default "agent"
        """
        if color is None:
            color = world.IDX_TO_COLOR[index]
        else:
            pass

        super().__init__(world, type, color, bg_color)
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

    def render(self, img: NDArray):
        """Render the agent at its current position

        Parameters
        ----------
        img : NDArray
            the image to render the agent in
        """
        c = self.world.COLORS[self.color]
        tri_fn = point_in_triangle(
            (0.12, 0.19),
            (0.87, 0.50),
            (0.12, 0.81),
        )
        # Rotate the agent based on its direction
        assert self.dir is not None
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * self.dir)
        fill_coords(
            img, tri_fn, c, self.world.COLORS[self.bg_color] if self.bg_color else None
        )

    def encode(self, current_agent=False):
        """Encode a description of this object as a 3-tuple of integers

        Parameters
        ----------
        current_agent : bool, optional
            whether the agent is the current agent, by default False
        """
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

    def move(
        self,
        next_pos: Position,
        grid: Grid,
        init_grid: Grid | None = None,
        dummy_move: bool = False,
        bg_color: str | None = None,
    ):
        """Move the agent to a new position

        Parameters
        ----------
        next_pos : Position
            the position to move the agent to
        grid : Grid
            the grid to move the agent in
        init_grid : Grid | None, optional
            the initial grid before agent is moved, by default None
        dummy_move : bool, optional
            whether the move is a dummy move, by default False
        bg_color : str | None, optional
            the background color of the tile containing agent, by default None
        """
        if self.pos is not None:
            direction = np.array(next_pos) - np.array(self.pos)

            for i, vec in enumerate(self.dir_to_vec):
                if np.array_equal(vec, direction):
                    self.dir = i
                    break

            if init_grid is not None:
                grid.set(*self.pos, init_grid.get(*self.pos))
        else:
            pass

        if dummy_move:
            pass
        else:
            self.pos = next_pos

        assert self.pos is not None
        grid.set(*self.pos, self)

        if bg_color is not None:
            self.bg_color = bg_color
        else:
            pass

    @property
    def dir_vec(self):
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.

        Returns
        -------
        NDArray
            the direction vector for forward movement in the current orientation of the agent
        """

        assert self.dir is not None
        assert self.dir >= 0 and self.dir < len(self.dir_to_vec)
        return self.dir_to_vec[self.dir]

    @property
    def right_vec(self):
        """
        Get the vector pointing to the right of the agent.

        Returns
        -------
        NDArray
            the vector pointing to the right of the agent
        """

        dx, dy = self.dir_vec
        return np.array((-dy, dx))

    @property
    def front_pos(self):
        """
        Get the position of the cell that is right in front of the agent

        Returns
        -------
        NDArray
            the position of the cell that is right in front of the agent
        """

        return self.pos + self.dir_vec

    def west_pos(self):
        """
        Get the position of the cell to the left of the agent

        Returns
        -------
        NDArray
            the position of the cell to the left of the agent
        """
        if self.pos is None:
            raise ValueError("Agent position is not set")
        else:
            return self.pos + np.array([-1, 0])

    def east_pos(self):
        """
        Get the position of the cell to the right of the agent

        Returns
        -------
        NDArray
            the position of the cell to the right of the agent
        """
        if self.pos is None:
            raise ValueError("Agent position is not set")
        else:
            return self.pos + np.array([1, 0])

    def north_pos(self):
        """
        Get the position of the cell above the agent

        Returns
        -------
        NDArray
            the position of the cell above the agent
        """
        if self.pos is None:
            raise ValueError("Agent position is not set")
        else:
            return self.pos + np.array([0, -1])

    def south_pos(self):
        """
        Get the position of the cell below the agent

        Returns
        -------
        NDArray
            the position of the cell below the agent
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

        Parameters
        ----------
        i : int
            x-coordinate in the grid
        j : int
            y-coordinate in the grid

        Returns
        -------
        tuple
            the coordinates of the grid in the agent's view
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

        Returns
        -------
        tuple
            the extents of the square set of tiles visible to the agent
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

        Parameters
        ----------
        x : int
            x-coordinate in the grid
        y : int
            y-coordinate in the grid

        Returns
        -------
        tuple | None
            the coordinates of the grid position in the agent's view or None if the position is not visible
        """

        vx, vy = self.get_view_coords(x, y)

        if vx < 0 or vy < 0 or vx >= self.view_size or vy >= self.view_size:
            return None

        return vx, vy

    def in_view(self, x, y):
        """
        check if a grid position is visible to the agent

        Parameters
        ----------
        x : int
            x-coordinate in the grid
        y : int
            y-coordinate in the grid

        Returns
        -------
        bool
            whether the grid position is visible to the agent
        """

        return self.relative_coords(x, y) is not None


class PolicyAgent(Agent):
    """
    Agent with a policy that determines its actions
    """

    def __init__(
        self,
        policy: AgentPolicyT,
        world: WorldT,
        index: int = 0,
        view_size: int = 7,
        actions: type[ActionsT] = DefaultActions,
        dir_to_vec: list[NDArray] = DIR_TO_VEC,
        color: str | None = None,
        bg_color: str | None = None,
        type: str = "agent",
    ):
        """Initialize the PolicyAgent object

        Parameters
        ----------
        policy : AgentPolicyT
            the policy that determines the agent's actions
        world : WorldT
            the world within which grid is situated
        index : int, optional
            a number useful to identify the instantiated agent, by default 0
        view_size : int, optional
            the size of agent view, if partial observability holds, by default 7
        actions : type[ActionsT], optional
            set of actions available to the agent, by default DefaultActions
        dir_to_vec : list[NDArray], optional
            map of agent direction indices to vectors, by default DIR_TO_VEC
        color : str | None, optional
            color of the agent, by default None
        bg_color : str | None, optional
            background color of the tile containing agent, by default None
        type : str, optional
            type of the world object, by default "agent"
        """
        super().__init__(
            world, index, view_size, actions, dir_to_vec, color, bg_color, type
        )
        self.policy: AgentPolicyT = policy

from typing import TypeVar, Final
import numpy as np
from numpy.typing import NDArray
from gym_multigrid.core.world import WorldT
from gym_multigrid.typing import Position
from gym_multigrid.utils.rendering import *
from gym_multigrid.core.constants import STATE_IDX_TO_COLOR_WILDFIRE


WorldObjT = TypeVar("WorldObjT", bound="WorldObj")


class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(
        self,
        world: WorldT,
        type: str = "base",
        color: str = "grey",
        bg_color: str | None = None,
    ):
        """Create a WorldObj object

        Parameters
        ----------
        world : WorldT
            the world in which the object exists
        type : str, optional
            type of the object, by default "base"
        color : str, optional
            color of the object, by default "grey"
        bg_color : str | None, optional
            background color of the tile containing object, by default None
        """
        assert type in world.OBJECT_TO_IDX, type
        assert color in world.COLOR_TO_IDX, color
        self.type: str = type
        self.color: str = color
        self.init_color: Final[str] = color
        self.contains = None
        self.world = world
        self.bg_color: str | None = bg_color
        self.init_bg_color: Final[str | None] = bg_color

        # Initial position of the object
        self.init_pos: Final[Position | None] = None

        # Current position of the object
        self._pos: Position | None = None

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        self._pos = value

    def reset(self) -> None:
        """
        Reset the object to its initial state.
        This method can be called before the start of every episode.
        """
        self.pos = self.init_pos
        self.contains = None
        self.color = self.init_color
        self.bg_color = self.init_bg_color

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

    def toggle(self, env, pos: Position):
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
    def decode(type_idx: int, color_idx: int, state: int):
        assert False, "not implemented"

    def render(self, r: NDArray) -> None:
        """Draw this object with the given renderer"""
        raise NotImplementedError


class ObjectGoal(WorldObj):
    def __init__(
        self,
        world: WorldT,
        index: int,
        target_type: str = "ball",
        reward: float = 1,
        color: int | None = None,
    ):
        if color is None:
            super().__init__(world, "objgoal", world.IDX_TO_COLOR[index])
        else:
            super().__init__(world, "objgoal", world.IDX_TO_COLOR[color])
        self.target_type: str = target_type
        self.index: int = index
        self.reward: float = reward

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

    def __init__(self, world: WorldT, color: str = "blue", type: str = "floor"):
        super().__init__(world, type, color)

    def can_overlap(self) -> bool:
        return True

    def render(self, img: NDArray):
        fill_coords(img, point_in_rect(0, 1, 0, 1), self.world.COLORS[self.color])


class Zone(Floor):
    """
    Alias for Floor
    """

    def __init__(self, world: WorldT, color: str = "blue", type: str = "zone"):
        super().__init__(world, color, type)


class Lava(WorldObj):
    def __init__(self, world: WorldT):
        super().__init__(world, "lava", "red")

    def can_overlap(self):
        return True

    def render(self, img: NDArray):
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
    def __init__(self, world: WorldT, color: str = "grey"):
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
    def __init__(
        self,
        world: WorldT,
        color: str,
        is_open: bool = False,
        is_locked: bool = False,
    ):
        super().__init__(world, "door", color)
        self.is_open: bool = is_open
        self.is_locked: bool = is_locked

    def can_overlap(self):
        """The agent can only walk over this cell when the door is open"""
        return self.is_open

    def see_behind(self):
        return self.is_open

    def toggle(self, env, pos: Position):
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
        state: int
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
    def __init__(self, world: WorldT, color: str = "blue"):
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
    def __init__(self, world: WorldT, index: int = 0, reward: float = 2):
        super().__init__(world, "ball", world.IDX_TO_COLOR[index])
        self.index = index
        self.reward = reward

    def can_pickup(self):
        return True

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), self.world.COLORS[self.color])

class Building(WorldObj):
    """
    Represents a building that can be constructed or burned down.
    """

    def __init__(self, world, burn_rate: int, build_speed: int, firefight_speed: int, fast_burning: bool = False):
        super().__init__(world, type="building", color="gray")
        
        self.burn_rate = burn_rate  # How fast the building burns if on fire
        self.build_speed = build_speed  # How fast the building is constructed
        self.firefight_speed = firefight_speed  # How fast firefighters put out fires
        self.fast_burning = fast_burning  # If True, burns 4x faster

        self.building_state = 50  # Starts at 50 (partially built)
        self.fire_rate = 0  # No fire initially

    def step(self):
        """Update building state per timestep"""
        if self.fire_rate > 0 and self.building_state > 0:
            # Fire decreases building state
            burn_speed = self.fire_rate * (4 if self.fast_burning else 1)
            self.building_state -= burn_speed

            # If building state reaches 0, it's burned down
            if self.building_state <= 0:
                self.building_state = 0
                return "burned_down"  # Trigger game penalty

        return None

    def build(self, builder_speed: int):
        """Increase construction progress, only if not burning."""
        if self.fire_rate == 0:  # Can only build if fire is out
            self.building_state += builder_speed
            if self.building_state >= 100:
                self.building_state = 100
                return "completed"  # Reward agents for finishing construction
        return None

    def burn(self):
        """Start fire if not already burning."""
        if self.fire_rate == 0 and self.building_state > 0:
            self.fire_rate = self.burn_rate  # Fire starts at the burn rate

    def fight_fire(self):
        """Firefighters reduce fire. If fire_rate reaches 0, fire is extinguished."""
        self.fire_rate = max(0, self.fire_rate - self.firefight_speed)

    def encode(self, current_agent: bool = False) -> tuple[int, ...]:
        """Encode the building state for observation."""
        return (
            self.world.OBJECT_TO_IDX[self.type],  # Object Type (Building)
            self.world.COLOR_TO_IDX[self.color],  # Color (Gray)
            int(self.building_state),  # Building state (0-100)
            int(self.fire_rate),  # Current fire intensity
        )
    
    def render(self, img):
        """
        Render the building in the environment.
        - Darker color for more construction progress.
        - Red overlay if the building is on fire.
        """

        # Get base color (gray) and modify brightness based on construction progress
        base_color = np.array(self.world.COLORS[self.color])
        brightness_factor = self.building_state / 100  # Scale from 0 (dark) to 1 (full brightness)
        building_color = (base_color * brightness_factor).astype(int)

        # Draw the main building as a rectangle (since buildings aren’t round like balls)
        fill_coords(img, point_in_rect(0.2, 0.8, 0.2, 0.8), tuple(building_color))

        # If the building is on fire, overlay a red glow
        if self.fire_rate > 0:
            fire_intensity = min(1, self.fire_rate / 50)  # Scale fire effect
            fire_color = np.array([255, 0, 0]) * fire_intensity
            fill_coords(img, point_in_circle(0.5, 0.5, 0.4), tuple(fire_color))



class Box(WorldObj):
    def __init__(self, world: WorldT, color: str, contains=None):
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

    def toggle(self, env, pos: Position):
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
        bg_color: str = "light_blue",
    ):
        super().__init__(world, type, color, bg_color)
        self.index: int = index

    def can_pickup(self):
        return True

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(
            img,
            point_in_circle(0.5, 0.5, 0.31),
            self.world.COLORS[self.color],
            self.world.COLORS[self.bg_color] if self.bg_color else None,
        )


class Tree(WorldObj):
    def __init__(
        self,
        world: WorldT,
        tree_state_idx: int = 0,
        region: str = "common",
    ):
        super().__init__(world, "tree", STATE_IDX_TO_COLOR_WILDFIRE[tree_state_idx])
        self.state = tree_state_idx
        self.agent_above = False
        self.region = region

    def can_overlap(self):
        return True

    def encode(self, current_agent: bool = False):
        return (
            self.world.OBJECT_TO_IDX[self.type],
            self.world.COLOR_TO_IDX[self.color],
            self.state,
        )

    def render(self, img):
        c = self.world.COLORS[self.color]

        fill_coords(img, point_in_rect(0, 1, 0, 1), c)


class AgentGoal(WorldObj):
    def __init__(
        self,
        world: WorldT,
        accepting_agent_idx: int,
        goal_group: int,
        type: str = "goal",
        color: str = "yellow",
        bg_color: str | None = None,
    ):
        super().__init__(world, type, color, bg_color)
        self.accepting_agent_idx: int = accepting_agent_idx
        self.goal_group: int = goal_group

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), self.world.COLORS[self.color])


class Block(WorldObj):
    def __init__(
        self,
        world: WorldT,
    ):
        super().__init__(world, "block", color="light_grey")

        self.locked: bool = True

    def can_overlap(self) -> bool:
        return not self.locked

    def open(self) -> None:
        self.locked = False
        self.color = "grey"

    def see_behind(self) -> bool:
        return False

    def render(self, img):
        c = self.world.COLORS[self.color]

        # Door frame and door
        fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
        fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
        fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))

        # Draw door handle
        fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)

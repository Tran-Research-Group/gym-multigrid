import enum
from itertools import chain
from typing import Final, Literal

from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray

from gym_multigrid.core.agent import ActionsT, DefaultActions
from gym_multigrid.core.world import WorldT, World
from gym_multigrid.multigrid import MultiGridEnv
from gym_multigrid.typing import Position


CtfColors: dict[str, NDArray] = {
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


class CtfActions(enum.IntEnum):
    stay = 0
    left = 1
    down = 2
    right = 3
    up = 4


Ctf1v1World = World(
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


class Ctf1v1Env(MultiGridEnv):
    """
    Environment for capture the flag with two agents.
    """

    def __init__(
        self,
        map_path: str,
        max_steps: int = 100,
        see_through_walls: bool = False,
        agents=None,
        partial_obs: bool = False,
        agent_view_size: int = 10,
        actions_set: type[ActionsT] = CtfActions,
        world: WorldT = Ctf1v1World,
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
    ):
        self._map_path: Final[str] = map_path
        self._field_map: Final[NDArray] = np.loadtxt(map_path)

        height: int
        width: int
        height, width = self._field_map.shape

        self.world: Final[WorldT] = world

        self.obstacle: Final[list[Position]] = list(zip(*np.where(self._field_map == self.world.OBJECT_TO_IDX["obstacle"])))  # type: ignore

        blue_flag: Final[Position] = list(zip(*np.where(self._field_map == self.world.OBJECT_TO_IDX["blue_flag"])))[0]  # type: ignore

        red_flag: Final[Position] = list(zip(*np.where(self._field_map == self.world.OBJECT_TO_IDX["red_flag"])))[0]  # type: ignore

        self.blue_territory: Final[list[Position]] = list(zip(*np.where(self._field_map == self.world.OBJECT_TO_IDX["blue_territory"]))) + [blue_flag]  # type: ignore

        self.red_territory: Final[list[Position]] = list(zip(*np.where(self._field_map == self.world.OBJECT_TO_IDX["red_territory"]))) + [red_flag]  # type: ignore

        super().__init__(
            width=width,
            height=height,
            max_steps=max_steps,
            see_through_walls=see_through_walls,
            agents=agents,
            partial_obs=partial_obs,
            agent_view_size=agent_view_size,
            actions_set=actions_set,
            world=world,
            render_mode=render_mode,
        )

    def _set_observation_space(self) -> spaces.Dict:
        observation_space = spaces.Dict(
            {
                "blue_agent": spaces.Box(
                    low=np.array([-1, -1]), high=np.array(self._field_map.shape) - 1, dtype=int  # type: ignore
                ),
                "red_agent": spaces.Box(
                    low=np.array([-1, -1]), high=np.array(self._field_map.shape) - 1, dtype=int  # type: ignore
                ),
                "blue_flag": spaces.Box(
                    low=np.array([0, 0]), high=np.array(self._field_map.shape) - 1, dtype=int  # type: ignore
                ),
                "red_flag": spaces.Box(
                    low=np.array([0, 0]), high=np.array(self._field_map.shape) - 1, dtype=int  # type: ignore
                ),
                "blue_territory": spaces.Box(
                    low=np.array(list(chain.from_iterable([[0, 0] for _ in range(len(self.blue_territory))]))),  # type: ignore
                    high=np.array(list(chain.from_iterable([self._field_map.shape for _ in range(len(self.blue_territory))]))).flatten() - 1,  # type: ignore
                    dtype=int,  # type: ignore
                ),
                "red_territory": spaces.Box(
                    low=np.array(list(chain.from_iterable([[0, 0] for _ in range(len(self.red_territory))]))),  # type: ignore
                    high=np.array(list(chain.from_iterable([self._field_map.shape for _ in range(len(self.red_territory))]))).flatten() - 1,  # type: ignore
                    dtype=int,  # type: ignore
                ),
                "obstacle": spaces.Box(
                    low=np.array(list(chain.from_iterable([[0, 0] for _ in range(len(self.obstacle))]))),  # type: ignore
                    high=np.array(list(chain.from_iterable([self._field_map.shape for _ in range(len(self.obstacle))]))).flatten() - 1,  # type: ignore
                    dtype=int,  # type: ignore
                ),
                "is_red_agent_defeated": spaces.Discrete(2),
            }
        )

        return observation_space

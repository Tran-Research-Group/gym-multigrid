from itertools import chain
from typing import Final, Literal, TypeAlias, TypedDict

from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray

from gym_multigrid.core.agent import Agent, PolicyAgent, AgentT, FRActions
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.object import Floor, Flag, Obstacle, WorldObjT
from gym_multigrid.core.world import FRWorld, World
from gym_multigrid.multigrid import MultiGridEnv
from gym_multigrid.policy.ctf.heuristic import RwPolicy, CtfPolicyT
from gym_multigrid.typing import Position
from gym_multigrid.utils.map import distance_area_point, distance_points, load_text_map


class ObservationDict(TypedDict):
    blue_agent: NDArray[np.int_]
    red_agent: NDArray[np.int_]
    blue_flag: NDArray[np.int_]
    red_flag: NDArray[np.int_]
    blue_territory: NDArray[np.int_]
    red_territory: NDArray[np.int_]
    obstacle: NDArray[np.int_]
    is_red_agent_defeated: int


class MultiAgentObservationDict(TypedDict):
    blue_agent: NDArray[np.int_]
    red_agent: NDArray[np.int_]
    blue_flag: NDArray[np.int_]
    red_flag: NDArray[np.int_]
    blue_territory: NDArray[np.int_]
    red_territory: NDArray[np.int_]
    obstacle: NDArray[np.int_]
    terminated_agents: NDArray[np.int_]


Observation: TypeAlias = ObservationDict | MultiAgentObservationDict | NDArray[np.int_]


class FourRooms(MultiGridEnv):
    """
    Environment for capture the flag with multiple agents with N blue agents and M red agents.
    """

    def __init__(
        self,
        grid_size: tuple = (10, 10),
        max_steps: int = 100,
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
    ) -> None:
        """
        Initialize a new capture the flag environment.

        Parameters
        ----------

        """
        partial_obs: bool = False
        agent_view_size: int = 10

        self.width = grid_size[0]
        self.height = grid_size[1]
        self.grid_size = grid_size
        self.world = FRWorld
        self.actions_set = FRActions
        see_through_walls: bool = False

        agents = Agent(
            self.world,
            color="blue",
            bg_color="light_blue",
            view_size=agent_view_size,
            actions=self.actions_set,
            type="agent",
        )

        super().__init__(
            width=self.width,
            height=self.height,
            max_steps=max_steps,
            see_through_walls=see_through_walls,
            agents=agents,
            partial_obs=partial_obs,
            agent_view_size=agent_view_size,
            actions_set=self.actions_set,
            world=self.world,
            render_mode=render_mode,
        )

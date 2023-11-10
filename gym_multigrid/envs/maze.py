import enum
from itertools import chain
import random
from typing import Final, Literal, TypedDict, Type

from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray

from gym_multigrid.core.agent import Agent, PolicyAgent, AgentT
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.object import Floor, Flag, Obstacle, WorldObjT
from gym_multigrid.core.world import World
from gym_multigrid.multigrid import MultiGridEnv
from gym_multigrid.policy.base import AgentPolicyT
from gym_multigrid.policy.ctf.heuristic import RwPolicy
from gym_multigrid.typing import Position
from gym_multigrid.utils.map import distance_area_point, distance_points

MazeColors: dict[str, NDArray] = {
    "red": np.array([228, 3, 3]),
    "orange": np.array([255, 140, 0]),
    "yellow": np.array([255, 237, 0]),
    "green": np.array([0, 128, 38]),
    "blue": np.array([0, 77, 255]),
    "purple": np.array([117, 7, 135]),
    "brown": np.array([120, 79, 23]),
    "grey": np.array([100, 100, 100]),
    "light_red": np.array([255, 228, 225]),
    "light_blue": np.array([240, 248, 255]),
    "white": np.array([255, 250, 250]),
}


class MazeActions(enum.IntEnum):
    stay = 0
    left = 1
    down = 2
    right = 3
    up = 4


MazeWorld = World(
    encode_dim=3,
    normalize_obs=1,
    COLORS=MazeColors,
    OBJECT_TO_IDX={
        "background": 0,
        "agent": 1,
        "flag": 2,
        "obstacle": 3,
    },
)


class Observation(TypedDict):
    agent: NDArray
    background: NDArray
    flag: NDArray
    obstacle: NDArray


class MazeSingleAgentEnv(MultiGridEnv):
    """
    Environment with a single agent and multiple flags
    """

    def __init__(
        self,
        map_path: str,
        max_steps: int = 100,
        flag_reward: float = 1.0,
        obstacle_penalty_ratio: float = 0.0,
        step_penalty_ratio: float = 0.01,
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
    ):
        """
        Initialize a new single agent maze environment

        Parameters
        ----------
        map_path : str
            Path to the map file.
        max_steps : int = 100
            Maximum number of steps that the agent can take.
        flag_reward : float = 1.0
            Reward given to the agent for reaching a flag.
        obstacle_penalty_ratio : float = 0.0
            Penalty given to the agent for hitting an obstacle.
        step_penalty_ratio : float = 0.01
            Penalty given to the agent for each step taken.
        render_mode : Literal["human", "rgb_array"] = "rgb_array"
            Render mode.
        """
        agent_view_size: Final[int] = 100

        self.world: Final[World] = MazeWorld
        self.actions_set = MazeActions

        self._map_path: Final[str] = map_path
        self._field_map: Final[NDArray] = np.loadtxt(map_path).T

        height: int
        width: int
        height, width = self._field_map.shape

        self.background: Final[list[Position]] = list(
            zip(*np.where(self._field_map == self.world.OBJECT_TO_IDX["background"]))
        )
        self.obstacle: Final[list[Position]] = list(
            zip(*np.where(self._field_map == self.world.OBJECT_TO_IDX["obstacle"]))
        )
        self.flag: Final[list[Position]] = list(
            zip(*np.where(self._field_map == self.world.OBJECT_TO_IDX["flag"]))
        )

        self._flag_reward: Final[float] = flag_reward
        self._obstacle_penalty_ratio: Final[float] = obstacle_penalty_ratio
        self._step_penalty_ratio: Final[float] = step_penalty_ratio

        blue_agent = Agent(
            self.world,
            index=0,
            color="blue",
            bg_color="white",
            view_size=agent_view_size,
            actions=self.actions_set,
            type="agent",
        )

        agents: list[AgentT] = [blue_agent]

        super().__init__(
            width=width,
            height=height,
            max_steps=max_steps,
            see_through_walls=True,
            agents=agents,
            partial_obs=False,
            agent_view_size=agent_view_size,
            actions_set=self.actions_set,
            world=self.world,
            render_mode=render_mode,
        )

    def _set_observation_space(self) -> spaces.Dict:
        observation_space: spaces.Dict = spaces.Dict(
            {
                "agent": spaces.Box(
                    low=np.array([-1, -1]),
                    high=np.array(self._field_map.shape) - 1,
                    dtype=np.int64,
                ),
                "background": spaces.Box(
                    low=np.array(
                        [[0, 0] for _ in range(len(self.background))]
                    ).flatten(),
                    high=np.array(
                        [self._field_map.shape for _ in range(len(self.background))]
                    ).flatten()
                    - 1,
                    dtype=np.int64,
                ),
                "flag": spaces.Box(
                    low=np.array([[0, 0] for _ in range(len(self.flag))]).flatten(),
                    high=np.array(
                        [self._field_map.shape for _ in range(len(self.flag))]
                    ).flatten()
                    - 1,
                    dtype=np.int64,
                ),
                "obstacle": spaces.Box(
                    low=np.array([[0, 0] for _ in range(len(self.obstacle))]).flatten(),
                    high=np.array(
                        [self._field_map.shape for _ in range(len(self.obstacle))]
                    ).flatten()
                    - 1,
                    dtype=np.int64,
                ),
            }
        )

        return observation_space

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height, self.world)

        for i, j in self.background:
            self.put_obj(Floor(self.world, color="white", type="background"), i, j)

        for i, j in self.obstacle:
            self.put_obj(
                Obstacle(
                    self.world, penalty=self._obstacle_penalty_ratio * self._flag_reward
                ),
                i,
                j,
            )

        for flag_idx, (i, j) in enumerate(self.flag):
            self.put_obj(
                Flag(self.world, index=flag_idx, color="red", bg_color="white"), i, j
            )

        self.init_grid: Grid = self.grid.copy()

        self.place_agent(self.agents[0], pos=random.choice(self.background))

    def reset(self, seed=None) -> tuple[Observation, dict[str, float]]:
        super().reset(seed)

        agent: Agent = self.agents[0]

        assert agent.pos is not None
        self.agent_traj: list[Position] = [agent.pos]

        obs: Observation = self._get_obs()
        info: dict[str, float] = self._get_info()

        return obs, info

    def _get_obs(self) -> Observation:
        for a in self.agents:
            assert a.pos is not None

        observation: Observation = {
            "agent": np.array(self.agents[0].pos),
            "background": np.array(self.background).flatten(),
            "flag": np.array(self.flag).flatten(),
            "obstacle": np.array(self.obstacle).flatten(),
        }

        return observation

    def _get_info(self) -> dict[str, float]:
        assert self.agents[0].pos is not None

        info = {
            "d_a_f": distance_area_point(self.agents[0].pos, self.flag),
            "d_a_ob": distance_area_point(self.agents[0].pos, self.obstacle),
        }
        return info

    def _move_agent(self, action: int, agent: AgentT) -> None:
        next_pos: Position

        assert agent.pos is not None

        match action:
            case self.actions_set.stay:
                next_pos = agent.pos
            case self.actions_set.left:
                next_pos = agent.west_pos()
            case self.actions_set.down:
                next_pos = agent.south_pos()
            case self.actions_set.right:
                next_pos = agent.east_pos()
            case self.actions_set.up:
                next_pos = agent.north_pos()
            case _:
                raise ValueError(f"Invalid action: {action}")

        if (
            next_pos[0] < 0
            or next_pos[1] < 0
            or next_pos[0] >= self.height
            or next_pos[1] >= self.width
        ):
            next_pos = agent.pos
        else:
            pass

        next_cell: WorldObjT | None = self.grid.get(*next_pos)

        bg_color: str = "white"

        if next_cell is None:
            agent.move(next_pos, self.grid, self.init_grid, bg_color=bg_color)
        elif next_cell.can_overlap():
            if next_cell.type == "obstacle":
                pass
            else:
                agent.move(next_pos, self.grid, self.init_grid, bg_color=bg_color)
        else:
            pass

    def _move_agents(self, actions: list[int]) -> None:
        # Move agent
        self._move_agent(actions[0], self.agents[0])

    def _is_agent_on_obj(self, agent_loc: Position | None, obj: list[Position]) -> bool:
        if agent_loc is None:
            assert self.agents[0].pos is not None
            agent_loc = self.agents[0].pos
        else:
            pass

        on_obj: bool = False

        for obj_loc in obj:
            if agent_loc[0] == obj_loc[0] and agent_loc[1] == obj_loc[1]:
                on_obj = True
                break
            else:
                pass

        return on_obj

    def step(
        self, action: int
    ) -> tuple[Observation, float, bool, bool, dict[str, float]]:
        self.step_count += 1

        actions: list[int] = [action]

        self._move_agents(actions)

        assert self.agents[0].pos is not None

        agent_loc: Position = self.agents[0].pos

        terminated: bool = False
        truncated: bool = self.step_count >= self.max_steps

        flag_reward: float = self._flag_reward
        obstacle_penalty: float = flag_reward * self._obstacle_penalty_ratio
        step_penalty: float = flag_reward * self._step_penalty_ratio
        reward: float = 0.0

        if self._is_agent_on_obj(agent_loc, self.flag):
            reward += flag_reward
            terminated = True
        else:
            pass

        if obstacle_penalty != 0:
            if self._is_agent_on_obj(agent_loc, self.obstacle):
                reward -= obstacle_penalty
                terminated = True

            else:
                pass

        else:
            pass

        reward -= step_penalty

        observation: Observation = self._get_obs()
        info: dict[str, float] = self._get_info()

        return observation, reward, terminated, truncated, info

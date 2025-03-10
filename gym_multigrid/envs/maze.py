from dataclasses import dataclass
from typing import Final, Literal, TypedDict, TypeAlias

from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray

from gym_multigrid.core.agent import ActionsT, Agent, AgentT, MazeActions
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.object import Floor, Flag, Obstacle, WorldObjT
from gym_multigrid.core.world import MazeWorld, World
from gym_multigrid.multigrid import (
    DEFAULT_FULL_OBS_ENV_PARTIAL_OBS_CONFIG,
    GridConfig,
    MultiGridEnv,
    PartialObsConfig,
    RenderingConfig,
)
from gym_multigrid.typing import Position
from gym_multigrid.utils.map import distance_area_point, load_text_map


class ObservationDict(TypedDict):
    agent: NDArray
    flag: NDArray
    wall: NDArray


Observation: TypeAlias = ObservationDict | NDArray


class LayoutConfig(TypedDict):
    width: int
    height: int
    flag_positions: list[tuple[int, int]]
    init_agent_positions: list[tuple[int, int] | None]
    wall_positions: list[tuple[int, int]]


@dataclass
class Layout:
    width: int
    height: int
    flag_positions: list[tuple[int, int]]
    init_agent_positions: list[tuple[int, int]]
    wall_positions: list[tuple[int, int]]


class RewardConfig(TypedDict):
    flag_reward: float
    wall_penalty_ratio: float
    step_penalty_ratio: float


@dataclass
class Reward:
    flag_reward: float
    wall_penalty_ratio: float
    step_penalty_ratio: float


class ResetOptions(TypedDict):
    layout_config: LayoutConfig


class ObservationOptionConfig(TypedDict):
    mode: Literal["tensor"]


@dataclass
class ObservationOption:
    mode: Literal["tensor"]


class ObservationFactory:
    @staticmethod
    def observation_space(): ...
    @staticmethod
    def create_observation() -> Observation: ...


class MazeEnv(MultiGridEnv):
    """
    Environment with a single agent and multiple flags
    """

    def __init__(
        self,
        layout_config: LayoutConfig,
        num_agents: int = 1,
        reward_config: RewardConfig = {
            "flag_reward": 1.0,
            "wall_penalty_ratio": 0.0,
            "step_penalty_ratio": 0.01,
        },
        observation_option: ObservationOptionConfig = {"mode": "tensor"},
        action_set: ActionsT = MazeActions,
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
        wall_penalty_ratio : float = 0.0
            Penalty given to the agent for hitting a wall.
        step_penalty_ratio : float = 0.01
            Penalty given to the agent for each step taken.
        observation_option : Literal["positional", "map"] = "map"
            Observation option. If "positional", the observation is the flattened positions of the objects. If "map", the observation is the same with the map.
        render_mode : Literal["human", "rgb_array"] = "rgb_array"
            Render mode.
        """

        world: Final[World] = MazeWorld

        self.layout = Layout(**layout_config)

        self.observation_option = ObservationOption(**observation_option)

        self.reward = Reward(**reward_config)

        agents: list[AgentT] = [
            Agent(
                self.world,
                index=i,
                color="blue",
                view_size=None,
                actions=action_set,
                type="agent",
            )
            for i in range(num_agents)
        ]

        grid_config: GridConfig = {
            "height": self.layout.height,
            "width": self.layout.width,
            "actions_set": action_set,
            "world": world,
        }

        rendering_config: RenderingConfig = {
            "render_mode": render_mode,
            "uncached_object_types": ["agent"],
        }

        partial_obs_config: PartialObsConfig = DEFAULT_FULL_OBS_ENV_PARTIAL_OBS_CONFIG

        super().__init__(
            agents=agents,
            **grid_config,
            **rendering_config,
            **partial_obs_config,
        )

    def _set_observation_space(self) -> spaces.Dict | spaces.Box:
        match self.observation_option.mode:
            case "tensor":
                observation_space = spaces.Box(
                    low=0,
                    high=len(self.world.OBJECT_TO_IDX) - 1,
                    shape=(2, self.height, self.width),
                    dtype=np.int64,
                )

            case _:
                raise ValueError(
                    f"Invalid observation option: {self.observation_option.mode}"
                )

        return observation_space

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height, self.world)

        for i, j in self.layout.wall_positions:
            self.put_obj(
                Obstacle(
                    self.world,
                    penalty=self.reward.wall_penalty_ratio * self.reward.flag_reward,
                ),
                i,
                j,
            )

        for flag_idx, (i, j) in enumerate(self.layout.flag_positions):
            self.put_obj(
                Flag(self.world, index=flag_idx, color="red", bg_color="white"), i, j
            )

        self.init_grid: Grid = self.grid.copy()

        agents_init_pos: list[tuple[int, int] | None] = self.layout.init_agent_positions
        match len(agents_init_pos):
            case 0:
                agents_init_pos = [None] * len(self.agents)
            case 1:
                agents_init_pos = agents_init_pos * len(self.agents)
            case len(self.agents):
                pass
            case _:
                raise ValueError(
                    f"Number of agents {len(self.agents)} and number of initial agent positions {len(agents_init_pos)} do not match: {len(self.agents)} != {len(agents_init_pos)}"
                )

        for agent, init_pos in zip(self.agents, agents_init_pos):
            agent.reset()
            self.place_agent(agent, pos=init_pos)

    def reset(
        self, seed: int | None = None, options: ResetOptions | None = None
    ) -> tuple[Observation, dict[str, float]]:

        if options is not None:
            self.layout = Layout(**options["layout_config"])

        self._reset_gym(seed=seed)
        self._gen_grid(self.width, self.height)

        self.agent_traj: list[list[tuple[int, int]]] = [
            agent.pos for agent in self.agents
        ]
        self.rewards: list[float] = []

        obs: Observation = self._get_obs()
        info: dict[str, float] = self._get_info()

        return obs, info

    def _get_obs(self) -> Observation:
        for a in self.agents:
            assert a.pos is not None

        observation: Observation

        match self.observation_option:
            case "positional":
                observation = {
                    "agent": np.array(self.agents[0].pos),
                    "flag": np.array(self.flag).flatten(),
                    "wall": np.array(self.wall).flatten(),
                }
            case "map":
                observation = self._encode_map()

            case _:
                raise ValueError(
                    f"Invalid observation option: {self.observation_option}"
                )

        return observation

    def _encode_map(self) -> NDArray:
        encoded_map: NDArray = np.zeros((self.width, self.height))

        for i, j in self.layout.wall_positions:
            encoded_map[i, j] = self.world.OBJECT_TO_IDX["wall"]
        for i, j in self.layout.flag_positions:
            encoded_map[i, j] = self.world.OBJECT_TO_IDX["flag"]

        assert self.agents[0].pos is not None
        encoded_map[self.agents[0].pos[0], self.agents[0].pos[1]] = (
            self.world.OBJECT_TO_IDX["agent"]
        )

        return encoded_map

    def _get_info(self) -> dict[str, float]:

        info = {}
        return info

    def _move_agent(self, action: int, agent: AgentT) -> None:
        next_pos: Position

        assert agent.pos is not None

        match action:
            case self.actions_set.stay:
                next_pos = agent.pos
            case self.actions_set.left:
                next_pos = agent.pos + np.array([0, -1])
            case self.actions_set.down:
                next_pos = agent.pos + np.array([-1, 0])
            case self.actions_set.right:
                next_pos = agent.pos + np.array([0, 1])
            case self.actions_set.up:
                next_pos = agent.pos + np.array([1, 0])
            case _:
                raise ValueError(f"Invalid action: {action}")

        if (
            next_pos[0] < 0
            or next_pos[1] < 0
            or next_pos[0] >= self.height
            or next_pos[1] >= self.width
        ):
            pass  # Do nothing
        else:
            next_cell: WorldObjT | None = self.grid.get(*next_pos)

            bg_color: str = "white"

            if next_cell is None:
                agent.move(next_pos, self.grid, self.init_grid, bg_color=bg_color)
            elif next_cell.can_overlap():
                agent.move(next_pos, self.grid, self.init_grid, bg_color=bg_color)
            else:
                pass

    def _move_agents(self, actions: list[int]) -> None:
        # Move agent
        order: list[int] = np.random.permutation(len(self.agents)).tolist()

        for i in order:
            self._move_agent(actions[i], self.agents[i])

    def _is_agent_on_obj(
        self, agent_loc: tuple[int, int] | None, obj: list[tuple[int, int]]
    ) -> bool:
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
        self, action: int | list[int]
    ) -> tuple[Observation, float, bool, bool, dict[str, float]]:
        self.step_count += 1

        actions: list[int] = np.array([action]).flatten().tolist()

        self._move_agents(actions)

        assert self.agents[0].pos is not None

        terminated: bool = False
        truncated: bool = True

        flag_reward: float = self.reward.flag_reward
        wall_penalty: float = flag_reward * self.reward.wall_penalty_ratio
        step_penalty: float = flag_reward * self.reward.step_penalty_ratio
        reward: float = 0.0

        all_agents_on_flag: bool = True
        for agent in self.agents:
            if not self._is_agent_on_obj(agent.pos, self.layout.flag_positions):
                all_agents_on_flag = False
            else:
                reward += flag_reward

        agent_on_wall: bool = False

        if wall_penalty != 0:
            for agent in self.agents:
                if self._is_agent_on_obj(agent.pos, self.layout.wall_positions):
                    agent_on_wall = True
                    reward -= wall_penalty
                else:
                    pass
        else:
            pass

        terminated: bool = all_agents_on_flag or agent_on_wall

        reward -= step_penalty

        self.agent_traj.append([agent.pos for agent in self.agents])
        self.rewards.append(reward)

        observation: Observation = self._get_obs()
        info: dict[str, float] = self._get_info()

        return observation, reward, terminated, truncated, info

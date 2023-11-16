import enum
from itertools import chain
import random
from typing import Final, Literal, TypeAlias, TypedDict, Type

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


CtfColors: dict[str, NDArray] = {
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


class ObservationDict(TypedDict):
    blue_agent: NDArray
    red_agent: NDArray
    blue_flag: NDArray
    red_flag: NDArray
    blue_territory: NDArray
    red_territory: NDArray
    obstacle: NDArray
    is_red_agent_defeated: int


Observation: TypeAlias = ObservationDict | NDArray


class Ctf1v1Env(MultiGridEnv):
    """
    Environment for capture the flag with two agents.
    """

    def __init__(
        self,
        map_path: str,
        enemy_policy: AgentPolicyT = RwPolicy(),
        battle_range: float = 1.0,
        randomness: float = 0.75,
        flag_reward: float = 1.0,
        battle_reward_ratio: float = 0.25,
        obstacle_penalty_ratio: float = 0.0,
        step_penalty_ratio: float = 0.01,
        max_steps: int = 100,
        observation_option: Literal["positional", "map"] = "map",
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
        uncached_object_types: list[str] = ["red_agent", "blue_agent"],
    ):
        """
        Initialize a new capture the flag environment.

        Parameters
        ----------
        map_path : str
            Path to the map file.
        enemy_policy : Type[AgentPolicyT]
            Policy of the enemy agent.
        randomness : float=0.75
            Probability of the enemy agent winning a battle within its territory.
        flag_reward : float=1.0
            Reward for capturing the enemy flag.
        battle_reward_ratio : float=0.25
            Ratio of the flag reward for winning a battle.
        obstacle_penalty_ratio : float=0.0
            Ratio of the flag reward for colliding with an obstacle.
        step_penalty_ratio : float=0.01
            Ratio of the flag reward for taking a step.
        max_steps : int=100
            Maximum number of steps per episode.
        render_mode : Literal["human", "rgb_array"]="rgb_array"
            Rendering mode.
        """

        self.battle_range: Final[float] = battle_range
        self.randomness: Final[float] = randomness
        self.flag_reward: Final[float] = flag_reward
        self.battle_reward: Final[float] = battle_reward_ratio * flag_reward
        self.obstacle_penalty: Final[float] = obstacle_penalty_ratio * flag_reward
        self.step_penalty: Final[float] = step_penalty_ratio * flag_reward

        self.observation_option: Final[
            Literal["positional", "map"]
        ] = observation_option

        partial_obs: bool = False
        agent_view_size: int = 10

        self.world = Ctf1v1World
        self.actions_set = CtfActions
        see_through_walls: bool = False

        self._map_path: Final[str] = map_path
        self._field_map: Final[NDArray] = np.loadtxt(map_path).T

        height: int
        width: int
        height, width = self._field_map.shape

        self.obstacle: Final[list[Position]] = list(
            zip(*np.where(self._field_map == self.world.OBJECT_TO_IDX["obstacle"]))
        )

        self.blue_flag: Final[Position] = list(
            zip(*np.where(self._field_map == self.world.OBJECT_TO_IDX["blue_flag"]))
        )[0]

        self.red_flag: Final[Position] = list(
            zip(*np.where(self._field_map == self.world.OBJECT_TO_IDX["red_flag"]))
        )[0]

        self.blue_territory: Final[list[Position]] = list(
            zip(
                *np.where(self._field_map == self.world.OBJECT_TO_IDX["blue_territory"])
            )
        ) + [self.blue_flag]

        self.red_territory: Final[list[Position]] = list(
            zip(*np.where(self._field_map == self.world.OBJECT_TO_IDX["red_territory"]))
        ) + [self.red_flag]

        blue_agent = Agent(
            self.world,
            index=0,
            color="blue",
            bg_color="light_blue",
            view_size=agent_view_size,
            actions=self.actions_set,
            type="blue_agent",
        )
        red_agent = PolicyAgent(
            enemy_policy,
            self.world,
            index=1,
            color="red",
            bg_color="light_red",
            view_size=agent_view_size,
            actions=self.actions_set,
            type="red_agent",
        )
        red_agent.type = "red_agent"

        agents: list[AgentT] = [blue_agent, red_agent]

        super().__init__(
            width=width,
            height=height,
            max_steps=max_steps,
            see_through_walls=see_through_walls,
            agents=agents,
            partial_obs=partial_obs,
            agent_view_size=agent_view_size,
            actions_set=self.actions_set,
            world=self.world,
            render_mode=render_mode,
            uncached_object_types=uncached_object_types,
        )

    def _set_observation_space(self) -> spaces.Dict | spaces.Box:
        match self.observation_option:
            case "positional":
                observation_space = spaces.Dict(
                    {
                        "blue_agent": spaces.Box(
                            low=np.array([-1, -1]),
                            high=np.array(self._field_map.shape) - 1,
                            dtype=np.int64,
                        ),
                        "red_agent": spaces.Box(
                            low=np.array([-1, -1]),
                            high=np.array(self._field_map.shape) - 1,
                            dtype=np.int64,
                        ),
                        "blue_flag": spaces.Box(
                            low=np.array([0, 0]),
                            high=np.array(self._field_map.shape) - 1,
                            dtype=np.int64,
                        ),
                        "red_flag": spaces.Box(
                            low=np.array([0, 0]),
                            high=np.array(self._field_map.shape) - 1,
                            dtype=np.int64,
                        ),
                        "blue_territory": spaces.Box(
                            low=np.array(list(chain.from_iterable([[0, 0] for _ in range(len(self.blue_territory))]))),  # type: ignore
                            high=np.array(list(chain.from_iterable([self._field_map.shape for _ in range(len(self.blue_territory))]))).flatten() - 1,  # type: ignore
                            dtype=np.int64,
                        ),
                        "red_territory": spaces.Box(
                            low=np.array(list(chain.from_iterable([[0, 0] for _ in range(len(self.red_territory))]))),  # type: ignore
                            high=np.array(list(chain.from_iterable([self._field_map.shape for _ in range(len(self.red_territory))]))).flatten() - 1,  # type: ignore
                            dtype=np.int64,
                        ),
                        "obstacle": spaces.Box(
                            low=np.array(list(chain.from_iterable([[0, 0] for _ in range(len(self.obstacle))]))),  # type: ignore
                            high=np.array(list(chain.from_iterable([self._field_map.shape for _ in range(len(self.obstacle))]))).flatten() - 1,  # type: ignore
                            dtype=np.int64,
                        ),
                        "is_red_agent_defeated": spaces.Discrete(2),
                    }
                )

            case "map":
                observation_space = spaces.Box(
                    low=0,
                    high=len(self.world.OBJECT_TO_IDX) - 1,
                    shape=self._field_map.shape,
                    dtype=np.int64,
                )

        return observation_space

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height, self.world)

        for i, j in self.blue_territory:
            self.put_obj(
                Floor(self.world, color="light_blue", type="blue_territory"), i, j
            )

        for i, j in self.red_territory:
            self.put_obj(
                Floor(self.world, color="light_red", type="red_territory"), i, j
            )

        for i, j in self.obstacle:
            self.put_obj(Obstacle(self.world, penalty=self.obstacle_penalty), i, j)

        self.put_obj(
            Flag(
                self.world,
                index=0,
                color="blue",
                type="blue_flag",
                bg_color="light_blue",
            ),
            *self.blue_flag,
        )
        self.put_obj(
            Flag(
                self.world, index=1, color="red", type="red_flag", bg_color="light_red"
            ),
            *self.red_flag,
        )

        self.init_grid: Grid = self.grid.copy()

        self.place_agent(
            self.agents[0],
            pos=self.blue_territory[np.random.randint(0, len(self.blue_territory))],
        )
        self.place_agent(
            self.agents[1],
            pos=self.red_territory[np.random.randint(0, len(self.red_territory))],
        )

    def reset(self, seed=None) -> tuple[Observation, dict[str, float]]:
        super().reset(seed)
        self._is_red_agent_defeated: bool = False

        assert self.agents[0].pos is not None
        assert self.agents[1].pos is not None
        self.blue_traj: list[Position] = [self.agents[0].pos]
        self.red_traj: list[Position] = [self.agents[1].pos]

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
                    "blue_agent": np.array(self.agents[0].pos),
                    "red_agent": np.array(self.agents[1].pos),
                    "blue_flag": np.array(self.blue_flag),
                    "red_flag": np.array(self.red_flag),
                    "blue_territory": np.array(self.blue_territory).flatten(),
                    "red_territory": np.array(self.red_territory).flatten(),
                    "obstacle": np.array(self.obstacle).flatten(),
                    "is_red_agent_defeated": int(self._is_red_agent_defeated),
                }
            case "map":
                observation = self._encode_map()

        return observation

    def _encode_map(self) -> NDArray:
        encoded_map: NDArray = np.zeros(self._field_map.shape, dtype=np.int64)

        for i, j in self.blue_territory:
            encoded_map[i, j] = self.world.OBJECT_TO_IDX["blue_territory"]

        for i, j in self.red_territory:
            encoded_map[i, j] = self.world.OBJECT_TO_IDX["red_territory"]

        for i, j in self.obstacle:
            encoded_map[i, j] = self.world.OBJECT_TO_IDX["obstacle"]

        encoded_map[self.blue_flag[0], self.blue_flag[1]] = self.world.OBJECT_TO_IDX[
            "blue_flag"
        ]

        encoded_map[self.red_flag[0], self.red_flag[1]] = self.world.OBJECT_TO_IDX[
            "red_flag"
        ]

        assert self.agents[0].pos is not None
        assert self.agents[1].pos is not None

        encoded_map[
            self.agents[0].pos[0], self.agents[0].pos[1]
        ] = self.world.OBJECT_TO_IDX["blue_agent"]

        encoded_map[self.agents[1].pos[0], self.agents[1].pos[1]] = (
            self.world.OBJECT_TO_IDX["red_agent"]
            if not self._is_red_agent_defeated
            else self.world.OBJECT_TO_IDX["obstacle"]
        )

        return encoded_map.T

    def _get_info(self) -> dict[str, float]:
        assert self.agents[0].pos is not None
        assert self.agents[1].pos is not None

        info = {
            "d_ba_ra": distance_points(self.agents[0].pos, self.agents[1].pos),
            "d_ba_bf": distance_points(self.agents[0].pos, self.blue_flag),
            "d_ba_rf": distance_points(self.agents[0].pos, self.red_flag),
            "d_ra_bf": distance_points(self.agents[1].pos, self.blue_flag),
            "d_ra_rf": distance_points(self.agents[1].pos, self.red_flag),
            "d_bf_rf": distance_points(self.blue_flag, self.red_flag),
            "d_ba_bb": distance_area_point(self.agents[0].pos, self.blue_territory),
            "d_ba_rb": distance_area_point(self.agents[0].pos, self.red_territory),
            "d_ra_bb": distance_area_point(self.agents[1].pos, self.blue_territory),
            "d_ra_rb": distance_area_point(self.agents[1].pos, self.red_territory),
            "d_ba_ob": distance_area_point(self.agents[0].pos, self.obstacle),
        }
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
            or next_pos[0] >= self.width
            or next_pos[1] >= self.height
        ):
            pass
        else:
            next_cell: WorldObjT | None = self.grid.get(*next_pos)

            is_agent_in_blue_territory: bool = self._is_agent_in_territory(
                agent.type, "blue", next_pos
            )
            is_agent_in_red_territory: bool = self._is_agent_in_territory(
                agent.type, "red", next_pos
            )

            if is_agent_in_blue_territory:
                bg_color = "light_blue"
            elif is_agent_in_red_territory:
                bg_color = "light_red"
            else:
                bg_color = None

            if next_cell is None:
                agent.move(next_pos, self.grid, self.init_grid, bg_color=bg_color)
            elif next_cell.can_overlap():
                agent.move(next_pos, self.grid, self.init_grid, bg_color=bg_color)
            else:
                pass

    def _move_agents(self, actions: list[int]) -> None:
        # Move blue agent
        self._move_agent(actions[0], self.agents[0])
        # Move red agent
        if not self._is_red_agent_defeated:
            self._move_agent(actions[1], self.agents[1])
        else:
            pass

    def _is_agent_in_territory(
        self,
        agent_type: Literal["blue_agent", "red_agent"],
        territory_name: Literal["blue", "red"],
        agent_loc: Position | None = None,
    ) -> bool:
        in_territory: bool = False

        territory: list[Position]
        if agent_loc is None:
            match agent_type:
                case "blue_agent":
                    assert self.agents[0].pos is not None
                    agent_loc = self.agents[0].pos
                case "red_agent":
                    assert self.agents[1].pos is not None
                    agent_loc = self.agents[1].pos
                case _:
                    raise ValueError(f"Invalid agent_name: {agent_type}")
        else:
            pass

        match territory_name:
            case "blue":
                territory = self.blue_territory
            case "red":
                territory = self.red_territory
            case _:
                raise ValueError(f"Invalid territory_name: {territory_name}")

        for i, j in territory:
            if agent_loc[0] == i and agent_loc[1] == j:
                in_territory = True
                break
            else:
                pass

        return in_territory

    def step(
        self, action: int
    ) -> tuple[Observation, float, bool, bool, dict[str, float]]:
        self.step_count += 1

        assert type(self.agents[1]) is PolicyAgent
        red_action: int = self.agents[1].policy.act(self._get_obs(), self.actions_set)

        actions: list[int] = [action, red_action]

        self._move_agents(actions)

        assert self.agents[0].pos is not None
        assert self.agents[1].pos is not None

        blue_agent_loc: Position = self.agents[0].pos
        red_agent_loc: Position = self.agents[1].pos

        terminated: bool = False
        truncated: bool = self.step_count >= self.max_steps

        reward: float = 0.0

        if (
            blue_agent_loc[0] == self.red_flag[0]
            and blue_agent_loc[1] == self.red_flag[1]
        ):
            reward += self.flag_reward
            terminated = True
        else:
            pass

        if (
            red_agent_loc[0] == self.blue_flag[0]
            and red_agent_loc[1] == self.blue_flag[1]
        ):
            reward -= self.flag_reward
            terminated = True
        else:
            pass

        if (
            distance_points(blue_agent_loc, red_agent_loc) <= self.battle_range
            and not self._is_red_agent_defeated
        ):
            blue_win: bool

            blue_agent_in_blue_territory: bool = self._is_agent_in_territory(
                "blue_agent", "blue"
            )
            red_agent_in_red_territory: bool = self._is_agent_in_territory(
                "red_agent", "red"
            )

            match (blue_agent_in_blue_territory, red_agent_in_red_territory):
                case (True, True):
                    blue_win = np.random.choice([True, False])
                case (True, False):
                    blue_win = np.random.choice(
                        [True, False], p=[self.randomness, 1 - self.randomness]
                    )
                case (False, True):
                    blue_win = np.random.choice(
                        [True, False], p=[1 - self.randomness, self.randomness]
                    )

                case (False, False):
                    blue_win = np.random.choice([True, False])

                case (_, _):
                    raise ValueError(
                        f"Invalid combination of blue_agent_in_blue_territory: {blue_agent_in_blue_territory} and red_agent_in_red_territory: {red_agent_in_red_territory}"
                    )

            if blue_win:
                reward += self.battle_reward
                self._is_red_agent_defeated = True
            else:
                reward -= self.battle_reward
                terminated = True

        if self.obstacle_penalty != 0:
            if blue_agent_loc in self.obstacle:
                reward -= self.obstacle_penalty
                terminated = True

            else:
                pass

        else:
            pass

        reward -= self.step_penalty

        observation: Observation = self._get_obs()
        info: dict[str, float] = self._get_info()

        return observation, reward, terminated, truncated, info

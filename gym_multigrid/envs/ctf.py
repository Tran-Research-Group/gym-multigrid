import enum
from itertools import chain
import random
from typing import Final, Literal, TypedDict, Type

from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray

from gym_multigrid.core.agent import Agent, PolicyAgent, AgentT
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.object import Floor, Flag, Obstacle
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


class Observation(TypedDict):
    blue_agent: NDArray
    red_agent: NDArray
    blue_flag: NDArray
    red_flag: NDArray
    blue_territory: NDArray
    red_territory: NDArray
    obstacle: NDArray
    is_red_agent_defeated: NDArray


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
        obstacle_penalty_ratio: float = 0.5,
        step_penalty_ratio: float = 0.01,
        max_steps: int = 100,
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
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

        partial_obs: bool = False
        agent_view_size: int = 10

        self.world = Ctf1v1World
        self.actions_set = CtfActions
        see_through_walls: bool = False

        self._map_path: Final[str] = map_path
        self._field_map: Final[NDArray] = np.rot90(np.loadtxt(map_path))

        height: int
        width: int
        height, width = self._field_map.shape

        self.obstacle: Final[list[Position]] = list(zip(*np.where(self._field_map == self.world.OBJECT_TO_IDX["obstacle"])))  # type: ignore

        self.blue_flag: Final[Position] = list(zip(*np.where(self._field_map == self.world.OBJECT_TO_IDX["blue_flag"])))[0]  # type: ignore

        self.red_flag: Final[Position] = list(zip(*np.where(self._field_map == self.world.OBJECT_TO_IDX["red_flag"])))[0]  # type: ignore

        self.blue_territory: Final[list[Position]] = list(zip(*np.where(self._field_map == self.world.OBJECT_TO_IDX["blue_territory"]))) + [self.blue_flag]  # type: ignore

        self.red_territory: Final[list[Position]] = list(zip(*np.where(self._field_map == self.world.OBJECT_TO_IDX["red_territory"]))) + [self.red_flag]  # type: ignore

        blue_agent = Agent(
            self.world,
            index=0,
            color="blue",
            view_size=agent_view_size,
            actions=self.actions_set,
            type="blue_agent",
        )
        red_agent = PolicyAgent(
            enemy_policy,
            self.world,
            index=1,
            color="red",
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

        bf = Flag(self.world, index=0, color="blue", type="blue_flag")
        bf.type = "blue_flag"
        self.put_obj(bf, *self.blue_flag)
        rf = Flag(self.world, index=1, color="red", type="red_flag")
        rf.type = "red_flag"
        self.put_obj(rf, *self.red_flag)

        self.place_agent(self.agents[0], pos=random.choice(self.blue_territory))
        self.place_agent(self.agents[1], pos=random.choice(self.red_territory))

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

        observation: Observation = {
            "blue_agent": np.array(self.agents[0].pos),
            "red_agent": np.array(self.agents[1].pos),
            "blue_flag": np.array(self.blue_flag),
            "red_flag": np.array(self.red_flag),
            "blue_territory": np.array(self.blue_territory),
            "red_territory": np.array(self.red_territory),
            "obstacle": np.array(self.obstacle),
            "is_red_agent_defeated": np.array(int(self._is_red_agent_defeated)),
        }

        return observation

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

        match action:
            case self.actions_set.stay:
                assert agent.pos is not None
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

        next_cell = self.grid.get(*next_pos)

        if next_cell is None:
            agent.move(next_pos, self.grid)
        elif next_cell.can_overlap():
            if agent.type == "red_agent" and next_cell.type == "obstacle":
                pass
            else:
                agent.move(next_pos, self.grid)
        else:
            pass

    def _move_agents(self, actions: list[int]) -> None:
        # Move blue agent
        self._move_agent(actions[0], self.agents[0])
        # Move red agent
        self._move_agent(actions[1], self.agents[1])

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

        if (blue_agent_loc == self.red_flag).all():
            reward += self.flag_reward
            terminated = True
        else:
            pass

        if (red_agent_loc == self.blue_flag).all():
            reward -= self.flag_reward
            terminated = True
        else:
            pass

        if (
            distance_points(blue_agent_loc, red_agent_loc) <= self.battle_range
            and not self._is_red_agent_defeated
        ):
            blue_win: bool

            blue_agent_in_blue_territory: bool = blue_agent_loc in self.blue_territory
            red_agent_in_red_territory: bool = red_agent_loc in self.red_territory

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

from typing import Final, Literal, TypeAlias, TypedDict

from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray

from gym_multigrid.core.agent import Agent, PolicyAgent, AgentT, CtfActions
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.object import Floor, Flag, Obstacle, WorldObjT
from gym_multigrid.core.world import CtfWorld, World
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


Observation: TypeAlias = (
    ObservationDict
    | MultiAgentObservationDict
    | NDArray[np.int_]
    | dict[str, NDArray[np.int_] | int]
)


class GameStats(TypedDict):
    blue_agent_defeated: list[bool]
    red_agent_defeated: list[bool]
    blue_flag_captured: bool
    red_flag_captured: bool


ObservationOption: TypeAlias = Literal[
    "positional", "map", "flattened", "pos_map", "pos_map_flattened", "tensor"
]


class CtfMvNEnv(MultiGridEnv):
    """
    Environment for capture the flag with multiple agents with N blue agents and M red agents.
    """

    def __init__(
        self,
        map_path: str,
        num_blue_agents: int = 2,
        num_red_agents: int = 2,
        enemy_policies: list[CtfPolicyT] | CtfPolicyT = RwPolicy(),
        battle_range: float = 1,
        randomness: float = 0.75,
        flag_reward: float = 1,
        battle_reward_ratio: float = 0.25,
        obstacle_penalty_ratio: float = 0,
        step_penalty_ratio: float = 0.01,
        max_steps: int = 100,
        observation_option: ObservationOption = "positional",
        observation_scaling: float = 1,
        render_mode: Literal["human"] | Literal["rgb_array"] = "rgb_array",
        uncached_object_types: list[str] = ["red_agent", "blue_agent"],
    ) -> None:
        """
        Initialize a new capture the flag environment.

        Parameters
        ----------
        map_path : str
            Path to the map file.
        num_blue_agents : int = 2
            Number of blue (friendly) agents.
        num_red_agents : int = 2
            Number of red (enemy) agents.
        enemy_policies : list[Type[CtfPolicyT]]=[RwwPolicy()]
            Policies of the enemy agents. If there is only one policy, it will be used for all enemy agents.
        battle_range : float = 1
            Range within which battles can occur.
        randomness : float = 0.75
            Probability of the enemy agent winning a battle within its territory.
        flag_reward : float = 1
            Reward for capturing the enemy flag.
        battle_reward_ratio : float = 0.25
            Ratio of the flag reward for winning a battle.
        obstacle_penalty_ratio : float=0
            Ratio of the flag reward for colliding with an obstacle.
        step_penalty_ratio : float = 0.01
            Ratio of the flag reward for taking a step.
        max_steps : int=100
            Maximum number of steps per episode.
        observation_option : Literal["positional", "map", "flattened"] = "positional"
            Observation option.
        observation_scaling : float = 1
            Scaling factor for the observation.
        render_mode : Literal["human", "rgb_array"] = "rgb_array"
            Rendering mode.
        uncached_object_types : list[str] = ["red_agent", "blue_agent"]
            Types of objects that should not be cached.
        """

        self.num_blue_agents: Final[int] = num_blue_agents
        self.num_red_agents: Final[int] = num_red_agents

        self.battle_range: Final[float] = battle_range
        self.randomness: Final[float] = randomness
        self.flag_reward: Final[float] = flag_reward
        self.battle_reward: Final[float] = battle_reward_ratio * flag_reward
        self.obstacle_penalty: Final[float] = obstacle_penalty_ratio * flag_reward
        self.step_penalty: Final[float] = step_penalty_ratio * flag_reward

        self.observation_option: Final[ObservationOption] = observation_option
        self.observation_scaling: Final[float] = observation_scaling

        partial_obs: bool = False
        agent_view_size: int = 10

        self.world = CtfWorld
        self.actions_set = CtfActions
        see_through_walls: bool = False

        self._map_path: Final[str] = map_path
        self._field_map: Final[NDArray] = load_text_map(map_path)
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

        blue_agents: list[AgentT] = [
            Agent(
                self.world,
                index=i,
                color="blue",
                bg_color="light_blue",
                view_size=agent_view_size,
                actions=self.actions_set,
                type="blue_agent",
            )
            for i in range(num_blue_agents)
        ]

        # Check if there is only one policy for all enemy agents.
        if type(enemy_policies) is not list:
            enemy_policies = [enemy_policies for _ in range(num_red_agents)]
        else:
            # Check if the number of policies is equal to the number of enemy agents.
            assert len(enemy_policies) == num_red_agents

        for policy in enemy_policies:
            if hasattr(policy, "random_generator"):
                if policy.random_generator is None:
                    policy.random_generator = self.np_random
                else:
                    pass
            else:
                pass

            if hasattr(policy, "field_map"):
                if policy.field_map is None:
                    policy.field_map = self._field_map
                else:
                    pass
            else:
                pass

        red_agents: list[AgentT] = [
            PolicyAgent(
                enemy_policies[i],
                self.world,
                index=self.num_blue_agents + i,
                color="red",
                bg_color="light_red",
                view_size=agent_view_size,
                actions=self.actions_set,
                type="red_agent",
            )
            for i in range(num_red_agents)
        ]

        agents: list[AgentT] = blue_agents + red_agents

        # Set the random generator and action set of the red agent from the env.
        for agent in agents:
            if type(agent) is PolicyAgent:
                agent.policy.random_generator = self.np_random
                agent.policy.action_set = self.actions_set
            else:
                pass

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

        self.action_space = spaces.MultiDiscrete(
            [len(self.actions_set) for _ in range(self.num_blue_agents)]
        )
        self.ac_dim = self.action_space.shape

    def _set_observation_space(self) -> spaces.Dict | spaces.Box:
        match self.observation_option:
            case "positional":
                observation_space = spaces.Dict(
                    {
                        "blue_agent": spaces.Box(
                            low=np.array(
                                [[-1, -1] for _ in range(self.num_blue_agents)]
                            ).flatten(),
                            high=np.array(
                                [
                                    self._field_map.shape
                                    for _ in range(self.num_blue_agents)
                                ]
                            ).flatten()
                            - 1,
                            dtype=np.int64,
                        ),
                        "red_agent": spaces.Box(
                            low=np.array(
                                [[-1, -1] for _ in range(self.num_red_agents)]
                            ).flatten(),
                            high=np.array(
                                [
                                    self._field_map.shape
                                    for _ in range(self.num_red_agents)
                                ]
                            ).flatten()
                            - 1,
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
                            low=np.array(
                                [[0, 0] for _ in range(len(self.blue_territory))]
                            ).flatten(),
                            high=np.array(
                                np.array(
                                    [
                                        self._field_map.shape
                                        for _ in range(len(self.blue_territory))
                                    ]
                                )
                            ).flatten()
                            - 1,
                            dtype=np.int64,
                        ),
                        "red_territory": spaces.Box(
                            low=np.array(
                                [[0, 0] for _ in range(len(self.red_territory))]
                            ).flatten(),
                            high=np.array(
                                np.array(
                                    [
                                        self._field_map.shape
                                        for _ in range(len(self.red_territory))
                                    ]
                                )
                            ).flatten()
                            - 1,
                            dtype=np.int64,
                        ),
                        "obstacle": spaces.Box(
                            low=np.array(
                                [[0, 0] for _ in range(len(self.obstacle))]
                            ).flatten(),
                            high=np.array(
                                np.array(
                                    [
                                        self._field_map.shape
                                        for _ in range(len(self.obstacle))
                                    ]
                                )
                            ).flatten()
                            - 1,
                            dtype=np.int64,
                        ),
                        "terminated_agents": spaces.Box(
                            low=np.zeros(
                                [self.num_blue_agents + self.num_red_agents],
                                dtype=np.int64,
                            ),
                            high=np.ones(
                                [self.num_blue_agents + self.num_red_agents],
                                dtype=np.int64,
                            ),
                        ),
                    }
                )

            case "map":
                observation_space = spaces.Box(
                    low=0,
                    high=len(self.world.OBJECT_TO_IDX) - 1,
                    shape=self._field_map.shape,
                    dtype=np.int64,
                )

            case "flattened":
                obs_size: int = (
                    2 * (self.num_blue_agents + self.num_red_agents)
                    + 4  # 2 * (blue_flag + red_flag)
                    + 2 * len(self.obstacle)
                    + 2 * len(self.blue_territory)
                    + 2 * len(self.red_territory)
                    + self.num_blue_agents
                    + self.num_red_agents
                )
                obs_high = (
                    np.ones([obs_size])
                    * (np.max(self._field_map.shape) - 1)
                    / self.observation_scaling
                )
                obs_high[-(self.num_blue_agents + self.num_red_agents) :] = 1
                observation_space = spaces.Box(
                    low=np.zeros([obs_size]),
                    high=obs_high,
                    dtype=np.int64,
                )

            case "pos_map":
                observation_space = spaces.Dict(
                    {
                        "blue_agent": spaces.Box(
                            low=np.array(
                                [[-1, -1] for _ in range(self.num_blue_agents)]
                            ).flatten(),
                            high=np.array(
                                [
                                    self._field_map.shape
                                    for _ in range(self.num_blue_agents)
                                ]
                            ).flatten()
                            - 1,
                            dtype=np.int64,
                        ),
                        "red_agent": spaces.Box(
                            low=np.array(
                                [[-1, -1] for _ in range(self.num_red_agents)]
                            ).flatten(),
                            high=np.array(
                                [
                                    self._field_map.shape
                                    for _ in range(self.num_red_agents)
                                ]
                            ).flatten()
                            - 1,
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
                        "static_map": spaces.Box(
                            low=0,
                            high=len(self.world.OBJECT_TO_IDX) - 1,
                            shape=self._field_map.shape,
                            dtype=np.int64,
                        ),
                        "terminated_agents": spaces.Box(
                            low=np.zeros(
                                [self.num_blue_agents + self.num_red_agents],
                                dtype=np.int64,
                            ),
                            high=np.ones(
                                [self.num_blue_agents + self.num_red_agents],
                                dtype=np.int64,
                            ),
                        ),
                    }
                )

            case "pos_map_flattened":
                obs_size: int = (
                    2 * (self.num_blue_agents + self.num_red_agents)
                    + 4  # 2 * (blue_flag + red_flag)
                    + np.prod(self._field_map.shape)
                    + self.num_blue_agents
                    + self.num_red_agents
                )
                obs_high = (
                    np.ones([obs_size])
                    * (np.max(self._field_map.shape) - 1)
                    / self.observation_scaling
                )
                obs_high[-(self.num_blue_agents + self.num_red_agents) :] = 1
                observation_space = spaces.Box(
                    low=np.zeros([obs_size]),
                    high=obs_high,
                    dtype=np.int64,
                )

            case "tensor":
                observation_space = spaces.Box(
                    low=0,
                    high=2,
                    shape=(self._field_map.shape[0], self._field_map.shape[1], 3),
                    dtype=np.int64,
                )

            case _:
                raise ValueError(
                    f"Invalid observation_option: {self.observation_option}"
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

        # Choose non-overlapping indices for the blue agents and place them in the blue territory.
        blue_indices: list[int] = self.np_random.choice(
            len(self.blue_territory), self.num_blue_agents, replace=False
        )
        for i in range(self.num_blue_agents):
            self.place_agent(
                self.agents[i],
                pos=self.blue_territory[blue_indices[i]],
                reset_agent_status=True,
            )

        # Choose non-overlapping indices for the red agents and place them in the red territory.
        red_indices: list[int] = self.np_random.choice(
            len(self.red_territory), self.num_red_agents, replace=False
        )
        for i in range(self.num_red_agents):
            self.place_agent(
                self.agents[self.num_blue_agents + i],
                pos=self.red_territory[red_indices[i]],
                reset_agent_status=True,
            )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[Observation, dict[str, float]]:
        super().reset(seed=seed, options=options)

        self.blue_traj: list[list[Position]] = [
            [agent.pos] for agent in self.agents[0 : self.num_blue_agents]
        ]
        self.red_traj: list[list[Position]] = [
            [agent.pos] for agent in self.agents[self.num_blue_agents :]
        ]

        obs: Observation = self._get_obs()
        info: dict[str, float] = self._get_info()

        self.game_stats: GameStats = {
            "blue_agent_defeated": [False for _ in range(self.num_blue_agents)],
            "red_agent_defeated": [False for _ in range(self.num_red_agents)],
            "blue_flag_captured": False,
            "red_flag_captured": False,
        }

        return obs, info

    def _get_obs(self) -> Observation:
        observation: Observation

        match self.observation_option:
            case "positional":
                observation = self._get_dict_obs()
            case "map":
                observation = self._encode_map()
            case "flattened":
                observation = np.array(
                    [
                        *np.array(
                            [
                                agent.pos
                                for agent in self.agents[0 : self.num_blue_agents]
                            ]
                        ).flatten(),
                        *np.array(
                            [agent.pos for agent in self.agents[self.num_blue_agents :]]
                        ).flatten(),
                        *np.array(self.blue_flag),
                        *np.array(self.red_flag),
                        *np.array(self.blue_territory).flatten(),
                        *np.array(self.red_territory).flatten(),
                        *np.array(self.obstacle).flatten(),
                        *[int(agent.terminated) for agent in self.agents],
                    ]
                )
            case "pos_map":
                encoded_map: NDArray = np.zeros(self._field_map.shape, dtype=np.int64)

                for i, j in self.blue_territory:
                    encoded_map[i, j] = self.world.OBJECT_TO_IDX["blue_territory"]

                for i, j in self.red_territory:
                    encoded_map[i, j] = self.world.OBJECT_TO_IDX["red_territory"]

                for i, j in self.obstacle:
                    encoded_map[i, j] = self.world.OBJECT_TO_IDX["obstacle"]

                observation: dict[str, NDArray | int] = {
                    "blue_agent": np.array(
                        [agent.pos for agent in self.agents[0 : self.num_blue_agents]]
                    ).flatten(),
                    "red_agent": np.array(
                        [agent.pos for agent in self.agents[self.num_blue_agents :]]
                    ).flatten(),
                    "blue_flag": np.array(self.blue_flag),
                    "red_flag": np.array(self.red_flag),
                    "static_map": encoded_map.T,
                    "terminated_agents": np.array(
                        [int(agent.terminated) for agent in self.agents]
                    ),
                }
            case "pos_map_flattened":
                encoded_map: NDArray = np.zeros(self._field_map.shape, dtype=np.int64)

                for i, j in self.blue_territory:
                    encoded_map[i, j] = self.world.OBJECT_TO_IDX["blue_territory"]

                for i, j in self.red_territory:
                    encoded_map[i, j] = self.world.OBJECT_TO_IDX["red_territory"]

                for i, j in self.obstacle:
                    encoded_map[i, j] = self.world.OBJECT_TO_IDX["obstacle"]

                observation = np.array(
                    [
                        *np.array(
                            [
                                agent.pos
                                for agent in self.agents[0 : self.num_blue_agents]
                            ]
                        ).flatten(),
                        *np.array(
                            [agent.pos for agent in self.agents[self.num_blue_agents :]]
                        ).flatten(),
                        *np.array(self.blue_flag),
                        *np.array(self.red_flag),
                        *encoded_map.T.flatten(),
                        *[int(agent.terminated) for agent in self.agents],
                    ]
                )
            case "tensor":
                static_object_layer: NDArray[np.int_] = np.zeros(
                    self._field_map.shape, dtype=np.int_
                )
                agent_flag_layer: NDArray[np.int_] = np.zeros(
                    self._field_map.shape, dtype=np.int_
                )
                agent_status_layer: NDArray[np.int_] = np.zeros(
                    self._field_map.shape, dtype=np.int_
                )

                for i, j in self.blue_territory:
                    static_object_layer[i, j] = 1

                for i, j in self.red_territory:
                    static_object_layer[i, j] = 2

                for i, j in self.obstacle:
                    static_object_layer[i, j] = 0

                for agent in self.agents:
                    assert agent.pos is not None
                    agent_flag_layer[agent.pos[0], agent.pos[1]] = (
                        1 if agent.color == "blue" else 2
                    )
                    agent_status_layer[agent.pos[0], agent.pos[1]] = (
                        1 if agent.terminated else 2
                    )

                observation = np.stack(
                    [static_object_layer, agent_flag_layer, agent_status_layer], axis=2
                )

            case _:
                raise ValueError(
                    f"Invalid observation_option: {self.observation_option}"
                )

        return observation

    def _get_dict_obs(self) -> MultiAgentObservationDict:
        for a in self.agents:
            assert a.pos is not None

        observation: MultiAgentObservationDict

        observation = {
            "blue_agent": np.array(
                [agent.pos for agent in self.agents[0 : self.num_blue_agents]]
            ).flatten(),
            "red_agent": np.array(
                [agent.pos for agent in self.agents[self.num_blue_agents :]]
            ).flatten(),
            "blue_flag": np.array(self.blue_flag),
            "red_flag": np.array(self.red_flag),
            "blue_territory": np.array(self.blue_territory).flatten(),
            "red_territory": np.array(self.red_territory).flatten(),
            "obstacle": np.array(self.obstacle).flatten(),
            "terminated_agents": np.array(
                [int(agent.terminated) for agent in self.agents]
            ),
        }

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

        for agent in self.agents:
            assert agent.pos is not None
            encoded_map[agent.pos[0], agent.pos[1]] = self.world.OBJECT_TO_IDX[
                agent.type if not agent.terminated else "obstacle"
            ]

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
            elif self.obstacle_penalty != 0 and (
                next_cell.type == "obstacle"
                or next_cell.type == "red_agent"
                or next_cell.type == "blue_agent"
            ):
                agent.collided = True
            else:
                pass

    def _move_agents(self, actions: list[int]) -> None:
        # Randomly generate the order of the agents by indices using self.np_random.
        agent_indices: list[int] = list(
            range(self.num_blue_agents + self.num_red_agents)
        )
        self.np_random.shuffle(agent_indices)
        for i in agent_indices:
            if self.agents[i].terminated:
                # Defeated agent doesn't move, sadly.
                pass
            else:
                self._move_agent(actions[i], self.agents[i])

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
        self, blue_actions: list[int]
    ) -> tuple[Observation, float, bool, bool, dict[str, float]]:
        self.step_count += 1

        red_actions: list[int] = []
        for red_agent in self.agents[self.num_blue_agents :]:
            assert type(red_agent) is PolicyAgent
            red_action: int = red_agent.policy.act(self._get_dict_obs(), red_agent.pos)
            red_actions.append(red_action)

        # Just in case NN outputs are, for some reason, not discrete.
        rounded_blue_actions: NDArray[np.int_] = np.round(blue_actions).astype(np.int_)
        # Concatenate the blue and red actions as 1D array
        actions: Final[list[int]] = rounded_blue_actions.tolist() + red_actions

        self._move_agents(actions)

        terminated: bool = False
        truncated: Final[bool] = self.step_count >= self.max_steps

        reward: float = 0.0

        # Calculate the collision penalty for the blue agents.
        if self.obstacle_penalty != 0:
            for blue_agent in self.agents[0 : self.num_blue_agents]:
                if blue_agent.collided:
                    reward -= self.obstacle_penalty
                    blue_agent.terminated = True
                    blue_agent.color = "blue_grey"
                else:
                    pass

            for red_agent in self.agents[self.num_blue_agents :]:
                if red_agent.collided:
                    red_agent.terminated = True
                    red_agent.color = "red_grey"
                else:
                    pass
        else:
            pass

        # Calculate the flag rewards of the blue agents
        for blue_agent in self.agents[0 : self.num_blue_agents]:
            if (
                blue_agent.pos[0] == self.red_flag[0]
                and blue_agent.pos[1] == self.red_flag[1]
            ):
                reward += self.flag_reward
                terminated = True
                self.game_stats["red_flag_captured"] = True
            else:
                pass

        # Calculate the flag rewards (penalties) of the red agents
        for red_agent in self.agents[self.num_blue_agents :]:
            if (
                red_agent.pos[0] == self.blue_flag[0]
                and red_agent.pos[1] == self.blue_flag[1]
            ):
                reward -= self.flag_reward
                terminated = True
                self.game_stats["blue_flag_captured"] = True
            else:
                pass

        # Calculate the distances between the blue and red agents and the battle outcomes if they are within the battle range.
        blue_agent_locs: list[Position] = [
            agent.pos for agent in self.agents[0 : self.num_blue_agents]
        ]
        red_agent_locs: list[Position] = [
            agent.pos for agent in self.agents[self.num_blue_agents :]
        ]
        blue_agent_locs_np: NDArray[np.float_] = np.array(blue_agent_locs)
        red_agent_locs_np: NDArray[np.float_] = np.array(red_agent_locs)

        distances: NDArray[np.float_] = np.linalg.norm(
            blue_agent_locs_np[:, np.newaxis] - red_agent_locs_np, axis=2
        )

        # Get the indices of distances that are less than the battle range.
        battle_indices: tuple[NDArray[np.int_], NDArray[np.int_]] = np.where(
            distances <= self.battle_range
        )
        # Iterate over the indices and perform the battles.
        for blue_agent_idx, red_agent_idx in zip(*battle_indices):
            # Battle only takes place if both agents are not defeated (terminated).
            if (
                not self.agents[blue_agent_idx].terminated
                and not self.agents[self.num_blue_agents + red_agent_idx].terminated
            ):
                blue_agent_in_blue_territory: bool = self._is_agent_in_territory(
                    "blue_agent", "blue", blue_agent_locs[blue_agent_idx]
                )
                red_agent_in_red_territory: bool = self._is_agent_in_territory(
                    "red_agent", "red", red_agent_locs[red_agent_idx]
                )

                blue_win: bool
                match (blue_agent_in_blue_territory, red_agent_in_red_territory):
                    case (True, True):
                        blue_win = self.np_random.choice([True, False])
                    case (True, False):
                        blue_win = self.np_random.choice(
                            [True, False], p=[self.randomness, 1 - self.randomness]
                        )
                    case (False, True):
                        blue_win = self.np_random.choice(
                            [True, False], p=[1 - self.randomness, self.randomness]
                        )
                    case (False, False):
                        blue_win = self.np_random.choice([True, False])
                    case (_, _):
                        raise ValueError(
                            f"Invalid combination of blue_agent_in_blue_territory: {blue_agent_in_blue_territory} and red_agent_in_red_territory: {red_agent_in_red_territory}"
                        )

                if blue_win:
                    reward += self.battle_reward
                    self.agents[self.num_blue_agents + red_agent_idx].terminated = True
                    self.agents[self.num_blue_agents + red_agent_idx].color = "red_grey"
                    self.game_stats["red_agent_defeated"][red_agent_idx] = True
                else:
                    reward -= self.battle_reward
                    self.agents[blue_agent_idx].terminated = True
                    self.agents[blue_agent_idx].color = "blue_grey"
                    self.game_stats["blue_agent_defeated"][blue_agent_idx] = True
            else:
                pass

        # Check if all the blue agents are defeated.
        if all(agent.terminated for agent in self.agents[0 : self.num_blue_agents]):
            terminated = True
        else:
            pass

        reward -= self.step_penalty * self.num_blue_agents

        observation: Observation = self._get_obs()
        info: dict[str, float] = self._get_info()

        return observation, reward, terminated, truncated, info


class Ctf1v1Env(CtfMvNEnv):
    """
    Environment for capture the flag game with one ego (blue) agent and one enemy (red) agent.
    """

    def __init__(
        self,
        map_path: str,
        enemy_policies: CtfPolicyT = RwPolicy(),
        battle_range: float = 1,
        randomness: float = 0.75,
        flag_reward: float = 1,
        battle_reward_ratio: float = 0.25,
        obstacle_penalty_ratio: float = 0,
        step_penalty_ratio: float = 0.01,
        max_steps: int = 100,
        observation_option: ObservationOption = "positional",
        observation_scaling: float = 1,
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
        uncached_object_types: list[str] = ["red_agent", "blue_agent"],
    ) -> None:

        num_blue_agents: Final[int] = 1
        num_red_agents: Final[int] = 1
        super().__init__(
            map_path,
            num_blue_agents,
            num_red_agents,
            enemy_policies,
            battle_range,
            randomness,
            flag_reward,
            battle_reward_ratio,
            obstacle_penalty_ratio,
            step_penalty_ratio,
            max_steps,
            observation_option,
            observation_scaling,
            render_mode,
            uncached_object_types,
        )

        self.action_space = spaces.Discrete(len(self.actions_set))
        self.ac_dim = self.action_space.n

    def step(
        self, action: int
    ) -> tuple[Observation, float, bool, bool, dict[str, float]]:
        return super().step([action])

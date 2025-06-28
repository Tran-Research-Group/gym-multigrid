from typing import Any, Literal, TypedDict

import numpy as np
import torch
from gymnasium import spaces
from numpy.typing import NDArray
from pydantic import BaseModel

from gym_multigrid.core.agent import Agent, NavigationActions
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.object import Goal, Hole, Lava, Wall
from gym_multigrid.core.world import RoomsWorld
from gym_multigrid.multigrid import (
    GridConfig,
    MultiGridEnv,
    ObservationMode,
    PartialObsConfig,
    RenderingConfig,
)
from gym_multigrid.typing import Position


class PositionalObs(ObservationMode["RoomsEnv", spaces.Box, NDArray[np.float32]]):
    def observation_space(self, env: "RoomsEnv") -> spaces.Box:
        return spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array(
                [env.width, env.height, env.width, env.height],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

    def create_observation(self, env: "RoomsEnv") -> NDArray[np.float32]:
        obs = np.array(
            [
                env.agents[0].pos[0],
                env.agents[0].pos[1],
                env.layout_config.spawn_configs[env.spawn_type].goal.pos[0],
                env.layout_config.spawn_configs[env.spawn_type].goal.pos[1],
            ]
        )
        obs = obs / np.maximum(env.width, env.height)

        return obs


class TensorObs(ObservationMode["RoomsEnv", spaces.Box, NDArray[np.int64]]):
    def observation_space(self, env: "RoomsEnv") -> spaces.Box:
        return spaces.Box(
            low=0,
            high=10,
            shape=(env.width, env.height),
            dtype=np.int64,
        )

    def create_observation(self, env: "RoomsEnv") -> NDArray[np.int64]:
        obs = np.zeros((env.width, env.height), dtype=np.int64)
        obs[:, :] = self.static_obs
        for agent in env.agents:
            obs[agent.pos[1], agent.pos[0]] = env.world.OBJECT_TO_IDX["agent"]

        return obs

    def save_static_obs(
        self, env: "RoomsEnv", options: dict[str, Any] | None = None
    ) -> None:
        static_obs: NDArray[np.int64] = (
            np.ones((env.width, env.height), dtype=np.int64)
            * env.world.OBJECT_TO_IDX["empty"]
        )
        for y, row in enumerate(env.layout_config.field_map):
            for x, cell in enumerate(row):
                if cell == "#":
                    static_obs[y, x] = env.world.OBJECT_TO_IDX["wall"]
                else:
                    pass
        for lava in env.layout_config.spawn_configs[env.spawn_type].lavas:
            static_obs[lava.pos[1], lava.pos[0]] = env.world.OBJECT_TO_IDX["lava"]
        for hole in env.layout_config.spawn_configs[env.spawn_type].holes:
            static_obs[hole.pos[1], hole.pos[0]] = env.world.OBJECT_TO_IDX["hole"]
        goal = env.layout_config.spawn_configs[env.spawn_type].goal
        static_obs[goal.pos[1], goal.pos[0]] = env.world.OBJECT_TO_IDX["goal"]
        self.static_obs = static_obs


class VectorizedTensorObs(TensorObs):
    def observation_space(self, env: "RoomsEnv") -> spaces.Box:
        return spaces.Box(
            low=0,
            high=10,
            shape=(env.width * env.height,),
            dtype=np.int64,
        )

    def create_observation(self, env: "RoomsEnv") -> NDArray[np.int64]:
        obs = super().create_observation(env)
        return obs.flatten()


class ObjConfig(BaseModel):
    pos: Position
    reward: float = 0.0


class ObjConfigDict(TypedDict):
    pos: Position
    reward: float


class SpawnConfig(BaseModel):
    agent: Position | None = None
    goal: ObjConfig  # List of goal objects, if any
    lavas: list[ObjConfig] = []  # List of lava objects, if any
    holes: list[ObjConfig] = []  # List of hole objects, if any

    model_config = {"arbitrary_types_allowed": True}


class SpawnConfigDict(TypedDict, total=False):
    agent: Position | None
    goal: ObjConfigDict
    lavas: list[ObjConfigDict]
    holes: list[ObjConfigDict]


class LayoutConfig(BaseModel):
    field_map: list[str]
    spawn_configs: list[SpawnConfig]


class LayoutConfigDict(TypedDict):
    field_map: list[str]
    spawn_configs: list[SpawnConfigDict]


class RewardConfig(BaseModel):
    step_penalty: float = 0.01
    sum_reward: bool = True


class RewardConfigDict(TypedDict):
    step_penalty: float
    sum_reward: bool


class RoomsEnv(MultiGridEnv[NDArray[np.int64] | NDArray[np.float32]]):
    """
    Environment for capture the flag with multiple agents with N blue agents and M red agents.
    """

    observation_modes: dict[
        str,
        type[ObservationMode["RoomsEnv", spaces.Box, NDArray[np.int64]]]
        | type[ObservationMode["RoomsEnv", spaces.Box, NDArray[np.float32]]],
    ] = {
        "positional": PositionalObs,
        "tensor": TensorObs,
        "vectorized_tensor": VectorizedTensorObs,
    }
    layout_config: LayoutConfig

    def __init__(
        self,
        spawn_type: int = 0,
        layout_config: LayoutConfigDict = {
            "field_map": [
                "#############",
                "#     #     #",
                "#     #     #",
                "#           #",
                "#     #     #",
                "#     #     #",
                "## ####     #",
                "#     ### ###",
                "#     #     #",
                "#     #     #",
                "#           #",
                "#     #     #",
                "#############",
            ],
            "spawn_configs": [
                {
                    "agent": (9, 3),
                    "goal": {"pos": (3, 9), "reward": 1.0},
                },
                {
                    "agent": (11, 1),
                    "goal": {"pos": (7, 9), "reward": 1.0},
                },
                {
                    "agent": (9, 3),
                    "goal": {"pos": (9, 9), "reward": 1.0},
                },
                {
                    "agent": (3, 9),
                    "goal": {"pos": (9, 4), "reward": 1.0},
                    "lavas": [
                        {"pos": (8, 4), "reward": 0},
                        {"pos": (9, 2), "reward": 0},
                        {"pos": (11, 1), "reward": 0},
                        {"pos": (5, 3), "reward": 0},
                        {"pos": (3, 5), "reward": 0},
                        {"pos": (3, 2), "reward": 0},
                        {"pos": (5, 9), "reward": -1},
                        {"pos": (3, 8), "reward": -1},
                        {"pos": (2, 11), "reward": -1},
                        {"pos": (10, 8), "reward": -1},
                        {"pos": (8, 9), "reward": -1},
                        {"pos": (7, 11), "reward": -1},
                    ],
                    "holes": [
                        {"pos": (7, 3), "reward": 0},
                        {"pos": (10, 5), "reward": 0},
                        {"pos": (8, 6), "reward": 0},
                        {"pos": (4, 4), "reward": -1},
                        {"pos": (2, 3), "reward": -1},
                        {"pos": (1, 1), "reward": -1},
                        {"pos": (2, 7), "reward": 0},
                        {"pos": (1, 9), "reward": 0},
                        {"pos": (4, 10), "reward": 0},
                        {"pos": (7, 8), "reward": -1},
                        {"pos": (9, 10), "reward": -1},
                        {"pos": (11, 11), "reward": -1},
                    ],
                },
            ],
        },
        state_representation: str = "tensor",
        reward_config: RewardConfigDict = {
            "step_penalty": 0.0,
            "sum_reward": True,
        },
        random_init_pos: bool = False,
        tile_size: int = 32,
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
    ) -> None:
        """
        Initialize a new capture the flag environment.

        Parameters
        ----------

        """
        ### fundamental parameters
        self.spawn_type: int = spawn_type
        # spawn_configs: list[SpawnConfig] = [
        #     SpawnConfig(**conf) for conf in layout_config["spawn_configs"]
        # ]
        self.layout_config: LayoutConfig = LayoutConfig.model_validate(layout_config)
        self.reward_config: RewardConfig = RewardConfig.model_validate(reward_config)
        self.random_init_pos: bool = random_init_pos

        grid_size: tuple[int, int] = (
            len(self.layout_config.field_map[0]),
            len(self.layout_config.field_map),
        )
        self.state_representation = self.observation_modes[state_representation]()

        width, height = grid_size
        world = RoomsWorld
        actions_set = NavigationActions

        # NOTE: currently only one agent is supported
        agents = [
            Agent(
                world,
                color="blue",
                bg_color=None,
                actions=actions_set,
                type="agent",
            )
        ]

        grid_config = GridConfig(
            grid_size=None,
            width=width,
            height=height,
            world=world,
            actions_set=actions_set,
        )
        render_config = RenderingConfig(tile_size=tile_size, render_mode=render_mode)
        partial_obs_config = PartialObsConfig()

        super().__init__(
            agents=agents,
            **dict(grid_config),
            **render_config.model_dump(),
            **dict(partial_obs_config),
        )

    def _set_observation_space(self) -> spaces.Box:
        observation_space = self.state_representation.observation_space(self)

        return observation_space

    def _gen_grid(self, width: int, height: int):
        # Create the grid
        self.grid = Grid(width, height, self.world)

        # Translate the maze structure into the grid
        for y, row in enumerate(self.layout_config.field_map):
            for x, cell in enumerate(row):
                if cell == "#":
                    self.grid.set(x, y, Wall(self.world))
                else:
                    pass

        # place goal
        goal = self.layout_config.spawn_configs[self.spawn_type].goal
        goal_obj = Goal(
            self.world,
            0,
            color="green",
            reward=goal.reward,
            absorbing=True,
        )
        assert isinstance(goal_obj, Goal), "Goal object must be of type Goal"
        self.put_obj(goal_obj, *goal.pos)

        # place lavas
        for i, lava in enumerate(
            self.layout_config.spawn_configs[self.spawn_type].lavas
        ):
            lava_obj = Lava(
                self.world,
                color="red",
                reward=lava.reward,
                absorbing=lava.reward < 0,
            )
            self.put_obj(lava_obj, *lava.pos)

        # place holes
        for i, hole in enumerate(
            self.layout_config.spawn_configs[self.spawn_type].holes
        ):
            hole_obj = Hole(
                self.world,
                color="purple",
                bg_color=None,
                reward=hole.reward,
                absorbing=hole.reward < 0,
            )
            self.put_obj(hole_obj, *hole.pos)

        self.state_representation.save_static_obs(self, {})

        self.lava_pos: list[Position] = [
            lava.pos for lava in self.layout_config.spawn_configs[self.spawn_type].lavas
        ]
        self.hole_pos: list[Position] = [
            hole.pos for hole in self.layout_config.spawn_configs[self.spawn_type].holes
        ]
        self.goal_pos: Position = self.layout_config.spawn_configs[
            self.spawn_type
        ].goal.pos

    def _reset_agents(self):
        """
        Reset the agents' positions in the grid.
        If random_init_pos is True, randomly select a position from the grid.
        """
        for agent in self.agents:
            agent_positions = self.layout_config.spawn_configs[self.spawn_type].agent
            if self.random_init_pos:
                agent_positions = None
            self.place_agent(agent, pos=agent_positions, reset_agent_status=True)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        if isinstance(options, dict):
            self.spawn_type = options.get("spawn_type", self.spawn_type)
            self.layout_config.spawn_configs = [
                SpawnConfig(**conf)
                for conf in options.get(
                    "spawn_configs",
                    [
                        config.model_dump()
                        for config in self.layout_config.spawn_configs
                    ],
                )
            ]

        super().reset(seed=seed, options=options)
        self.state_representation.save_static_obs(self)

        ### NOTE: NOT MULTIAGENT SETTING
        observations = self._get_obs()
        info = {"is_success": False}

        return observations, info

    def step(
        self, action: np.int64 | NDArray[np.int64]
    ) -> tuple[
        NDArray[np.int64] | NDArray[np.float32],
        NDArray[np.float64] | float,
        bool,
        bool,
        dict[str, Any],
    ]:
        self.step_count += 1

        ### NOTE: MULTIAGENT SETTING NOT IMPLEMENTED
        assert action.size == 1, "Only one agent is supported in this environment."
        actions: list[int] = np.array([action], dtype=np.int64).flatten().tolist()
        order = np.random.permutation(len(actions))

        rewards = np.zeros(len(actions))
        info = {"is_success": False}
        terminated: bool = False
        for i in order:
            agent: Agent = self.agents[i]
            assert isinstance(agent, Agent)
            if agent.terminated or agent.paused or not agent.started:
                continue

            # Get the current agent position
            curr_pos: Position = agent.pos

            # Rotate left
            self.actions: type[NavigationActions]
            fwd_pos: Position
            match actions[i]:
                case self.actions.LEFT:
                    fwd_pos = agent.west_pos(in_tuple=True)
                case self.actions.RIGHT:
                    fwd_pos = agent.east_pos(in_tuple=True)
                case self.actions.UP:
                    fwd_pos = agent.north_pos(in_tuple=True)
                case self.actions.DOWN:
                    fwd_pos = agent.south_pos(in_tuple=True)
                case self.actions.STAY:
                    fwd_pos = curr_pos
                case _:
                    raise ValueError(
                        f"Unknown action: {actions[i]}. Expected one of {self.actions}"
                    )

            info["is_success"] = False
            fwd_cell = self.grid.get(*fwd_pos)
            if fwd_cell is not None:
                if fwd_cell.can_overlap():
                    agent.move(fwd_pos, self.grid, self.init_grid)
                    if fwd_cell.absorbing:
                        terminated = True
                        rewards[i] = fwd_cell.reward
                        if isinstance(fwd_cell, Goal):
                            info["is_success"] = True
                        else:
                            pass
                else:
                    pass
            else:
                agent.move(fwd_pos, self.grid, self.init_grid)

        if self.reward_config.sum_reward:
            rewards = np.sum(rewards)
        else:
            pass
        rewards -= self.reward_config.step_penalty

        ### NOTE: not multiagent setting
        truncated: bool = False

        observations = self._get_obs()

        return observations, rewards, terminated, truncated, info

    def _get_obs(self):
        return self.state_representation.create_observation(self)

    def get_rewards_heatmap(self, extractor: torch.nn.Module, eigenvectors: np.ndarray):
        assert self.state_representation in [
            "vectorized_tensor",
            "tensor",
        ], f"Unsupported state representation: {self.state_representation}"

        # Environment indices
        empty_idx = 1
        goal_idx = 8
        agent_idx = 10
        wall_idx = 2

        # Get base state
        state, _ = self.reset()
        agent_pos = np.where(state == agent_idx)
        state[agent_pos] = empty_idx
        self.close()

        if self.state_representation != "tensor":
            state = state.reshape(self.width, self.height, -1)

        mask = (state != wall_idx) & (state != goal_idx)
        non_mask = ~mask

        heatmaps = []
        grid_shape = (self.width, self.height, 1)
        for n in range(eigenvectors.shape[0]):
            eig = eigenvectors[n]
            reward_map = np.full(grid_shape, fill_value=np.nan)

            for i in range(grid_shape[0]):
                for j in range(grid_shape[1]):
                    current_idx = (i, j, 0)
                    current_val = state[current_idx]

                    if current_val == wall_idx or current_val == goal_idx:
                        reward_map[current_idx] = 0.0
                    else:
                        # Copy and manipulate state
                        state_copy = np.copy(state)
                        state_copy[current_idx] = agent_idx
                        with torch.no_grad():
                            feature, _ = extractor(state_copy)
                        feature = feature.cpu().numpy().squeeze(0)

                        reward = np.dot(eig, feature)
                        reward_map[current_idx] = reward

            # reward_map = # normalize between -1 to 1
            pos_mask = np.logical_and(mask, (reward_map > 0))
            neg_mask = np.logical_and(mask, (reward_map < 0))

            # Normalize positive values to [0, 1]
            if np.any(pos_mask):
                pos_max, pos_min = (
                    reward_map[pos_mask].max(),
                    reward_map[pos_mask].min(),
                )
                if pos_max != pos_min:
                    reward_map[pos_mask] = (reward_map[pos_mask] - pos_min) / (
                        pos_max - pos_min + 1e-4
                    )

            # Normalize negative values to [-1, 0]
            if np.any(neg_mask):
                neg_max, neg_min = (
                    reward_map[neg_mask].max(),
                    reward_map[neg_mask].min(),
                )
                if neg_max != neg_min:
                    reward_map[neg_mask] = (reward_map[neg_mask] - neg_min) / (
                        neg_max - neg_min + 1e-4
                    ) - 1.0

            # Set all other entries (walls, empty) to 0
            reward_map = reward_map.reshape(self.width, self.height, -1)
            reward_map = self.reward_map_to_rgb(reward_map, mask)

            # set color theme as blue and red (blue = -1 and red = 1)
            # set wall color at value 0 and goal idx as 1
            heatmaps.append(reward_map)

        return heatmaps

    def reward_map_to_rgb(self, reward_map: np.ndarray, mask) -> np.ndarray:
        rgb_img = np.zeros((self.width, self.height, 3), dtype=np.float32)

        pos_mask = np.logical_and(mask, (reward_map > 0))
        neg_mask = np.logical_and(mask, (reward_map < 0))

        # Blue for negative: map [-1, 0] → [1, 0]
        rgb_img[neg_mask[:, :, 0], 2] = -reward_map[neg_mask]  # blue channel

        # Red for positive: map [0, 1] → [0, 1]
        rgb_img[pos_mask[:, :, 0], 0] = reward_map[pos_mask]  # red channel

        # rgb_img.flatten()[mask] to grey
        rgb_img[~mask[:, :, 0], :] = 0.5

        return rgb_img

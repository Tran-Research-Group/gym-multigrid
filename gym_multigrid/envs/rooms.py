from typing import Any, Literal

import numpy as np
import torch
from gymnasium import spaces
from numpy.typing import NDArray
from pydantic import BaseModel

from gym_multigrid.core.agent import Agent, GridActions
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.object import Goal, Wall
from gym_multigrid.core.world import GridWorld
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
                env.layout_config.spawn_configs[env.spawn_type].goal_pos[0],
                env.layout_config.spawn_configs[env.spawn_type].goal_pos[1],
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
            obs[agent.pos[0], agent.pos[1]] = env.world.OBJECT_TO_IDX["agent"]

        return obs

    def save_static_obs(
        self, env: "RoomsEnv", options: dict[str, Any] | None = None
    ) -> None:
        static_obs: NDArray[np.int64] = (
            np.ones((env.width, env.height), dtype=np.int64)
            * env.world.OBJECT_TO_IDX["empty"]
        )
        for x, row in enumerate(env.layout_config.field_map):
            for y, cell in enumerate(row):
                if cell == "#":
                    static_obs[x, y] = env.world.OBJECT_TO_IDX["wall"]
                else:
                    pass
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


class SpawnConfig(BaseModel):
    agent_pos: Position
    goal_pos: Position


class LayoutConfig(BaseModel):
    field_map: list[str]
    spawn_configs: list[SpawnConfig]


DEFAULT_LAYOUT_CONFIG = LayoutConfig(
    field_map=[
        "#############",
        "#    #      #",
        "#    #      #",
        "#           #",
        "#    #      #",
        "#    #      #",
        "## ###### ###",
        "#     #     #",
        "#     #     #",
        "#     #     #",
        "#           #",
        "#     #     #",
        "#############",
    ],
    spawn_configs=[
        SpawnConfig(agent_pos=(9, 3), goal_pos=(3, 9)),
        SpawnConfig(agent_pos=(11, 1), goal_pos=(7, 9)),
        SpawnConfig(agent_pos=(9, 3), goal_pos=(9, 9)),
    ],
)


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
        layout_config: LayoutConfig = DEFAULT_LAYOUT_CONFIG,
        tile_size: int = 10,
        state_representation: str = "tensor",
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
    ) -> None:
        """
        Initialize a new capture the flag environment.

        Parameters
        ----------

        """
        ### fundamental parameters
        self.layut_config = layout_config
        self.spawn_type: int = spawn_type
        grid_size: tuple[int, int] = (
            len(self.layut_config.field_map[0]),
            len(self.layut_config.field_map),
        )
        self.state_representation = self.observation_modes[state_representation]()

        width, height = grid_size
        world = GridWorld
        actions_set = GridActions

        # NOTE: currently only one agent is supported
        agents = [
            Agent(
                self.world,
                color="blue",
                bg_color="light_blue",
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
        for x, row in enumerate(self.layout_config.field_map):
            for y, cell in enumerate(row):
                if cell == "#":
                    self.grid.set(x, y, Wall(self.world))
                else:
                    pass

        # place goal
        goal = Goal(self.world, 0)
        self.put_obj(goal, *self.layut_config.spawn_configs[self.spawn_type].goal_pos)

        self.state_representation.save_static_obs(self, {})

    def _reset_agents(self, random_init_pos: bool = False):
        """
        Reset the agents' positions in the grid.
        If random_init_pos is True, randomly select a position from the grid.
        """
        for agent in self.agents:
            if random_init_pos:
                self.place_agent(agent)
            else:
                agent_positions = self.layut_config.spawn_configs[
                    self.spawn_type
                ].agent_pos
                self.place_agent(agent, pos=agent_positions)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        # obs, info = super().reset(seed=seed, options=options)
        super().reset(seed=seed, options=options)
        self.state_representation.save_static_obs(self)

        ### NOTE: NOT MULTIAGENT SETTING
        observations = self._get_obs()
        info = {"success": False}

        return observations, info

    def step(self, action: np.int64 | NDArray[np.int64]):
        self.step_count += 1

        ### NOTE: MULTIAGENT SETTING NOT IMPLEMENTED
        assert action.size == 1, "Only one agent is supported in this environment."
        actions: list[int] = np.array([action], dtype=np.int64).flatten().tolist()
        order = np.random.permutation(len(actions))

        rewards = np.zeros(len(actions))
        info = {"success": False}
        terminated: bool = False
        for i in order:
            agent: Agent = self.agents[i]
            assert isinstance(agent, Agent)
            if agent.terminated or agent.paused or not agent.started:
                continue

            # Get the current agent position
            curr_pos: Position = agent.pos

            # Rotate left
            self.actions: type[GridActions]
            if actions[i] == self.actions.LEFT:
                # Get the contents of the cell in front of the agent
                fwd_pos = agent.west_pos(in_tuple=True)
                fwd_cell = self.grid.get(*fwd_pos)

            # Rotate right
            elif actions[i] == self.actions.RIGHT:
                # Get the contents of the cell in front of the agent
                fwd_pos = agent.east_pos(in_tuple=True)
                fwd_cell = self.grid.get(*fwd_pos)

            # Move forward
            elif actions[i] == self.actions.UP:
                # Get the contents of the cell in front of the agent
                fwd_pos = agent.north_pos(in_tuple=True)
                fwd_cell = self.grid.get(*fwd_pos)

            elif actions[i] == self.actions.DOWN:
                # Get the contents of the cell in front of the agent
                fwd_pos = agent.south_pos(in_tuple=True)
                fwd_cell = self.grid.get(*fwd_pos)
            elif actions[i] == self.actions.STAY:
                # Get the contents of the cell in front of the agent
                fwd_pos = curr_pos
                fwd_cell = self.grid.get(*fwd_pos)
            else:
                assert False, "unknown action"

            if fwd_cell is not None:
                if fwd_cell.type == "goal":
                    terminated = True
                    rewards = self._reward(i, rewards, 1)
                    info["success"] = True
                else:
                    pass
            elif fwd_cell is None or fwd_cell.can_overlap():
                self.grid.set(*agent.pos, None)
                self.grid.set(*fwd_pos, agent)
                agent.pos = fwd_pos
            else:
                # If the cell in front of the agent is not empty, do nothing
                pass

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

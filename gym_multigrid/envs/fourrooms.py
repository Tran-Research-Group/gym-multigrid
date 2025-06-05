import random
from typing import Literal, TypeAlias, TypedDict

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from numpy.typing import NDArray

from gym_multigrid.core.agent import Agent, GridActions
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.object import Goal, Wall
from gym_multigrid.core.world import GridWorld
from gym_multigrid.multigrid import (
    DEFAULT_FULL_OBS_ENV_PARTIAL_OBS_CONFIG,
    GridConfig,
    MultiGridEnv,
    ObservationMode,
    PartialObsConfig,
    RenderingConfig,
)
from gym_multigrid.typing import Position


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


class PositionalObs(ObservationMode["FourRoomsEnv", NDArray[np.float32]]):
    def observation_space(self, env: "FourRoomsEnv") -> spaces.Box:
        return spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array(
                [env.width, env.height, env.width, env.height],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

    def create_observation(self, env: "FourRoomsEnv") -> NDArray[np.float32]:
        obs = np.array(
            [
                env.agents[0].pos[0],
                env.agents[0].pos[1],
                env.goal_positions[env.grid_type][0],
                env.goal_positions[env.grid_type][1],
            ]
        )
        obs = obs / np.maximum(env.grid_size[0], env.grid_size[1])

        return obs


class TensorObs(ObservationMode["FourRoomsEnv", NDArray[np.int64]]):
    def observation_space(self, env: "FourRoomsEnv") -> spaces.Box:
        return spaces.Box(
            low=0,
            high=10,
            shape=(env.width, env.height),
            dtype=np.int64,
        )

    def create_observation(self, env: "FourRoomsEnv") -> NDArray[np.int64]:
        obs = np.zeros((env.width, env.height), dtype=np.int64)
        obs[:, :] = env.static_obs
        for agent in env.agents:
            obs[agent.pos[0], agent.pos[1]] = env.world.OBJECT_TO_IDX["agent"]

        return obs


class FourRoomsEnv(MultiGridEnv[NDArray[np.int64] | NDArray[np.float32]]):
    """
    Environment for capture the flag with multiple agents with N blue agents and M red agents.
    """

    metadata = MultiGridEnv.metadata.copy()
    # metadata["observation_modes"] = {
    #     "vectorized_tensor":
    #     "tensor": TensorObservationMode,
    #     "positional": MapObservationMode,
    # }

    def __init__(
        self,
        grid_type: int = 0,
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
        self.grid_size = (13, 13)
        self.state_representation = state_representation

        if grid_type < 0 or grid_type >= 3:
            raise ValueError(
                f"The Fourroom only accepts grid_type of 0 and 1, given {grid_type}"
            )
        else:
            self.grid_type = grid_type

        width = self.grid_size[0]
        height = self.grid_size[1]
        world = GridWorld
        actions_set = GridActions

        see_through_walls: bool = False

        agents = [
            Agent(
                self.world,
                color="blue",
                bg_color="light_blue",
                actions=actions_set,
                type="agent",
            )
        ]

        self.goal_positions: list[Position] = [
            (3, 9),
            (7, 9),
            (9, 9),
        ]
        self.agent_positions: list[Position] = [
            (9, 3),
            (11, 1),
            (9, 3),
        ]

        self.map: list[str] = [
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
        ]

        super().__init__(
            width=width,
            height=height,
            see_through_walls=see_through_walls,
            agents=agents,
            actions_set=actions_set,
            world=world,
            render_mode=render_mode,
            tile_size=tile_size,
        )

    def _set_observation_space(self) -> spaces.Dict | spaces.Box:
        match self.state_representation:
            case "positional":
                observation_space = spaces.Box(
                    low=np.array([0, 0, 0, 0], dtype=np.float32),
                    high=np.array(
                        [self.width, self.height, self.width, self.height],
                        dtype=np.float32,
                    ),
                    dtype=np.float32,
                )
            case "tensor":
                observation_space = spaces.Box(
                    low=0,
                    high=10,
                    shape=(self.width, self.height, self.world.encode_dim),
                    dtype=np.int64,
                )
            case "vectorized_tensor":
                observation_space = spaces.Box(
                    low=0,
                    high=10,
                    shape=(self.width * self.height * self.world.encode_dim,),
                    dtype=np.int64,
                )
            case _:
                raise ValueError(
                    f"Invalid state representation: {self.state_representation}"
                )

        return observation_space

    def _gen_grid(self, width: int, height: int):
        # Create the grid
        self.grid = Grid(width, height, self.world)

        # Translate the maze structure into the grid
        self.static_obs: NDArray[np.int64] = np.zeros(
            (self.width, self.height), dtype=np.int64
        )
        for x, row in enumerate(self.map):
            for y, cell in enumerate(row):
                if cell == "#":
                    self.grid.set(x, y, Wall(self.world))
                    self.static_obs[x, y] = self.world.OBJECT_TO_IDX["wall"]
                elif cell == " ":
                    self.grid.set(x, y, None)

        # place goal
        goal = Goal(self.world, 0)
        self.put_obj(goal, *self.goal_positions[self.grid_type])
        goal.init_pos, goal.pos = self.goal_positions[self.grid_type]

        # place agent
        if options.get("random_init_pos"):
            coords = self.find_obj_coordinates(None)
            agent_positions = random.sample(coords, 1)[0]
        else:
            agent_positions = self.agent_positions[self.grid_type]

        for agent in self.agents:
            self.place_agent(agent, pos=agent_positions)

    def find_obj_coordinates(self, obj) -> tuple[int, int] | None:
        """
        Finds the coordinates (i, j) of the first occurrence of None in the grid.
        Returns None if no None value is found.
        """
        coord_list = []
        for index, value in enumerate(self.grid.grid):
            if value is obj:
                # Calculate the (i, j) coordinates from the 1D index
                i = index % self.width
                j = index // self.width
                coord_list.append((i, j))
        return coord_list

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict = {},
    ):
        # obs, info = super().reset(seed=seed, options=options)
        super().reset(seed=seed, options=options)

        ### NOTE: NOT MULTIAGENT SETTING
        observations = self.get_obs()
        info = {"success": False}

        return observations, info

    def step(self, actions):
        self.step_count += 1

        ### NOTE: MULTIAGENT SETTING NOT IMPLEMENTED
        actions = np.argmax(actions)
        actions = [actions]
        order = np.random.permutation(len(actions))

        rewards = np.zeros(len(actions))
        info = {"success": False}
        for i in order:
            if (
                self.agents[i].terminated
                or self.agents[i].paused
                or not self.agents[i].started
            ):
                continue

            # Get the current agent position
            curr_pos = self.agents[i].pos
            done = False

            # Rotate left
            if actions[i] == self.actions.left:
                # Get the contents of the cell in front of the agent
                fwd_pos = tuple(a + b for a, b in zip(curr_pos, (0, -1)))
                fwd_cell = self.grid.get(*fwd_pos)

                if fwd_cell is not None:
                    if fwd_cell.type == "goal":
                        done = True
                        rewards = self._reward(i, rewards, 1)
                        info["success"] = True
                elif fwd_cell is None or fwd_cell.can_overlap():
                    self.grid.set(*self.agents[i].pos, None)
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.agents[i].pos = fwd_pos
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)

            # Rotate right
            elif actions[i] == self.actions.right:
                # Get the contents of the cell in front of the agent
                fwd_pos = tuple(a + b for a, b in zip(curr_pos, (0, +1)))
                fwd_cell = self.grid.get(*fwd_pos)
                if fwd_cell is not None:
                    if fwd_cell.type == "goal":
                        done = True
                        rewards = self._reward(i, rewards, 1)
                        info["success"] = True
                elif fwd_cell is None or fwd_cell.can_overlap():
                    self.grid.set(*self.agents[i].pos, None)
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.agents[i].pos = fwd_pos
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)

            # Move forward
            elif actions[i] == self.actions.up:
                # Get the contents of the cell in front of the agent
                fwd_pos = tuple(a + b for a, b in zip(curr_pos, (-1, 0)))
                fwd_cell = self.grid.get(*fwd_pos)
                if fwd_cell is not None:
                    if fwd_cell.type == "goal":
                        done = True
                        rewards = self._reward(i, rewards, 1)
                        info["success"] = True
                elif fwd_cell is None or fwd_cell.can_overlap():
                    self.grid.set(*self.agents[i].pos, None)
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.agents[i].pos = fwd_pos
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)

            elif actions[i] == self.actions.down:
                # Get the contents of the cell in front of the agent
                fwd_pos = tuple(a + b for a, b in zip(curr_pos, (+1, 0)))
                fwd_cell = self.grid.get(*fwd_pos)
                if fwd_cell is not None:
                    if fwd_cell.type == "goal":
                        done = True
                        rewards = self._reward(i, rewards, 1)
                        info["success"] = True
                elif fwd_cell is None or fwd_cell.can_overlap():
                    self.grid.set(*self.agents[i].pos, None)
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.agents[i].pos = fwd_pos
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)
            elif actions[i] == self.actions.stay:
                # Get the contents of the cell in front of the agent
                fwd_pos = curr_pos
                fwd_cell = self.grid.get(*fwd_pos)
                self.agents[i].pos = fwd_pos
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)
            else:
                assert False, "unknown action"

        ### NOTE: not multiagent setting
        terminated = done
        truncated = False

        observations = self.get_obs()

        return observations, rewards, terminated, truncated, info

    def get_obs(
        self,
    ):
        if self.state_representation == "positional":
            obs = np.array(
                [
                    self.agents[0].pos[0],
                    self.agents[0].pos[1],
                    self.goal_positions[self.grid_type][0],
                    self.goal_positions[self.grid_type][1],
                ]
            )
            obs = obs / np.maximum(self.grid_size[0], self.grid_size[1])
        elif self.state_representation == "tensor":
            obs = [
                self.grid.encode_for_agents(agent_pos=self.agents[i].pos)
                for i in range(len(self.agents))
            ]
            obs = [self.world.normalize_obs * ob for ob in obs]
            obs = obs[0][:, :, 0:1]
        elif self.state_representation == "vectorized_tensor":
            obs = [
                self.grid.encode_for_agents(agent_pos=self.agents[i].pos)
                for i in range(len(self.agents))
            ]
            obs = [self.world.normalize_obs * ob for ob in obs]
            obs = obs[0][:, :, 0:1].flatten()
        else:
            raise ValueError(
                f"Unknown state representation {self.state_representation}. "
                "Please use 'positional' or 'tensor'."
            )
        return obs

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

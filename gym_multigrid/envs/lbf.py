from enum import IntEnum
from collections import defaultdict
from typing import Any, Type, Literal, Optional
import math

import numpy as np
from numpy.typing import NDArray
from gymnasium import spaces

from gym_multigrid.core.agent import Agent, LBFActions
from gym_multigrid.core.constants import DIR_TO_VEC
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.object import Goal, Wall, WorldObj
from gym_multigrid.core.world import LBFWorld, World
from gym_multigrid.utils.rendering import fill_coords, point_in_circle, point_in_rect
from gym_multigrid.typing_utils import Position
from gym_multigrid.multigrid import MultiGridEnv


# from gym_multigrid.policy import AgentPolicy
# from gym_multigrid.policy.prey_pred import PREY_PRED_POLICIES
from gym_multigrid.policy.prey_pred.utils import a_star


class LBFAgent(Agent):
    def __init__(
        self,
        world: World,
        index: int,
        color: str,
        view_size: Optional[int] = None,
        init_pos: Optional[tuple[int, int]] = None,
        level: Optional[int] = None,
    ):

        self.init_pos = init_pos
        self.level = level

        self.reward: float = 0.0
        self.neighbor_pos_offsets: NDArray[np.int_] = np.array(
            [[-1, 0], [1, 0], [0, -1], [0, 1]]
        )
        self.neighbor_pos: NDArray[np.int_] = np.zeros((4, 2), dtype=np.int_)

        super().__init__(
            world=world,
            index=index,
            actions=LBFActions,
            color=color,
            type="agent",
            view_size=view_size,
            dir_to_vec=DIR_TO_VEC,
        )

    @property
    def pos(self) -> Position:
        return (self._pos[0], self._pos[1])

    @pos.setter
    def pos(self, pos: Position) -> None:
        self._pos = pos
        if pos is not None:
            self.neighbor_pos = pos + self.neighbor_pos_offsets

    def reset(self, level: int, init_pos: tuple[int, int]) -> None:
        super().reset()
        if self.pos is not None:
            self.neighbor_pos = self.pos + self.neighbor_pos_offsets
        else:
            self.neighbor_pos = np.zeros((4, 2), dtype=np.int_)

        self.level = level
        self.init_pos = init_pos
        self.reward: float = 0.0

    def encode(self, current_agent: bool = False) -> tuple[int, ...]:
        """Encode a description of this object as a 3-tuple of integers

        Parameters
        ----------
        current_agent : bool, optional
            whether the agent is the current agent, by default False
        """
        if self.world.encode_dim == 3:
            return (
                self.world.OBJECT_TO_IDX[self.type],
                self.world.COLOR_TO_IDX[self.color],
                self.level,
            )

    def render(self, img: NDArray[np.uint8]):
        fill_coords(
            img, point_in_rect(0.15, 0.85, 0.15, 0.85), self.world.COLORS[self.color]
        )


class Fruit(WorldObj):
    def __init__(
        self,
        world: World,
        color: str,
        level: int,
    ):
        self.level = level

        self.neighbor_pos_offsets: NDArray[np.int_] = np.array(
            [[-1, 0], [1, 0], [0, -1], [0, 1]]
        )
        self.neighbor_pos: NDArray[np.int_] = np.zeros((4, 2), dtype=np.int_)

        super().__init__(
            world=world,
            color=color,
            type="fruit",
        )

    @property
    def pos(self) -> Position:
        return (self._pos[0], self._pos[1])

    @pos.setter
    def pos(self, pos: Position) -> None:
        self._pos = pos
        if pos is not None:
            self.neighbor_pos = pos + self.neighbor_pos_offsets

    def can_pickup(self):
        return True

    def check_capture_condition(self, grid: Grid) -> bool:
        """
        Check if the fruit can be collected.

        Parameters
        ----------
        grid : Grid
            The grid of the environment.

        Returns
        -------
        capturable : bool
            True if the fruit can be collected, False otherwise.
        """
        level_neighbor_agents: int = 0

        agent_ids: list[int] = []

        for neighbor_pos in self.neighbor_pos:
            cell = grid.get(*neighbor_pos)
            if isinstance(cell, Agent):
                level_neighbor_agents += cell.level
                agent_ids.append(cell.index)
            else:
                pass

        capturable: bool = level_neighbor_agents >= self.level
        # agent_ids = agent_ids if capturable else []

        return capturable

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), self.world.COLORS[self.color])

    def reset(self) -> None:
        super().reset()
        if self.pos is not None:
            self.neighbor_pos = self.pos + self.neighbor_pos_offsets
        else:
            self.neighbor_pos = np.zeros((4, 2), dtype=np.int_)

    def encode(self, current_agent: bool = False) -> tuple[int, ...]:
        """Encode the a description of this object as a 3-tuple of integers"""
        if self.world.encode_dim == 3:
            return (
                self.world.OBJECT_TO_IDX[self.type],
                self.world.COLOR_TO_IDX[self.color],
                self.level,
            )
        else:
            return (
                self.world.OBJECT_TO_IDX[self.type],
                self.world.COLOR_TO_IDX[self.color],
                self.level,
                0,
                0,
                0,
            )


class LBFGameEnv(MultiGridEnv):
    """
    Environment in which the agents have to collect the balls
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self,
        n_agents: int = 4,
        max_num_fruit: int = 2,
        sight: int = 2,
        field_size: Optional[int] = None,
        width: Optional[int] = 10,
        height: Optional[int] = 10,
        normalize_reward: bool = True,
        min_agent_level: int = 1,
        max_agent_level: int = 2,
        min_fruit_level: int = 1,
        max_fruit_level: Optional[int] = None,
        failed_load_penalty: float = 0.0,
        force_coop: bool = False,
        observe_agent_levels: bool = True,
        state_type: Literal["original", "multigrid_encode"] = "original",
        obs_type: Literal["original"] = "original",
        layout_config: dict = {},
        fruit_config: dict = {},
    ):
        """
        Initialize the LBFGameEnv.

        Parameters
        ----------
        size : int
            Size of grid if square. Default 19
        num_fruit : list[int]
            Number of fruit of each type present in environment.
        agents_index : list[int]
            Colour index for each agent.
        fruits_index : list[int]
            Colour index for each fruit type.
        fruits_reward : list[float]
            Reward given for collecting each fruit type.
        """
        # add 2 b/c of the outer wall
        if field_size is not None:
            width = field_size + 2
            height = field_size + 2
        else:
            if width is not None and height is not None:
                width = width + 2
                height = height + 2
            else:
                raise ValueError("Please specify either field_size or (width, height)")

        # reward config
        self.failed_load_penalty = failed_load_penalty
        self._normalize_reward = normalize_reward

        self.num_agents = n_agents
        self.max_num_fruit: int = max_num_fruit
        self._num_fruit_spawned: int = 0

        self.sight = sight
        self.force_coop = force_coop
        self._observe_agent_levels = observe_agent_levels

        self.state_type = state_type
        self.obs_type = obs_type

        self.min_agent_levels = np.array([min_agent_level] * self.num_agents)
        self.max_agent_levels = np.array([max_agent_level] * self.num_agents)

        self.min_fruit_level = min_fruit_level
        self.max_fruit_level = max_fruit_level
        self.spawn_attempts: int = 1000

        # multi-room support
        self.num_rooms = layout_config.get("num_rooms", None)

        self.collected_fruit: int = 0

        self.world = LBFWorld
        self.actions_set = LBFActions
        partial_obs: bool = False

        # initial encoding for objects in the observation
        self.init_object_obs = np.array([-1, -1, 0]).reshape(1, -1)

        # init agents
        agents = []
        """
        old logic, doesn't allow random spawning of agent + fruit
        if kwargs.get("agent_config"):
            # agent colors, unique indices
            self.agents_index = kwargs["agent_config"]["index"]

            for i, agent_index in enumerate(self.agents_index):
                init_pos = kwargs["agent_config"]["init_pos"][i]
                if isinstance(init_pos, list):
                    init_pos = tuple(init_pos)

                temp_agent_config: AgentConfigDict = {
                    "init_pos": init_pos,
                    "level": kwargs["agent_config"]["level"][i],
                    "color": self.world.IDX_TO_COLOR[agent_index],
                }
                agents.append(
                    LBFAgent(
                        world=self.world,
                        index=i,
                        agent_config=temp_agent_config,
                        view_size=self.sight,
                    )
                )
        """
        for i in range(self.num_agents):
            agents.append(
                LBFAgent(
                    world=self.world,
                    index=i,
                    color=self.world.IDX_TO_COLOR[0],
                    view_size=self.sight,
                )
            )

        # optional env config, allows deterministic design of env with a config file, does not support random spawning of objects
        self.waypoint_pos: list[Position]

        if layout_config.get("room_configs", False):
            self.waypoints = self._config_lists_to_tuples(layout_config["room_configs"])
        else:
            self.waypoints = None

        self.field_map = layout_config.get("field_map", None)

        if fruit_config:
            self.max_num_fruit = kwargs["fruit_config"]["num_fruit"]
            self.total_num_fruits = int(
                np.sum(np.array(kwargs["fruit_config"]["num_fruit"]))
            )
            self.fruits_index = kwargs["fruit_config"]["fruits_index"]
            self.fruits_reward = kwargs["fruit_config"]["fruits_reward"]
            self.num_fruit_types = len(kwargs["fruit_config"]["fruits_index"])

        super().__init__(
            width=width,
            height=height,
            world=self.world,
            see_through_walls=False,
            agents=agents,
            partial_obs=partial_obs,
            actions_set=self.actions_set,
            render_mode="human",
        )

    # grid generation
    def _gen_grid(self, width: int, height: int):
        # Create a blank grid for this episode
        self.grid = Grid(width, height, self.world)

        # outer wall to stop agents from going off the edge of the world
        self.grid.wall_rect(x=0, y=0, w=self.width, h=self.height)

        # spawn agents
        self._spawn_agents(self.min_agent_levels, self.max_agent_levels)

        # old logic, doesn't match original LBF
        # if self.agent_config is None:
        # randomly spawn
        # self.spawn_agents(self.min_agent_level, self.max_agent_level)
        # else:
        #     # Place the agents based on given initial positions
        #     for agent in self.agents:
        #         self.place_agent(
        #             agent,
        #             agent.init_pos,
        #         )

        # spawn fruit
        if self.max_fruit_level is not None:
            max_fruit_levels = self.max_fruit_level
        else:
            # sum of 3 highest-level agents
            agent_levels = sorted([agent.level for agent in self.agents])
            max_fruit_levels = sum(agent_levels[:3]) * np.ones(self.max_num_fruit)

        self._num_fruit_spawned = self._spawn_fruit(
            self.max_num_fruit,
            min_levels=self.min_fruit_level * np.ones(self.max_num_fruit),
            max_levels=max_fruit_levels,
        )

        # only do if field_map is given in the config file
        if self.field_map is not None:
            # Translate the maze structure into the grid
            for y, row in enumerate(self.field_map):
                for x, cell in enumerate(row):
                    if cell == "#":
                        self.put_obj(Wall(self.world, type="wall", color="grey"), x, y)
                    elif cell == "1" or cell == "2" or cell == "3":
                        temp_fruit_config: FruitTypeDict = {  # type: ignore
                            "capture_reward": float(self.fruits_reward[int(cell) - 1]),
                            "level": int(cell),
                            "color": str(
                                self.world.IDX_TO_COLOR[
                                    self.fruits_index[int(cell) - 1]
                                ]
                            ),
                        }
                        self.put_obj(Fruit(temp_fruit_config, self.world), x, y)
                    else:
                        pass

        # only do if layout_config is given in the config file
        self.waypoint_pos = []
        if self.waypoints is not None:
            # Place Waypoints
            for r in range(self.num_rooms):
                for waypoint in self.waypoints[r]["flag_positions"]:
                    waypoint_obj = Goal(
                        self.world,
                        color="green",
                        reward=5,
                    )
                    self.put_obj(waypoint_obj, waypoint[0], waypoint[1])
                    self.waypoint_pos.append(waypoint_obj.pos)

    def _spawn_agents(self, min_agent_levels: np.ndarray, max_agent_levels: np.ndarray):
        # permute agent levels
        agent_permutation = self.np_random.permutation(self.num_agents)
        min_agent_levels = min_agent_levels[agent_permutation]
        max_agent_levels = max_agent_levels[agent_permutation]

        # Reset the agents. If agent_pos_list is provided, use it to reset the agents
        # agent_pos_list: list[tuple[int, int]] | None = None
        # if options is not None:
        #     agent_pos_list = options.get("agent_pos_list", None)
        # else:
        #     pass
        # self._reset_agents(agent_pos_list=agent_pos_list)

        for agent, min_agent_level, max_agent_level in zip(
            self.agents, min_agent_levels, max_agent_levels
        ):
            attempts = 0

            while attempts < self.spawn_attempts:
                # -1 to avoid including the outer wall in the sample
                pos = (
                    self.np_random.integers(1, self.width - 1),
                    self.np_random.integers(1, self.height - 1),
                )

                if self._valid_agent_cell(self.grid.get(*pos)):
                    level = self.np_random.integers(
                        min_agent_level, max_agent_level + 1
                    )
                    agent.reset(init_pos=pos, level=level)
                    self.place_agent(agent, pos=pos)
                    break

                attempts += 1

    def _spawn_fruit(
        self, max_num_fruit: int, min_levels: np.ndarray, max_levels: np.ndarray
    ) -> int:
        """
        Returns
        -------
        int
            number of fruit spawned in the environment, may be less than max_num_fruit
        """

        fruit_count = 0
        attempts = 0
        min_levels = max_levels if self.force_coop else min_levels

        # permute fruit levels
        fruit_permutation = self.np_random.permutation(max_num_fruit)
        min_levels = min_levels[fruit_permutation]
        max_levels = max_levels[fruit_permutation]

        while fruit_count < max_num_fruit and attempts < 1000:
            attempts += 1
            # -1 to avoid including the outer wall in the sample
            pos = (
                self.np_random.integers(1, self.width - 1),
                self.np_random.integers(1, self.height - 1),
            )

            # check if any fruit in the neighborhood
            grid = self.grid.encode()
            radius_objects = self._get_neighborhood(*pos)[:, :, 0]
            plus_objects = self._get_neighborhood(*pos, radius=2, ignore_diag=True)[
                :, 0
            ]

            if (
                np.any(radius_objects == self.world.OBJECT_TO_IDX["fruit"])
                or np.any(plus_objects == self.world.OBJECT_TO_IDX["fruit"])
                or (grid[*pos, 0] != self.world.OBJECT_TO_IDX["empty"])
            ):
                continue

            self.place_object(
                Fruit(
                    world=self.world,
                    level=self.np_random.integers(
                        min_levels[fruit_count], max_levels[fruit_count] + 1
                    ),
                    color=self.world.IDX_TO_COLOR[3],
                )
            )
            fruit_count += 1

        return fruit_count

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[NDArray[np.int_], dict[str, Any]]:
        # despawn any agents or fruit from the previous episode
        self._reset_gym(seed=seed)

        # reset other params
        # self.step_count: int = 0
        self.collected_fruit: int = 0
        if self.num_rooms is not None:
            self.current_room: int = 0
            self.room_cleared = [False for _ in range(self.num_rooms)]

        # generate new env layout
        self._gen_grid(self.width, self.height)

        obs: NDArray[np.int_] = self.get_obs()
        info: dict[str, Any] = self._get_info(reset=True)

        return obs, info

    # step
    def step(
        self, action: NDArray[np.int_] | np.int_
    ) -> tuple[NDArray[np.int_], float, bool, bool, dict[str, Any]]:
        """
        Take a step in the environment.

        Parameters
        ----------
        action : int
            The action to take.

        Returns
        -------
        observation : NDArray[np.int_]
            The observation of the environment.
        reward : float
            The reward for the action.
        terminated: bool
            Whether the episode has terminated.
        terminated: bool
            Whether the episode has terminated.
        info : dict[str, Any]
            Additional information about the environment.
        """
        terminated: bool = False
        actions: list[int] = np.array(action).flatten().astype(np.int_).tolist()

        for a in self.agents:
            a.reward = 0.0

        moving_agents = defaultdict(list)
        loading_agents = set()

        for agent, action in zip(self.agents, actions):
            next_pos: tuple[int, int] = self._get_next_pos(agent, action)
            next_cell: None | WorldObj = self.grid.get(*next_pos)
            if isinstance(next_cell, Wall):
                pass
            elif action == self.actions.LOAD:
                loading_agents.add(agent)
            else:
                moving_agents[next_pos].append(agent)

        # move agents
        self._move_agents(moving_agents)

        # process the loadings
        self._load_fruit(loading_agents)

        # # Order to apply the actions to the agents
        # order: NDArray[np.int_] = self.np_random.permutation(self.num_agents)

        # for i in order:
        #     agent = self.agents[i]
        #     act: int = actions[i]

        #     if agent.terminated:
        #         continue
        #     else:
        #         next_pos: tuple[int, int] = self._get_next_pos(agent, act)
        #         next_cell: None | WorldObj = self.grid.get(*next_pos)

        #         if isinstance(next_cell, Wall):
        #             continue

        #         elif act == self.actions.LOAD:
        #             for neighbor_pos in agent.neighbor_pos:
        #                 cell = self.grid.get(*neighbor_pos)
        #                 if isinstance(cell, Fruit) and cell.check_capture_condition(
        #                     self.grid
        #                 ):
        #                     # print(
        #                     #     f"Agent {i} captured fruit at {cell.pos} with level {cell.fruit_config['level']}!"
        #                     # )
        #                     self._handle_pickup(i, rewards, neighbor_pos, cell)
        #                 else:
        #                     pass

        #         elif self._valid_agent_cell(next_cell):
        #             # Move agent
        #             self.grid.set(*next_pos, agent)
        #             self.grid.set(*agent.pos, None)
        #             # Change the dir of the agent
        #             if agent.pos != next_pos:
        #                 dir_vec: NDArray[np.int_] = np.array(next_pos) - np.array(
        #                     agent.pos
        #                 )
        #                 agent.dir = agent.vec2dir(dir_vec)
        #             agent.pos = next_pos

        #             if self.waypoint_pos is not None:
        #                 # Check if the agent has reached a waypoint
        #                 if (
        #                     agent.pos in self.waypoint_pos
        #                     and self.room_cleared[self.current_room]
        #                 ):
        #                     # print(f"Agent {i} reached waypoint at {agent.pos} in room {self.current_room}!")
        #                     self._reward(i, rewards, next_cell.reward if next_cell else 0)

        # if self.waypoints is not None:
        #     self._update_waypoints()

        terminated = self._terminated()

        # truncated handled by TimeLimit wrapper
        truncated = False

        # self.step_count += 1
        # print(
        #     f"Step: {self.step_count}, Total Collected Fruit: {self.collected_fruit}, Current Room: {self.current_room}"
        # )

        obs: NDArray[np.int_] = self.get_obs()

        agent_rewards = [a.reward for a in self.agents]
        reward: float = float(np.sum(agent_rewards))

        info = self._get_info()

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )

    def _move_agents(self, moving_agents: dict[list]):
        # if two or more players try to move to the same location they all fail
        for next_pos, agents in moving_agents.items():
            next_cell: None | WorldObj = self.grid.get(*next_pos)

            # make sure no more than one agent will arrive at location

            if len(agents) == 1 and self._valid_agent_cell(next_cell):
                # do movements for non colliding players
                agent = agents[0]
                # Move agent
                self.grid.set(*next_pos, agent)
                self.grid.set(*agent.pos, None)

                agent.pos = next_pos

                if self.waypoint_pos is not None:
                    # Check if the agent has reached a waypoint
                    if (
                        agent.pos in self.waypoint_pos
                        and self.room_cleared[self.current_room]
                    ):
                        # print(f"Agent {i} reached waypoint at {agent.pos} in room {self.current_room}!")
                        self._reward(i, rewards, next_cell.reward if next_cell else 0)

    def _load_fruit(self, loading_agents: set):
        while loading_agents:
            agent = loading_agents.pop()
            for neighbor_pos in agent.neighbor_pos:
                cell = self.grid.get(*neighbor_pos)
                if isinstance(cell, Fruit) and cell.check_capture_condition(self.grid):
                    fruit_level = cell.level
                    fruit_pos = neighbor_pos

                    # get the agents that helped load the fruit
                    adj_agents: list[LBFAgent] = []
                    for fruit_neighbor_pos in cell.neighbor_pos:
                        fruit_neighbor_cell = self.grid.get(*fruit_neighbor_pos)
                        if isinstance(fruit_neighbor_cell, LBFAgent):
                            adj_agents.append(fruit_neighbor_cell)

                        adj_agent_levels = [int(a.level) for a in adj_agents]
                        tot_agent_levels = sum(adj_agent_levels)

                        # failed to load
                        if tot_agent_levels < fruit_level:
                            for a in adj_agents:
                                a.reward -= self.failed_load_penalty
                        else:
                            # the fruit was loaded and each player scores points
                            for a in adj_agents:
                                a.reward += float(a.level * fruit_level)
                                if self._normalize_reward:
                                    a.reward = a.reward / float(
                                        tot_agent_levels * self._num_fruit_spawned
                                    )

                            # remove the fruit from the map
                            cell.pos = np.array([-1, -1])
                            self.grid.set(*fruit_pos, None)
                            self.collected_fruit += 1

                    # remove these agents so they are not checked again
                    loading_agents -= set(adj_agents)

                    # print(
                    #     f"Agents {[agent.index for agent in adj_agents]} with levels {adj_agent_levels} collected fruit at {fruit_pos} with level {fruit_level}"
                    # )

    def _get_next_pos(self, agent, action: int) -> tuple[int, int]:
        self.actions: LBFActions

        match action:
            case self.actions.STAY | self.actions.LOAD:
                next_pos = agent.pos
            case self.actions.LEFT:
                next_pos = agent.west_pos(in_tuple=True)
            case self.actions.RIGHT:
                next_pos = agent.east_pos(in_tuple=True)
            case self.actions.UP:
                next_pos = agent.north_pos(in_tuple=True)
            case self.actions.DOWN:
                next_pos = agent.south_pos(in_tuple=True)
            case _:
                raise ValueError(f"Invalid action: {action}")

        return next_pos

    def _terminated(self) -> bool:
        # Terminate the episode if all rooms have been cleared or max steps reached
        terminated = False

        if self.waypoints is not None:
            terminated = self.current_room == self.num_rooms
        else:
            terminated = self.collected_fruit == self._num_fruit_spawned

        return terminated

    def _reward(
        self, current_agent: int, rewards: NDArray[np.float64], reward: float = 1
    ) -> None:
        """
        Compute the reward to be given upon success
        """
        rewards[current_agent] += reward

    def _get_info(self, reset: bool = False) -> dict:
        # step info
        info = {}

        # TODO follow join1's example here
        # if X happens, info["battle_won"] = True
        # else info["battle_won"] = False

        # if not reset and some other condition
        # info["battle_won"] = True

        # else:
        # info["battle_won"] = False

        return info

    # state
    def get_state(self):
        # return a state of size (n_state_features)
        # use multigrid's basic state for now, come back to this later
        match self.state_type:
            case "original":
                # reproduce the original LBF "state" by concatenating all agent obs into a long vector
                # NOTE: technically not a "state" from an RL theory perspective since
                # there may be info not observed by any agent given their limited obs range
                obs = self.get_obs()
                state = np.concatenate(obs)

            case "multigrid_encode":
                state = self.grid.encode()
                state = state.flatten()

            case _:
                raise NotImplementedError

        return state

    def _get_state_size(self) -> int:
        """standard function to interface with EPyMARL training loop,
        returns the flattened size of the global state."""
        match self.state_type:
            case "original":
                state_size: int = self._get_obs_size() * self.num_agents

            case "multigrid_encode":
                state = self.get_state()
                state_size = math.prod(state.shape)

            case _:
                raise NotImplementedError

        return state_size

    # obs
    def get_obs(self) -> NDArray:
        """get the team's joint observation

        Returns
        -------
        NDArray
            obs of size (num_agents, num_obs_features

        """
        # return an obs of size (num_agents, num_obs_features)
        # use multigrid's basic obs for now, come back to this later
        match self.obs_type:
            case "original":
                # same as _make_gym_obs from original LBF
                obs = self._get_original_obs()

            case _:
                raise NotImplementedError

        return obs

    def _get_original_obs(self) -> NDArray:
        joint_obs = np.vstack([self._get_agent_obs(agent) for agent in self.agents])

        """
        # not using this, low priority to implement
        # if self._grid_observation:
        #     layers = self._make_global_grid_arrays()
        #     agents_bounds = [
        #         self._get_agent_grid_bounds(*player.position) for player in self.players
        #     ]
        #     nobs = tuple(
        #         [
        #             layers[:, start_x:end_x, start_y:end_y]
        #             for start_x, end_x, start_y, end_y in agents_bounds
        #         ]
        #     )
        # else:
        #   nobs = tuple([self._make_obs_array(obs) for obs in observations])
        """

        return joint_obs

    def _get_agent_obs(self, agent: LBFAgent) -> NDArray:
        # get the agent's local obs
        radius_obs = self._get_neighborhood(*agent.pos, radius=self.sight)

        fruit_obs = self._get_object_obs(
            agent_radius_obs=radius_obs,
            obj_type="fruit",
            num_objects=self.max_num_fruit,
        )

        agent_obs = self._get_object_obs(
            agent_radius_obs=radius_obs,
            obj_type="agent",
            num_objects=self.num_agents,
            ego_agent=agent,
        )

        obs = np.concatenate((fruit_obs, agent_obs)).flatten()

        return obs

    def _get_object_obs(
        self,
        agent_radius_obs: NDArray,
        obj_type: Literal["agent", "fruit"],
        num_objects: int,
        ego_agent: Optional[LBFAgent] = None,
    ) -> NDArray:
        obj_obs = np.repeat(self.init_object_obs, repeats=num_objects, axis=0)

        obj_positions = np.vstack(
            np.where(agent_radius_obs[:, :, 0] == self.world.OBJECT_TO_IDX[obj_type])
        ).T

        # ego agent is first in its observations of the agents
        if ego_agent is not None:
            # ego agent's position in its local frame
            (y, x) = self._transform_to_ego_agent_frame(
                center=ego_agent.pos, sight=self.sight, position=ego_agent.pos
            )
            obj_obs[0, :] = np.array([y, x, ego_agent.level])

            # remove ego_agent to avoid double counting
            rows_remove = np.argwhere(
                np.all(obj_positions == np.array([y, x]), axis=1) == True
            )
            obj_positions = np.delete(obj_positions, rows_remove, axis=0)

        for i, (y, x) in enumerate(obj_positions):
            level = agent_radius_obs[y, x, 2]
            obj_obs[i, :] = np.array([y, x, level])

        # remove agent levels from the obs
        if (not self._observe_agent_levels) and (obj_type == "agent"):
            obj_obs = obj_obs[:, 0:2]

        return obj_obs

    def _transform_to_ego_agent_frame(
        self, center: tuple[int, int], sight: int, position: tuple[int, int]
    ):
        return (
            position[0] - center[0] + min(sight, center[0]),
            position[1] - center[1] + min(sight, center[1]),
        )

    def _get_obs_size(self) -> int:
        """standard function to interface with EPyMARL training loop, returns the flattened size of a single agent's observation."""
        match self.obs_type:
            case "original":
                obs_size: int = self.observation_space.shape[1]

            case _:
                raise NotImplementedError

        return obs_size

    def _set_observation_space(self) -> spaces.Space:
        match self.obs_type:
            case "original":
                # get obs space for a single agent
                agent_levels = sorted(self.max_agent_levels)
                max_fruit_level = (
                    self.max_fruit_level
                    if self.max_fruit_level is not None
                    else sum(agent_levels[:3])
                )
                max_pos = (self.height - 1, self.width - 1)

                # fruit obs bounds
                max_fruit_obs_single = np.array([[*max_pos, max_fruit_level]])

                min_fruit_obs = np.repeat(
                    self.init_object_obs, repeats=self.max_num_fruit, axis=0
                ).flatten()
                max_fruit_obs = np.repeat(
                    max_fruit_obs_single,
                    repeats=self.max_num_fruit,
                    axis=0,
                ).flatten()

                # agent obs bounds
                min_agent_obs_single = self.init_object_obs
                max_agent_obs_single = np.array([[*max_pos, max(agent_levels)]])

                if not self._observe_agent_levels:
                    min_agent_obs_single = min_agent_obs_single[:, 0:2]
                    max_agent_obs_single = max_agent_obs_single[:, 0:2]

                min_agent_obs = np.repeat(
                    min_agent_obs_single, repeats=self.num_agents, axis=0
                ).flatten()
                max_agent_obs = np.repeat(
                    max_agent_obs_single,
                    repeats=self.num_agents,
                    axis=0,
                ).flatten()

                # min and max total obs (single agent)
                min_obs_single = np.concat([min_fruit_obs, min_agent_obs]).reshape(
                    1, -1
                )
                max_obs_single = np.concat([max_fruit_obs, max_agent_obs]).reshape(
                    1, -1
                )

                # get joint obs space for the team
                team_obs_space = spaces.Box(
                    low=np.repeat(min_obs_single, repeats=self.num_agents, axis=0),
                    high=np.repeat(max_obs_single, repeats=self.num_agents, axis=0),
                    dtype=np.int_,
                )

                # following original env
                # team_obs_space = spaces.Tuple(
                #     tuple([obs_space] * len(self.agents))
                # )

            case _:
                raise NotImplementedError

        return team_obs_space

    # actions
    def get_avail_actions(self):
        # added this method to interface with PYMARL training loop
        # returns list of agent lists, where each agent's list has binary values representing
        # available actions
        # based on the MAIC paper's implementation of LBF with some minor cleanup
        # https://github.com/mansicer/MAIC/blob/main/src/envs/lbforaging/foraging.py

        return [self.get_avail_agent_actions(agent) for agent in self.agents]

    def get_avail_agent_actions(self, agent: LBFAgent) -> list[int]:
        avail_actions = [0] * len(self.actions)
        agent_actions = [
            action
            for action in self.actions_set
            if self._is_valid_action(agent, action)
        ]

        for action in agent_actions:
            avail_actions[action.value] = 1

        return avail_actions

    def _is_valid_action(self, agent, action):
        valid = False

        if action == self.actions_set.STAY:
            valid = True
        elif action == self.actions_set.LOAD:
            valid = self._adjacent_fruit(agent) > 0
        else:
            # movement actions
            next_pos = self._get_next_pos(agent, action)
            next_cell: None | WorldObj | bool
            try:
                next_cell = self.grid.get(*next_pos)
            except:
                # next cell is not a valid cell to move to
                next_cell = False

            if self._valid_agent_cell(next_cell):
                match action:
                    case self.actions_set.UP:
                        valid = agent.pos[0] > 0

                    case self.actions_set.DOWN:
                        valid = agent.pos[0] < self.height - 1

                    case self.actions_set.LEFT:
                        valid = agent.pos[1] > 0

                    case self.actions_set.RIGHT:
                        valid = agent.pos[1] < self.width - 1

        return valid

    def _set_action_space(self) -> tuple[spaces.Space, int]:
        action_space = spaces.Tuple(
            tuple([spaces.Discrete(len(self.actions))] * len(self.agents))
        )
        ac_dim = len(self.actions)

        return action_space, ac_dim

    # helper methods
    def _get_neighborhood(
        self, row: int, col: int, radius: int = 1, ignore_diag: bool = False
    ) -> NDArray:
        # neighborhood not same thing as adjacent, it's more general
        grid = self.grid.encode()

        if ignore_diag:
            # find objects in a plus-shape centered on the given position
            grids = []

            x_min, x_max = max(row - radius, 0), min(row + radius + 1, self.width)
            grids.append(grid[x_min:x_max, col, :])

            y_min, y_max = max(col - radius, 0), min(col + radius + 1, self.height)
            grids.append(grid[row, y_min:y_max, :])
            neighbor_objects = np.concatenate(grids)

        else:
            x_min, x_max = max(row - radius, 0), min(row + radius + 1, self.width)
            y_min, y_max = max(col - radius, 0), min(col + radius + 1, self.height)

            neighbor_objects = grid[x_min:x_max, y_min:y_max, :]

        return neighbor_objects

    def _adjacent_fruit(self, agent: LBFAgent):
        """
        original logic for adjacent_fruit takes (x, y) as input and tells you the sum around it. Fruit encoded as a 1 in original lbf, so the logic below equivalent to "if any positions around me have a fruit in them, I can take the load action"

        Parameters
        ----------
        agent : LBFAgent

        Returns
        -------
        bool: true if agent is next to a piece of fruit
        """
        fruit_neighbor: list[bool] = [False] * len(agent.neighbor_pos)
        for i, neighbor_pos in enumerate(agent.neighbor_pos):
            if isinstance(self.grid.get(*neighbor_pos), Fruit):
                fruit_neighbor[i] = True

        return any(fruit_neighbor)

    def _update_waypoints(self):
        # if all agents are at waypoints, move to the next room
        if (
            all(agent.pos in self.waypoint_pos for agent in self.agents)
            and self.room_cleared[self.current_room]
        ):
            print(
                f"All agents reached waypoints for room {self.current_room}. Moving to next room..."
            )
            # remove walls and waypoints for the current room
            for waypoint in self.waypoints[self.current_room]["flag_positions"]:
                # self.grid.set(waypoint[0], waypoint[1], None)
                self.waypoint_pos.remove(waypoint)
            for wall_pos in self.waypoints[self.current_room]["wall_positions"]:
                # print(f"Removing wall at {wall_pos}")
                self.grid.set(wall_pos[0], wall_pos[1], None)
            self.current_room += 1

    def _config_lists_to_tuples(self, data: list[dict]) -> list[dict]:
        # convert from list of lists to list of tuples in a list of config dicts
        for d in data:
            for k, v in d.items():
                updated_config = []
                for item in v:
                    if isinstance(item, list):
                        updated_config.append(tuple(item))
                    else:
                        updated_config.append(item)
                d[k] = updated_config

        return data

    def _valid_agent_cell(self, cell: WorldObj | bool | None) -> bool:
        if isinstance(cell, bool):
            return cell
        else:
            return cell is None or cell.can_overlap()


class AgentState(IntEnum):
    MOVING_TO_WAYPOINT = 0
    LOADING_FRUIT = 1
    WAITING_AT_GOAL = 2


class GreedyPredatorPolicy:
    def __init__(
        self,
        fruit_list,
        goal_list,
        random_prob: float = 0.1,
        action_set: Type[IntEnum] = LBFActions,
        dir_to_vec: list[NDArray[np.int_]] = DIR_TO_VEC,
        random_generator: np.random.Generator | None = None,
    ):
        self.fruit_list = fruit_list
        self.fruit_idx: int = 0
        self.goal_list = goal_list
        self.goal_idx: int = 0
        self.random_prob: float = random_prob
        self.action_set: Type[IntEnum] = action_set
        self.dir_to_vec: list[NDArray[np.int_]] = dir_to_vec
        self.random_generator: np.random.Generator = (
            random_generator
            if random_generator is not None
            else np.random.default_rng()
        )
        self.state = AgentState.MOVING_TO_WAYPOINT

    def vec2dir(self, vec: NDArray[np.int_]) -> int:
        """
        Convert a vector to a direction.

        Parameters
        ----------
        vec : NDArray[np.int_]
            The vector to convert.

        Returns
        -------
        dir : int
            The direction corresponding to the vector.
        """
        for direc, dir_vec in enumerate(self.dir_to_vec):
            if np.array_equal(dir_vec, vec):
                return direc
        raise ValueError(f"Invalid vector: {vec}")

    def act(self, current_pos, current_room, grid) -> int:
        # If we have entered a new room, update the goal and reset fruit index
        if self.goal_idx != current_room:
            self.goal_idx = current_room
            self.fruit_idx = 0

        # Finished all fruits in current room
        if self.fruit_idx >= len(self.fruit_list[self.goal_idx]):
            target_pos = self.goal_list[self.goal_idx]
            self.state = AgentState.MOVING_TO_WAYPOINT
        else:  # Still fruits to pick up in current room
            target_pos = self.fruit_list[self.goal_idx][self.fruit_idx]

        if self.state == AgentState.LOADING_FRUIT:
            # Check if the fruit we were loading still exists
            fruit_nearby = self._fruit_adjacent(current_pos, grid)
            if fruit_nearby:
                # Keep attempting LOAD until fruit disappears
                return self.action_set.LOAD
            else:
                # Fruit collected
                self.fruit_idx += 1
                if self.fruit_idx >= len(self.fruit_list[self.goal_idx]):
                    target_pos = self.goal_list[self.goal_idx]
                else:
                    target_pos = self.fruit_list[self.goal_idx][self.fruit_idx]
                self.state = AgentState.MOVING_TO_WAYPOINT

        if current_pos == target_pos:
            if target_pos == self.goal_list[self.goal_idx]:
                self.state = AgentState.WAITING_AT_GOAL
                return self.action_set.STAY
            elif self._fruit_adjacent(current_pos, grid):
                self.state = AgentState.LOADING_FRUIT
                return self.action_set.LOAD
            else:
                self.fruit_idx += 1
                return self.action_set.STAY

        act_randomly: bool = (
            False
            if target_pos is not None
            and self.random_generator.random() >= self.random_prob
            else True
        )

        action: int

        match act_randomly:
            case True:
                action = self.random_generator.integers(0, len(self.action_set))
            case False:
                path: list[tuple[int, int]] = a_star(current_pos, target_pos, grid)
                next_pos: tuple[int, int] = path[1] if len(path) > 1 else current_pos
                dir_vec: NDArray[np.int_] = np.array(next_pos) - np.array(current_pos)
                action = self.vec2dir(dir_vec)

        return action

    def _fruit_adjacent(self, pos: tuple[int, int], grid) -> bool:
        """Return True if any fruit is orthogonally adjacent to pos."""
        r, c = pos
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if grid.get(r + dr, c + dc) and isinstance(grid.get(r + dr, c + dc), Fruit):
                return True
        return False

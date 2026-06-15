from os.path import join, dirname
from warnings import warn
from enum import IntEnum
from collections import defaultdict
from typing import Any, Type, Literal, Optional
from math import prod
from ast import literal_eval
import yaml
from cv2 import putText

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from gymnasium import spaces

from gym_multigrid.core.agent import Agent, LBFActions
from gym_multigrid.core.constants import DIR_TO_VEC, TILE_PIXELS
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.object import Goal, Wall, WorldObj
from gym_multigrid.core.world import LBFWorld, World
from gym_multigrid.utils.rendering import fill_coords, point_in_circle, point_in_rect, FontConfig
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
        init_grid: Optional[Grid] = None,
        level: Optional[int] = None,
        tile_size: int = TILE_PIXELS,
    ):

        self.init_pos = init_pos
        self.init_grid = init_grid
        self.level = level
        self.tile_size = tile_size

        self.reward: float = 0.0
        self.neighbor_pos_offsets: NDArray[np.int_] = np.array(
            [[-1, 0], [1, 0], [0, -1], [0, 1]]
        )
        self.neighbor_pos: NDArray[np.int_] = np.zeros((4, 2), dtype=np.int_)

        # an agent may have multiple current goal states
        self.room_goals: dict[int, NDArray] = {}

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

    def add_goal_pos(self, pos: Position, room_idx: int):
        pos = np.array([pos])

        if room_idx not in self.room_goals:
            self.room_goals[room_idx] = pos

        elif not np.any(np.all(pos == self.room_goals[room_idx], axis=1)):
            self.room_goals[room_idx] = np.vstack((self.room_goals[room_idx], pos))

    def in_goal_set(self, current_room: int, pos: Position = None) -> bool:
        if pos is None:
            pos = self.pos

        if np.any(np.all(pos == self.room_goals[current_room], axis=1)):
            return True
        else:
            return False

    def reset(self, level: int, init_pos: tuple[int, int]) -> None:
        super().reset()
        if self.pos is not None:
            self.neighbor_pos = self.pos + self.neighbor_pos_offsets
        else:
            self.neighbor_pos = np.zeros((4, 2), dtype=np.int_)

        self.level = level
        self.init_pos = init_pos
        self.reward: float = 0.0

    def encode(self, current_agent: bool = False) -> tuple[int]:
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
            img,
            point_in_rect(0.15, 0.85, 0.15, 0.85),
            color=self.world.COLORS[self.color],
            bg_color=self.bg_color,
        )

        self._render_object_info(img, level=True, index=True)


class Fruit(WorldObj):
    def __init__(
        self,
        world: World,
        color: str,
        level: int,
        tile_size: int = TILE_PIXELS,
    ):
        self.level = level
        self.tile_size = tile_size

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

    def check_load_condition(self, grid: Grid, loading_agents: list[LBFAgent]) -> bool:
        """
        Check if the fruit was collected by its surrounding agents.

        Parameters
        ----------
        grid : Grid
            The grid of the environment.
        loading_agents : list[LBFAgent]
            list of all agents taking the load action in the env

        Returns
        -------
        curr_fruit_loading_agents: list[LBFAgent]
            agents that attempted to load this fruit
        load_success : bool
            True if the fruit was loaded by the agents in curr_fruit_loading_agents
        tot_agent_levels: int
            total level of the agents that contributed to loading this fruit
        """
        tot_agent_levels: int = 0
        curr_fruit_loading_agents: list[LBFAgent] = []

        for neighbor_pos in self.neighbor_pos:
            cell = grid.get(*neighbor_pos)
            if isinstance(cell, Agent) and cell in loading_agents:
                tot_agent_levels += cell.level
                curr_fruit_loading_agents.append(cell)
            else:
                pass

        load_success: bool = tot_agent_levels >= self.level

        return curr_fruit_loading_agents, load_success, tot_agent_levels

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), self.world.COLORS[self.color])
        self._render_object_info(img, level=True)

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


class RewardConfig:
    def __init__(
        self,
        num_agents: int,
        normalize_fruit_reward: bool = True,
        failed_load_penalty: float = 0.0,
        agent_reach_goal_mult: float = 0.5,
        agent_leave_goal_mult: float = -0.7,
        all_agents_reach_goal_proportion_per_agent: float = 2.0,
        # move_mult: float = 0.0,
        # max_dense_reach_goal_proportion: float,
    ) -> None:
        """
        all signs for rewarding events assume to be handled in their definition since we only use the "+=" operator in the code below for simplicity and easier design
        values are <=0 for "penalties" and >= 0 for "rewards"

        Parameters
        ----------
        num_agents : int
        normalize_fruit_reward : bool, optional
        failed_load_penalty : float, optional
            value <= 0, by default 0.0
        agent_reach_goal_mult : float, optional
            value >= 0 for "rewards", by default 0.5
        agent_leave_goal_mult : float, optional
            <= 0, by default -0.7
        all_agents_reach_goal_proportion_per_agent : float, optional
            _description_, by default 2.0
        """
        # does NOT consider fruit loading rewards since those are based on fruit levels

        # this setup worked well for 3 agents
        # did not work for 6
        # the gap between reach and leave may not be large enough and may get blown out
        # movement_reward=0.0,
        # agent_reach_goal_reward=0.9,
        # agent_leave_goal_reward=-1.0,
        # all_agents_at_goal_reward=10.0,

        self.normalize_fruit_reward = normalize_fruit_reward
        self.failed_load_penalty: float = failed_load_penalty

        # overall scaling for all rewards in the env
        # use to prevent rewards from getting to large
        # and producing too-large gradients for learning
        self.base_goal_reward = 1.0

        # individual agent rewards, do not multiply by n_agents

        # reach goal serves as the base reward that all others are derived from
        # using reward multipliers. This helps design a reward function that can work for
        # different team sizes
        self.agent_reach_goal_reward: float = (
            self.base_goal_reward * agent_reach_goal_mult
        )

        self.agent_leave_goal_reward: float = (
            self.base_goal_reward * agent_leave_goal_mult
        )

        # reward for entire team, needs to be multiplied by number of agents
        # to work for different team sizes
        self.all_agents_at_goal_reward: float = (
            self.agent_reach_goal_reward
            * all_agents_reach_goal_proportion_per_agent
            * num_agents
        )

        # self.movement_reward: float = self.base_reward * move_mult

        # max_dense_reward needs to be set small enough so agents can't just
        # wander around to get large total reward. They would do that if the
        # discounted summed reward from the "dense" component of R(s, a)
        # is larger than the final reward they get from the team-level goal.
        # It also needs to be small enough so during learning, they
        # have an incentive get the "reach individual goal" reward.
        # self.max_dense_reward: float = (
        #     self.agent_reach_goal_reward * max_dense_reach_goal_proportion
        # )
        # self.max_dense_reward_per_agent: float = self.max_dense_reward / num_agents


class LBFGameEnv(MultiGridEnv):
    """
    Environment in which the agents have to collect fruit. Extends original LBF by supporting multiple rooms and a hierarchical representation of "tasks" in the environment. Also includes comms allocation decisions in the hierarchical version.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self,
        map_name: Optional[str] = None,
        width: Optional[int] = 10,
        height: Optional[int] = 10,
        n_agents: int = 4,
        sight: int = 2,
        min_agent_level: int = 1,
        max_agent_level: int = 2,
        max_num_fruit: int = 2,
        max_num_fruit_per_room: Optional[int] = None,
        min_fruit_level: int = 1,
        max_fruit_level: Optional[int] = None,
        force_coop: bool = False,
        observe_agent_levels: bool = True,
        state_type: Literal["original", "multigrid_flattened"] = "original",
        obs_type: Literal["original", "multigrid_flattened"] = "original",
        observe_other_agents: bool = True,
        highlight_visible_cells: bool = False,
        reward_config: dict[str, float | bool] = {
            "agent_reach_goal_mult": 0.5,
            "agent_leave_goal_mult": -0.7,
            "all_agents_reach_goal_proportion_per_agent": 2.0,
            "failed_load_penalty": 0.0,
            "normalize_fruit_reward": True,
        },
        # team_bandwidth_allocation_action: bool = False,
        # num_comms_values: Optional[int] = 4,
    ):
        """
        Initialize the LBFGameEnv.

        Parameters
        ----------
        map_name:
            name of the map to load (yaml file)
            if None, just 1 room w/ some width and height

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
        use_mdp: bool, whether to use the project MDP or not
        state_type: Literal["original", "multigrid_flattened"] = "original"
            format for the state, orignal breaks when using Goal objects since they were not in the original LBF env
        obs_type: Literal["original", "multigrid"] = "original"
            format for the observation, orignal breaks when using Goal objects since they were not in the original LBF env
        """
        self.num_agents = n_agents
        self.reward_config = RewardConfig(num_agents=self.num_agents, **reward_config)

        # multi-room support
        self.field_map: pd.DataFrame | None = None
        self.agent_spawn_room = 0
        self.room_coords: dict[int, tuple]
        self.num_rooms: int
        self.room_has_goals: dict[int, bool] = {0: False}

        if map is not None:
            if width is not None or height is not None:
                warn("(height, width) and field map provided, using field map size.")

            self.field_map = self._load_field_map(map_name)
            height, width = self.field_map.shape

        else:
            # add 2 b/c of the outer wall that automatically spawns
            # when no map is specified
            self.num_rooms = 1
            width = width + 2
            height = height + 2

            self.room_coords = {
                0: {
                    "x_limits": (0, width),
                    "y_limits": (0, height),
                }
            }

        # whether or not to include an action that represents the allocation of bandwidth to a team
        # self.team_bandwidth_allocation_action: bool = team_bandwidth_allocation_action
        # self.num_comms_values: Optional[int] = num_comms_values

        # hierarchical model of environment project the tasks that comprise it
        self.max_num_fruit_per_room = max_num_fruit_per_room

        # initialize per-room fruit counters
        self.num_fruit_per_room: dict[int, int] = defaultdict(int)

        # determine max number of fruit to spawn overall. If a per-room limit
        # is provided, use it to cap the total possible fruits across all rooms.
        if max_num_fruit_per_room is not None:
            # ensure integer multiplication
            self.max_num_fruit = int(max_num_fruit_per_room) * int(self.num_rooms)
        else:
            self.max_num_fruit = max_num_fruit

        self._num_fruit_spawned: int = 0

        self.state_type = state_type

        # obs options for this specific env
        self.env_obs_type = obs_type
        match self.env_obs_type:
            # in both cases, sight is like the "radius" of the square obs centered on the agent
            # this is here b/c multigrid and the original LBF obs handle it a little differently
            case "original":
                self.sight = sight
            case "multigrid_flattened":
                self.sight = 2 * sight + 1

        # if observe_other_agents=False, agents cannot see the other agents and those cells replaced with empty spaces
        self.observe_other_agents = observe_other_agents
        self.observe_agent_levels = observe_agent_levels

        self.force_coop = force_coop
        self.min_agent_levels = np.array([min_agent_level] * self.num_agents)
        self.max_agent_levels = np.array([max_agent_level] * self.num_agents)

        self.min_fruit_level = min_fruit_level
        self.max_fruit_level = max_fruit_level
        self.spawn_attempts: int = 1000

        self.world = LBFWorld
        self.actions_set = LBFActions

        # initial encoding for objects in the observation
        self.init_object_obs = np.array([-1, -1, 0]).reshape(1, -1)

        # init agents
        agents = []

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

        # objects that disappear when each room is cleared
        self.room_despawn_objects: dict[int, list]

        # fruit tracking
        self.num_fruit_per_room: dict[int, int]
        self.num_fruit_collected_per_room: dict[int, int]

        super().__init__(
            width=width,
            height=height,
            world=self.world,
            see_through_walls=False,
            agents=agents,
            actions_set=self.actions_set,
            render_mode="rgb_array",
            obs_type="symmetrical",
            agent_view_size=self.sight,
            highlight_visible_cells=highlight_visible_cells,
        )

    # grid generation
    def _load_field_map(self, map_name: str) -> pd.DataFrame:
        # read the room layout yaml file to compose the rooms into a cohesive env
        map_dir = join(dirname(__file__), "maps", "lbf", map_name)
        config_path = join(map_dir, "config.yaml")
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # grab all the rooms arranged in a grid
        rooms = defaultdict(list)
        self.room_coords = {}
        x_min, y_min = 0, 0
        room_idx = 0
        for j, row in enumerate(config["room_layout"]):
            for i, room_config in enumerate(row):
                room_load_path = join(map_dir, f"{room_config}.csv")
                room = pd.read_csv(room_load_path, header=None).astype(object)
                rooms[j].append(room)

                x_min, x_max = x_min, x_min + room.shape[1]
                y_min, y_max = y_min, y_min + room.shape[0]

                self.room_coords[room_idx] = {
                    "x_limits": (x_min, x_max),
                    "y_limits": (y_min, y_max),
                }
                self.room_has_goals[room_idx] = False

                x_min = x_max
                room_idx += 1

            y_min = y_max

        # concat all the rooms into a single env
        rows = [pd.concat(rooms[i], axis=1, ignore_index=True) for i in rooms]
        field_map = pd.concat(rows, axis=0, ignore_index=True)
        self.num_rooms = room_idx

        # astype(object) allows literal_eval to convert strings to tuples where needed
        for y, row in field_map.iterrows():
            for x in row.index:
                cell = field_map.loc[y, x]
                if pd.isnull(cell):
                    continue
                elif isinstance(cell, str) and len(cell) > 1:
                    # insert quotes so ast sees the object encoding as a valid string
                    cell = f'{cell[0: 1]}"{cell[1: 2]}"{cell[2:]}'
                    # convert to tuple data types from strings
                    field_map.at[y, x] = literal_eval(cell)
                else:
                    pass

        return field_map

    def _gen_grid(self, width: int, height: int, start_room: int = 0):
        # Create a blank grid for this episode
        self.grid = Grid(width, height, self.world)

        # when a room is "completed", all objects in room_despawn_objects[room_idx] are removed
        self.room_despawn_objects: dict[int, list] = defaultdict(list)

        # place objects from field_map
        if self.field_map is not None:
            self._parse_field_map(obj_place=["w", "g", "d"])
        else:
            # add outer wall to stop agents from going off the edge of the env
            self.grid.wall_rect(x=0, y=0, w=self.width, h=self.height)

        # objects spawned before init_grid is initialized will respawn after an agent steps on them and leaves that cell
        # need separate logic to modify self.init_grid to despawn those objects if desired
        self.init_grid: Grid = self.grid.copy()

        # spawn agents
        self._spawn_agents(self.min_agent_levels, self.max_agent_levels)

        # spawn fruit
        if self.max_fruit_level is not None:
            max_fruit_level = self.max_fruit_level * np.ones(self.max_num_fruit)
        else:
            # sum of 3 highest-level agents
            agent_levels = sorted([agent.level for agent in self.agents])
            max_fruit_level = sum(agent_levels[:3])

        self._num_fruit_spawned = self._spawn_fruit(
            min_levels=self.min_fruit_level * np.ones(self.max_num_fruit),
            max_levels=max_fruit_level * np.ones(self.max_num_fruit),
        )

        # If a start_room was provided prior to grid generation, apply adjustments
        # via helper that encapsulates the logic for starting mid-episode.
        if start_room > 0:
            self._apply_start_room_adjustments(start_room)

    def _parse_field_map(self, obj_place: Optional[list] = None) -> dict:
        """
        obj_place: list of object types to place
        """
        num_spawned_objects = defaultdict(int)

        for y, row in self.field_map.iterrows():
            for x in row.index:
                cell = row[x]
                room_idx = self._get_object_room((x, y))

                if pd.isnull(cell):
                    # empty cells
                    continue

                if isinstance(cell, tuple):
                    obj_type, *obj_args = cell

                    # spawn goals, doors, and walls first so they're in init_grid
                    # do agents and fruit after init_grid is defined
                    if obj_type in obj_place:
                        match obj_type.lower():
                            case "a":
                                # place agents
                                level, agent_idx = obj_args[0], obj_args[1]
                                for agent in self.agents:
                                    if agent.index == agent_idx:
                                        agent.reset(init_pos=(x, y), level=level)
                                        self.place_agent(
                                            agent, pos=(x, y), init_grid=self.init_grid
                                        )
                                        num_spawned_objects["a"] += 1
                                        break

                            case "f":
                                # place fruit
                                level = obj_args[0]

                                obj = Fruit(
                                    world=self.world,
                                    level=level,
                                    color=self.world.IDX_TO_COLOR[3],
                                )
                                self.place_object(obj, pos=(x, y))
                                self.num_fruit_per_room[room_idx] += 1
                                num_spawned_objects["f"] += 1

                            case "g":
                                # place goals + assign to agents
                                assigned_agent_idx = obj_args[0]
                                obj = Goal(self.world, color="yellow")
                                self.place_object(obj, pos=(x, y))
                                for agent in self.agents:
                                    if agent.index == assigned_agent_idx:
                                        agent.add_goal_pos((x, y), room_idx)
                                self.room_has_goals[room_idx] = True
                                self.room_despawn_objects[room_idx].append(obj)

                elif isinstance(cell, str):
                    obj_type = cell
                    match obj_type.lower():
                        case "g":
                            obj = Goal(self.world, color="yellow")
                            self.place_object(obj, pos=(x, y))

                            # each goal assigned to all agents
                            for agent in self.agents:
                                agent.add_goal_pos((x, y), room_idx)
                            self.room_has_goals[room_idx] = True
                            self.room_despawn_objects[room_idx].append(obj)

                        case "w":
                            obj = Wall(self.world, type="wall", color="grey")
                            self.place_object(obj, pos=(x, y))

                        case "d":
                            obj = Wall(self.world, type="wall", color="grey")
                            self.place_object(obj, pos=(x, y))
                            self.room_despawn_objects[room_idx].append(obj)

                else:
                    raise NotImplementedError("invalid object in field map config file")

        return num_spawned_objects

    def _apply_start_room_adjustments(self, start_room: int) -> None:
        """Apply adjustments to the grid and agents so the environment appears
        as if earlier rooms were already completed.
        """
        # Despawn fruits and update counters for previous rooms
        self._despawn_previous_room_fruits_and_objects(start_room)

        # Place agents on the goals/waypoints of the most recent completed room
        self._place_agents_for_start_room(start_room)

    def _despawn_previous_room_fruits_and_objects(self, start_room: int) -> None:
        grid_enc = self.grid.encode()
        fruit_idx = self.world.OBJECT_TO_IDX["fruit"]
        for i in range(grid_enc.shape[0]):
            for j in range(grid_enc.shape[1]):
                if grid_enc[i, j, 0] == fruit_idx:
                    room_idx = self._get_object_room((i, j))
                    if room_idx < start_room:
                        obj = self.grid.get(i, j)
                        if obj is not None:
                            self.despawn_object(obj)
                            if self.num_fruit_per_room.get(room_idx, 0) > 0:
                                self.num_fruit_per_room[room_idx] -= 1
                            self.num_fruit_collected_per_room[room_idx] = (
                                self.num_fruit_per_room.get(room_idx, 0)
                            )
                            self.all_room_fruit_collected[room_idx] = True

        # Also despawn room-specific objects (doors, waypoints)
        for room_idx in range(start_room):
            for obj in list(self.room_despawn_objects.get(room_idx, [])):
                try:
                    self.despawn_object(obj)
                except Exception:
                    pass

    def _place_agents_for_start_room(self, start_room: int) -> None:
        # Choose most-recent completed room with goals
        completed_rooms = [
            r
            for r in range(self.num_rooms)
            if (r < start_room and self.room_has_goals.get(r, False))
        ]
        if len(completed_rooms) > 0:
            prev_room = max(completed_rooms)
        else:
            prev_room = start_room - 1

        for agent in self.agents:
            if prev_room in agent.room_goals and len(agent.room_goals[prev_room]) > 0:
                goal_pos = tuple(agent.room_goals[prev_room][0])
                try:
                    if agent.pos is not None:
                        self.despawn_object(agent)
                except Exception:
                    pass

                agent.reset(init_pos=goal_pos, level=agent.level)
                self.place_agent(agent, pos=goal_pos, init_grid=self.init_grid)

    def _get_object_room(self, pos: Position) -> int:
        x, y = pos
        for room_idx, room_coords in self.room_coords.items():
            x_min, x_max = room_coords["x_limits"]
            y_min, y_max = room_coords["y_limits"]

            if (x_min <= x < x_max) and (y_min <= y < y_max):
                return room_idx

    def _spawn_agents(
        self,
        min_agent_levels: np.ndarray,
        max_agent_levels: np.ndarray,
    ):

        num_spawned_agents = 0

        if self.field_map is not None:
            # parse field map to spawn any agents defined there
            num_spawned_agents = self._parse_field_map(obj_place=["a"])["a"]

        if num_spawned_agents < self.num_agents:
            # permute agent levels
            agent_permutation = self.np_random.permutation(self.num_agents)
            min_agent_levels = min_agent_levels[agent_permutation]
            max_agent_levels = max_agent_levels[agent_permutation]

            for agent, min_agent_level, max_agent_level in zip(
                self.agents, min_agent_levels, max_agent_levels
            ):
                attempts = 0
                while attempts < self.spawn_attempts:
                    # make sure the agents spawn in the first room
                    x_min, x_max = self.room_coords[self.agent_spawn_room]["x_limits"]
                    y_min, y_max = self.room_coords[self.agent_spawn_room]["y_limits"]

                    pos = (
                        self.np_random.integers(x_min, x_max),
                        self.np_random.integers(y_min, y_max),
                    )

                    if self._valid_agent_cell(self.grid.get(*pos)):
                        level = self.np_random.integers(
                            min_agent_level, max_agent_level + 1
                        )
                        agent.reset(init_pos=pos, level=level)
                        self.place_agent(agent, pos=pos, init_grid=self.init_grid)
                        break

                    attempts += 1

    def _spawn_fruit(
        self,
        min_levels: np.ndarray,
        max_levels: np.ndarray,
    ) -> int:
        """
        Returns
        -------
        int
            number of fruit spawned in the environment, may be less than max_num_fruit
        """
        num_spawned_fruit = 0

        if self.field_map is not None:
            # parse field map to spawn any agents defined there
            num_manually_placed_fruit = self._parse_field_map(obj_place=["f"])["f"]

            # if any manually placed fruit, skip rest of random fruit placement logic
            if num_manually_placed_fruit > 0:
                num_spawned_fruit = num_manually_placed_fruit
                self.max_num_fruit = num_manually_placed_fruit
                return num_spawned_fruit

        if num_spawned_fruit < self.max_num_fruit:
            attempts = 0
            min_levels = max_levels if self.force_coop else min_levels

            # permute fruit levels
            fruit_permutation = self.np_random.permutation(self.max_num_fruit)
            min_levels = min_levels[fruit_permutation]
            max_levels = max_levels[fruit_permutation]

            while (
                num_spawned_fruit < self.max_num_fruit
                and attempts < self.spawn_attempts
            ):
                attempts += 1

                # offset by 2 here to avoid avoid spawning a high-level fruit along the wall or in the corner
                # where there may not be enough available sides for the agents to surround and load it
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

                # ensure the per-room fruit limit is not exceeded (if specified)
                room_idx = self._get_object_room(pos)
                if (self.max_num_fruit_per_room is not None) and (
                    self.num_fruit_per_room.get(room_idx, 0)
                    >= self.max_num_fruit_per_room
                ):
                    continue

                # fruit cannot spawn if:
                # next to a wall (helps prevent generation of un-solvable tasks, e.g. level 4 fruit w/ 2 sides blocked by a wall and only level 1 agents available)
                # next to or near another fruit (helps prevent generation of un-solvable tasks, helps space out the fruit)
                # on a space another object already occupies (prevent spawning on goals, walls, agents, etc.)
                if (
                    np.any(radius_objects == self.world.OBJECT_TO_IDX["wall"])
                    or np.any(radius_objects == self.world.OBJECT_TO_IDX["fruit"])
                    or np.any(plus_objects == self.world.OBJECT_TO_IDX["fruit"])
                    or (grid[*pos, 0] != self.world.OBJECT_TO_IDX["empty"])
                ):
                    continue

                self.place_object(
                    Fruit(
                        world=self.world,
                        level=self.np_random.integers(
                            min_levels[num_spawned_fruit],
                            max_levels[num_spawned_fruit] + 1,
                        ),
                        color=self.world.IDX_TO_COLOR[3],
                    ),
                    pos=pos,
                )
                # update room counters
                self.num_fruit_per_room[room_idx] += 1
                num_spawned_fruit += 1

            return num_spawned_fruit

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[NDArray[np.int_], dict[str, Any]]:
        # despawn any agents or fruit from the previous episode
        self._reset_gym(seed=seed)

        # used to render actions
        self._pre_step_actions = [None] * self.num_agents
        self._t_render = None

        # reset fruit tracking
        self.num_fruit_per_room: dict[int, int] = defaultdict(int)
        self.num_fruit_collected_per_room = {
            room_idx: 0 for room_idx in range(self.num_rooms)
        }
        self.all_room_fruit_collected = [False] * self.num_rooms

        # current room / task
        start_room = 0
        if options is not None and "start_room" in options:
            start_room = options["start_room"]

        self.current_room: int = start_room
        for room_idx in range(start_room):
            self.all_room_fruit_collected[room_idx] = True

        # generate new env layout
        self._gen_grid(self.width, self.height, start_room=start_room)

        obs: NDArray[np.int_] = self.obs
        info: dict[str, Any] = self._get_info()

        return obs, info

    # step
    def step(
        self, action: NDArray[np.int_] | np.int_
    ) -> tuple[NDArray[np.int_], float, bool, bool, dict[str, Any]]:
        """
        Take a step in the environment.

        Parameters
        ----------
        action :
            The action to take.

        Returns
        -------
        observation : NDArray[np.int_]
            The observation of the environment.
        reward : float
            The reward for the action.
        terminated: bool
            Whether the episode has terminated.
        truncated: bool
            Whether the episode has been truncated (out of time).
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

        # update waypoints
        room_completed = self._update_room()

        # check if entire project is complete
        terminated = self._terminated()

        # truncated handled by TimeLimit wrapper
        truncated = False

        obs: NDArray[np.int_] = self.obs
        agent_rewards = [a.reward for a in self.agents]
        reward: float = float(np.sum(agent_rewards))
        info = self._get_info(terminated=terminated, room_completed=room_completed)

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )

    def _move_agents(self, moving_agents: dict[Position, list]):
        # if two or more players try to move to the same location they all fail
        for next_pos, agents in moving_agents.items():
            next_cell: None | WorldObj = self.grid.get(*next_pos)

            # make sure no more than one agent will arrive at location
            if len(agents) == 1 and self._valid_agent_cell(next_cell):
                # do movements for non colliding players
                agent = agents[0]

                # run goal logic if there are goals in the current room
                self._goal_reward_logic(agent, next_pos)

                # Move agent
                agent.move(next_pos=next_pos, grid=self.grid, init_grid=self.init_grid)

    def _goal_reward_logic(self, agent, next_pos):
        # only enable goal rewards + penalties if all fruit has been collected in the room
        if (
            self.room_has_goals[self.current_room]
            and self.all_room_fruit_collected[self.current_room]
        ):
            # if not at goal and reach goal, get a reward
            if (not agent.in_goal_set(self.current_room)) and agent.in_goal_set(
                self.current_room, pos=next_pos
            ):
                agent.reward += self.reward_config.agent_reach_goal_reward

            # if agent at goal and next pos is not a goal, get penalty
            if agent.in_goal_set(self.current_room) and not agent.in_goal_set(
                self.current_room, pos=next_pos
            ):
                agent.reward += self.reward_config.agent_leave_goal_reward

    def _update_room(self) -> bool:
        room_completed = False

        # if there are goals in the room, reaching goals after collecting all fruit completes this room
        if self.room_has_goals[self.current_room]:
            reached_room_goal: list[bool] = [
                agent.in_goal_set(self.current_room) for agent in self.agents
            ]

            if (
                all(reached_room_goal)
                and self.all_room_fruit_collected[self.current_room]
            ):
                # print(
                #     f"All agents reached goals for room {self.current_room}. Moving to next room."
                # )
                for obj in self.room_despawn_objects[self.current_room]:
                    self.despawn_object(obj)

                # move on to the next room
                self.current_room += 1
                room_completed = True

        # otherwise collecting all fruit completes this room
        elif self.all_room_fruit_collected[self.current_room]:
            for obj in self.room_despawn_objects[self.current_room]:
                self.despawn_object(obj)

            # move on to the next room
            self.current_room += 1
            room_completed = True

        return room_completed

    def _load_fruit(self, loading_agents: set):
        loading_agents_tmp = loading_agents.copy()

        while loading_agents_tmp:
            agent = loading_agents_tmp.pop()
            for neighbor_pos in agent.neighbor_pos:
                cell = self.grid.get(*neighbor_pos)
                if isinstance(cell, Fruit):
                    curr_fruit_loading_agents, load_success, tot_agent_levels = (
                        cell.check_load_condition(self.grid, list(loading_agents))
                    )

                    if load_success:
                        fruit_level = cell.level
                        # the fruit was loaded and each player that helped load scores points
                        for a in curr_fruit_loading_agents:
                            a.reward += float(a.level * fruit_level)

                            if self.reward_config.normalize_fruit_reward:
                                a.reward = a.reward / float(
                                    tot_agent_levels * self._num_fruit_spawned
                                )

                        # despawn the fruit from the env
                        self.grid.set(*cell.pos, None)
                        cell.pos = np.array([-1, -1])
                        self.num_fruit_collected_per_room[self.current_room] += 1

                    else:
                        for a in curr_fruit_loading_agents:
                            a.reward += self.reward_config.failed_load_penalty

                    # remove these agents so they are not checked again
                    loading_agents_tmp -= set(curr_fruit_loading_agents)

                    # print(
                    #     f"Agents {[agent.index for agent in adj_agents]} with levels {adj_agent_levels} collected fruit at {fruit_pos} with level {fruit_level}"
                    # )

        if (
            self.num_fruit_collected_per_room[self.current_room]
            == self.num_fruit_per_room[self.current_room]
        ):
            self.all_room_fruit_collected[self.current_room] = True

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

        # convert from np ints to ints if needed
        if isinstance(next_pos[0], np.int_):
            next_pos = tuple(map(int, next_pos))
        return next_pos

    def _terminated(self) -> bool:
        # Terminate the episode if all rooms have been cleared or max steps reached
        terminated = False

        # project is completed when you reach the end of the final room
        # TODO needs to be updated to handle non-sequential rooms
        terminated = self.current_room == self.num_rooms

        return terminated

    def _get_info(self, terminated: bool = False, room_completed: bool = False) -> dict:
        # step info
        info = {}

        # Agents only succeed at the full "project" if terminated = True
        # info["project_completed"] = terminated

        info["task_completed"] = room_completed

        return info

    # state
    @property
    def state(self):
        # define as a class attribute so you can access the state using env.state
        # no matter how many layers of wrappers are around this env
        # return a state of size (n_state_features)

        # use multigrid's basic state for now, come back to this later
        match self.state_type:
            case "original":
                # reproduce the original LBF "state" by concatenating all agent obs into a long vector
                # NOTE: technically not a "state" from an RL theory perspective since
                # there may be info not observed by any agent given their limited obs range
                obs = self.obs
                state = np.concatenate(obs)

            case "multigrid_flattened":
                # NOTE: compared to original, currently does NOT have the coordinates of other objects in the ego agent's frame, so that could reduce training performance
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

            case "multigrid_flattened":
                state_size = prod(self.state.shape)

            case _:
                raise NotImplementedError

        return state_size

    # obs
    @property
    def obs(self) -> NDArray:
        """get the team's joint observation

        Returns
        -------
        NDArray
            obs of size (num_agents, num_obs_features

        """
        # return an obs of size (num_agents, num_obs_features)
        # use multigrid's basic obs for now, come back to this later
        match self.env_obs_type:
            case "original":
                # same as _make_gym_obs from original LBF
                obs = self._get_original_obs()

            case "multigrid_flattened":
                obs_list = self.gen_obs(observe_other_agents=self.observe_other_agents)

                for i, obs in enumerate(obs_list):
                    obs_list[i] = obs.flatten()
                obs = np.vstack(obs_list)

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
            y, x = self._transform_to_ego_agent_frame(
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
        if (not self.observe_agent_levels) and (obj_type == "agent"):
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
        match self.env_obs_type:
            case "original":
                obs_size: int = self.observation_space.shape[1]
            case "multigrid_flattened":
                obs_size: int = self.observation_space.shape[1]

            case _:
                raise NotImplementedError

        return obs_size

    def _set_observation_space(self) -> spaces.Space:
        match self.env_obs_type:
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

                if not self.observe_agent_levels:
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

                # following original env
                # team_obs_space = spaces.Tuple(
                #     tuple([obs_space] * len(self.agents))
                # )

                # get joint obs space for the team
                team_obs_space = spaces.Box(
                    low=np.repeat(min_obs_single, repeats=self.num_agents, axis=0),
                    high=np.repeat(max_obs_single, repeats=self.num_agents, axis=0),
                    dtype=np.int_,
                )

            case "multigrid_flattened":
                # agent obs is a square with radius of self.view_size centered on the agent
                team_obs_space = spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        self.num_agents,
                        self.world.encode_dim * self.sight * self.sight,
                    ),
                    dtype=np.int_,
                )

            case _:
                raise NotImplementedError

        return team_obs_space

    # actions
    @property
    def avail_actions(self):
        # added this method to interface with PYMARL training loop
        # returns list of agent lists, where each agent's list has binary values representing
        # available actions
        # based on the MAIC paper's implementation of LBF with some minor cleanup
        # https://github.com/mansicer/MAIC/blob/main/src/envs/lbforaging/foraging.py

        return [self._get_avail_agent_actions(agent) for agent in self.agents]

    def _get_avail_agent_actions(self, agent: LBFAgent) -> list[int]:
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
        env_agent_action_space = spaces.Discrete(len(self.actions))
        action_space = [env_agent_action_space] * len(self.agents)
        ac_dim = len(self.actions)

        # if self.team_bandwidth_allocation_action:
        #     comms_action_space = spaces.Box(low=0, high=1)
        #     action_space.append(comms_action_space)
        #     ac_dim = ac_dim + 1

        # convert from list of spaces to gymnasium space
        action_space = spaces.Tuple(action_space)

        return action_space, ac_dim

    # rendering
    def render(self):
        img = super().render()

        # render actions in a separate image that we then append to base env image
        # determine info image width and create a blank white image
        info_img = 255 * np.ones((img.shape[0], 4 * self.tile_size, 3), dtype=img.dtype)

        # draw each agent's action as text stacked vertically
        line_height = int(self.tile_size * 0.9)

        # header indicating these are pre-transition actions
        header_lines = f"Pre-transition\nactions, t={self._t_render}:"
        header_y = int(self.tile_size * 0.5)
        for line in header_lines.split("\n"):
            putText(
                info_img,
                line,
                (10, header_y),
                fontFace=FontConfig.fontFace,
                fontScale=0.4,
                color=(0, 0, 0),
                thickness=1,
                lineType=FontConfig.lineType,
            )
            header_y += int(self.tile_size * 0.5)

        # start_y set below header to avoid overlap
        start_y = header_y + int(line_height * 0.8)
        # actions
        for i, action in enumerate(self._pre_step_actions):

            # convert from int to action name if not none
            if action is not None:
                action = self.actions(action).name.title()
            text = f"Agent {i}: {action}"

            y = start_y + i * line_height
            putText(
                info_img,
                text,
                (10, y),
                fontFace=FontConfig.fontFace,
                fontScale=0.4,
                color=(0, 0, 0),
                thickness=1,
                lineType=FontConfig.lineType,
            )

        # append info image to the right of env image
        updated_img = np.concatenate([img, info_img], axis=1)

        return updated_img

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

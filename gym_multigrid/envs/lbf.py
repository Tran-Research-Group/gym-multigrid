import enum
import warnings
from typing import Any, Literal, Type, TypedDict
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from gym_multigrid.core.agent import Agent, LBFActions
from gym_multigrid.core.constants import NAV_DIR_TO_VEC
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.object import Goal, Wall, WorldObj
from gym_multigrid.core.world import LBFWorld, World
from gym_multigrid.utils.rendering import (
    fill_coords,
    point_in_circle,
)
from gym_multigrid.typing_utils import Position
from gym_multigrid.multigrid import MultiGridEnv
from gym_multigrid.policy import AgentPolicy
from gym_multigrid.policy.prey_pred import PREY_PRED_POLICIES
from gym_multigrid.policy.prey_pred.utils import a_star


class FruitTypeDict(TypedDict):
    """
    Configuration for a type of prey object in the environment.

    Attributes
    ----------
    reward : float
        The reward for collecting fruit of this type.
    level : int
        The total level of adjacent agents required to collect this fruit.
    """

    capture_reward: float
    level: int
    color: str

class Fruit(WorldObj):
    def __init__(
        self,
        fruit_config: FruitTypeDict,
        world: World,
    ):
        self.fruit_config: FruitTypeDict = fruit_config

        self.neighbor_pos_offsets: NDArray[np.int_] = np.array(
            [[-1, 0], [1, 0], [0, -1], [0, 1]]
        )
        self.neighbor_pos: NDArray[np.int_] = np.zeros((4, 2), dtype=np.int_)

        super().__init__(
            world=world,
            reward=fruit_config["capture_reward"],
            color=fruit_config["color"],
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
        required_level = self.fruit_config["level"]
        level_neighbor_agents: int = 0

        agent_ids: list[int] = []

        for neighbor_pos in self.neighbor_pos:
            cell = grid.get(*neighbor_pos)
            if isinstance(cell, Agent):
                level_neighbor_agents += cell.get_level()
                agent_ids.append(cell.index)
            else:
                pass

        capturable: bool = level_neighbor_agents >= required_level
        agent_ids = agent_ids if capturable else []

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
                self.fruit_config["level"],
            )
        else:
            return (
                self.world.OBJECT_TO_IDX[self.type],
                self.world.COLOR_TO_IDX[self.color],
                self.fruit_config["level"],
                0,
                0,
                0,
            )
class AgentConfigDict(TypedDict):
    init_pos: tuple[int, int]
    level: int
    color: str

class LBFAgent(Agent):
    def __init__(
        self,
        world: World,
        index: int,
        agent_config: AgentConfigDict,
        view_size: int | None = None,
    ):
        self.agent_config: AgentConfigDict = agent_config
        self.neighbor_pos_offsets: NDArray[np.int_] = np.array(
            [[-1, 0], [1, 0], [0, -1], [0, 1]]
        )
        self.neighbor_pos: NDArray[np.int_] = np.zeros((4, 2), dtype=np.int_)

        super().__init__(
            world=world,
            index=index,
            actions=LBFActions,
            color=agent_config["color"],
            type="agent",
            view_size=view_size,
            dir_to_vec=NAV_DIR_TO_VEC,
        )

    def get_level(self) -> int:
        return self.agent_config["level"]
    
    def get_init_pos(self) -> tuple[int, int]:
        return self.agent_config["init_pos"]
    
    @property
    def pos(self) -> Position:
        return (self._pos[0], self._pos[1])

    @pos.setter
    def pos(self, pos: Position) -> None:
        self._pos = pos
        if pos is not None:
            self.neighbor_pos = pos + self.neighbor_pos_offsets

    def reset(self) -> None:
        super().reset()
        if self.pos is not None:
            self.neighbor_pos = self.pos + self.neighbor_pos_offsets
        else:
            self.neighbor_pos = np.zeros((4, 2), dtype=np.int_)
    
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
                self.agent_config["level"],
            )

class LBFGameEnv(MultiGridEnv):
    """
    Environment in which the agents have to collect the balls
    """

    def __init__(self, *args, **kwargs):
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
        self.size = kwargs["layout_config"]["size"]
        self.num_rooms = kwargs["layout_config"]["num_rooms"]
        self.waypoints = kwargs["layout_config"]["room_configs"]
        self.field_map = kwargs["layout_config"]["field_map"]
        
        self.num_fruits = kwargs["fruit_config"]["num_fruit"]
        self.total_num_fruits = int(np.sum(np.array(kwargs["fruit_config"]["num_fruit"])))
        self.collected_fruit = 0
        self.fruits_index = kwargs["fruit_config"]["fruits_index"]
        self.fruits_reward = kwargs["fruit_config"]["fruits_reward"]
        self.num_fruit_types = len(kwargs["fruit_config"]["fruits_index"])
        
        self.agents_index = kwargs["agent_config"]["index"]
        self.world = LBFWorld
        self.actions_set = LBFActions
        partial_obs: bool = False
        self.info = {}

        agents = []
        for i, agent_index in enumerate(self.agents_index):
            temp_agent_config: AgentConfigDict = {
                "init_pos": kwargs["agent_config"]["init_pos"][i],
                "level": kwargs["agent_config"]["level"][i],
                "color": self.world.IDX_TO_COLOR[agent_index],
            }
            agents.append(LBFAgent(self.world, i, temp_agent_config))

        super().__init__(
            grid_size=self.size,
            width=None,
            height=None,
            max_steps=400,
            world=self.world,
            see_through_walls=False,
            agents=agents,
            partial_obs=partial_obs,
            actions_set=self.actions_set,
            render_mode="rgb_array",
        )

    def _gen_grid(self, width: int, height: int):
        # Create the grid
        self.grid = Grid(width, height, self.world)

        # Translate the maze structure into the grid
        for y, row in enumerate(self.field_map):
            for x, cell in enumerate(row):
                if cell == "#":
                    self.put_obj(Wall(self.world, type="wall", color="grey"), x, y)
                elif cell == "1" or cell == "2" or cell == "3":
                    temp_fruit_config: FruitTypeDict = {  # type: ignore
                        "capture_reward": float(self.fruits_reward[int(cell) - 1]),
                        "level": int(cell),
                        "color": str(self.world.IDX_TO_COLOR[self.fruits_index[int(cell) - 1]])
                    }
                    self.put_obj(Fruit(temp_fruit_config, self.world), x, y)
                else:
                    pass

        # Place Waypoints
        self.waypoint_pos: list[Position] = []
        for r in range(self.num_rooms):
            for waypoint in self.waypoints[r]["flag_positions"]:
                waypoint_obj = Goal(
                    self.world,
                    color="green",
                    reward=5,
                )
                self.put_obj(waypoint_obj, waypoint[0], waypoint[1])
                self.waypoint_pos.append(waypoint_obj.pos)
        
        # Place the agents
        for agent in self.agents:
            self.place_agent(
                agent,
                agent.get_init_pos(),
            )
        self.init_grid: Grid = self.grid.copy()
    
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.int_], dict[str, Any]]:
        self.step_count: int = 0
        self.collected_fruit: NDArray[np.int_] = np.zeros(self.num_fruit_types, dtype=np.int_)
        self.current_room: int = 0
        self.room_cleared = [False for _ in range(self.num_rooms)]

        self._reset_gym(seed=seed)
        self._gen_grid(self.width, self.height)

        # Reset the agents. If agent_pos_list is provided, use it to reset the agents
        agent_pos_list: list[tuple[int, int]] | None = None
        if options is not None:
            agent_pos_list = options.get("agent_pos_list", None)
        else:
            pass
        self._reset_agents(agent_pos_list=agent_pos_list)

        obs: NDArray[np.int_] = self.grid.encode()
        info: dict[str, Any] = self.info

        return obs, info

    def _reset_agents(
        self, agent_pos_list: list[tuple[int, int]] | None = None
    ) -> None:
        """
        Reset the agents in the environment.
        """
        for agent in self.agents:
            agent.reset(self.np_random)

        use_agent_pos_list: bool = agent_pos_list is not None
        if agent_pos_list is not None:
            if len(agent_pos_list) != len(self.agents):
                warnings.warn(
                    f"""Invalid agent position list: Expected {len(self.agents)} positions, got {len(agent_pos_list)}.
                    Resetting agents to their initial positions.""",
                    UserWarning,
                )
                use_agent_pos_list = False
            else:
                pass

        for agent in self.agents:
            agent_pos: tuple[int, int] = (
                agent.get_init_pos()
                if not use_agent_pos_list or not isinstance(agent_pos_list, list)
                else agent_pos_list.pop(0)
            )
            self.place_agent(agent, agent_pos)
        
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
        rewards: NDArray[np.float64] = np.zeros(len(actions))
        
        # Order to apply the actions to the agents
        order: NDArray[np.int_] = self.np_random.permutation(len(self.agents))

        for i in order:
            agent = self.agents[i]
            act: int = actions[i]

            if agent.terminated:
                continue
            else:
                next_pos: tuple[int, int] = self._get_next_pos(agent, act)
                next_cell: None | WorldObj = self.grid.get(*next_pos)

                if isinstance(next_cell, Wall):
                    continue
                elif act == self.actions.LOAD:
                    for neighbor_pos in agent.neighbor_pos:
                        cell = self.grid.get(*neighbor_pos)
                        if isinstance(cell, Fruit) and cell.check_capture_condition(self.grid):
                            print(f"Agent {i} captured fruit at {cell.pos} with level {cell.fruit_config['level']}!")
                            self._handle_pickup(i, rewards, neighbor_pos, cell)
                        else:
                            pass
                elif next_cell is None or next_cell.can_overlap():
                    # Move agent
                    self.grid.set(*next_pos, agent)
                    self.grid.set(*agent.pos, None)
                    # Change the dir of the agent
                    if agent.pos != next_pos:
                        dir_vec: NDArray[np.int_] = np.array(next_pos) - np.array(
                            agent.pos
                        )
                        agent.dir = agent.vec2dir(dir_vec)
                    agent.pos = next_pos
                    
                    # Check if the agent has reached a waypoint
                    if agent.pos in self.waypoint_pos and self.room_cleared[self.current_room]:
                        #print(f"Agent {i} reached waypoint at {agent.pos} in room {self.current_room}!")
                        self._reward(i, rewards, next_cell.reward if next_cell else 0)
        # if all agents are at waypoints, move to the next room
        if all(agent.pos in self.waypoint_pos for agent in self.agents) and self.room_cleared[self.current_room]:
            print(f"All agents reached waypoints for room {self.current_room}. Moving to next room...")
            # remove walls and waypoints for the current room
            for waypoint in self.waypoints[self.current_room]["flag_positions"]:
                #self.grid.set(waypoint[0], waypoint[1], None)
                self.waypoint_pos.remove(waypoint)
            for wall_pos in self.waypoints[self.current_room]["wall_positions"]:
                #print(f"Removing wall at {wall_pos}")
                self.grid.set(wall_pos[0], wall_pos[1], None)
            self.current_room += 1
        
        # Terminate the episode if all rooms have been cleared or max steps reached
        terminated = self.current_room == self.num_rooms
        truncated = self.step_count >= self.max_steps

        self.step_count += 1
        print(f"Step: {self.step_count}, Total Collected Fruit: {self.collected_fruit}, Current Room: {self.current_room}")

        return self.grid.encode(), float(np.sum(rewards)), terminated, truncated, self.info
    
    def _get_next_pos(self, agent, action: int) -> tuple[int, int]:
        self.actions: Type[LBFActions]

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

    def _handle_pickup(
        self,
        i,
        rewards: NDArray[np.float64],
        fwd_pos: Position,
        fwd_cell: WorldObj | None,
    ) -> None:
        if fwd_cell and isinstance(fwd_cell, Fruit):
            fwd_cell.pos = np.array([-1, -1])  # Move the fruit off the grid
            self.grid.set(*fwd_pos, None)
            self.collected_fruit[fwd_cell.fruit_config["level"] - 1] += 1
            # print(f"Collected fruit of type {fwd_cell.fruit_config['level']}. Total collected: {self.collected_fruit}")
            # reset fruit collected for the room if room cleared
            if self.collected_fruit.tolist() == self.waypoints[self.current_room]["fruit_count"]:
                self.room_cleared[self.current_room] = True
                self.collected_fruit: NDArray[np.int_] = np.zeros(self.num_fruit_types, dtype=np.int_)
            self._reward(i, rewards, fwd_cell.reward)
    
    def _reward(
        self, current_agent: int, rewards: NDArray[np.float64], reward: float = 1
    ) -> None:
        """
        Compute the reward to be given upon success
        """
        rewards[current_agent] += reward

class AgentState(IntEnum):
    MOVING_TO_WAYPOINT = 0
    LOADING_FRUIT      = 1
    WAITING_AT_GOAL    = 2

class GreedyPredatorPolicy:
    def __init__(
        self,
        fruit_list,
        goal_list,
        random_prob: float = 0.1,
        action_set: Type[enum.IntEnum] = LBFActions,
        dir_to_vec: list[NDArray[np.int_]] = NAV_DIR_TO_VEC,
        random_generator: np.random.Generator | None = None,
    ):
        self.fruit_list = fruit_list
        self.fruit_idx: int = 0
        self.goal_list = goal_list
        self.goal_idx: int = 0
        self.random_prob: float = random_prob
        self.action_set: Type[enum.IntEnum] = action_set
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
        else: # Still fruits to pick up in current room
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
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            if grid.get(r + dr, c + dc) and isinstance(grid.get(r + dr, c + dc), Fruit):
                return True
        return False
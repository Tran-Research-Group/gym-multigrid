import enum
import warnings
from typing import Any, Literal, Type, TypedDict

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from gym_multigrid.core.agent import Agent, NavigationActions
from gym_multigrid.core.constants import NAV_DIR_TO_VEC, COLORS, COLOR_TO_IDX, IDX_TO_COLOR
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.object import Goal, Ball, Wall, WorldObj
from gym_multigrid.core.world import GridWorld, World
from gym_multigrid.typing import Position

class AgentConfigDict(TypedDict):
    init_pos: tuple[int, int]
    level: int
    color: str

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

    def check_capture_condition(self, grid: Grid) -> tuple[bool, list[int]]:
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
            cell: None | WorldObj = grid.get(*neighbor_pos)
            if isinstance(cell, Agent):
                level_neighbor_agents += 1
                agent_ids.append(cell.index)
            else:
                pass

        capturable: bool = level_neighbor_agents >= required_level
        agent_ids = agent_ids if capturable else []

        return capturable, agent_ids

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
        self.size = kwargs["size"]
        self.num_fruits = kwargs["num_fruit"]
        self.total_num_fruits = int(np.sum(np.array(kwargs["num_fruit"])))
        self.collected_fruit = 0
        self.fruits_index = kwargs["fruits_index"]
        self.fruits_reward = kwargs["fruits_reward"]
        self.num_fruit_types = len(kwargs["fruits_index"])
        self.agents_index = kwargs["agents_index"]
        self.world = GridWorld
        self.actions_set = NavigationActions
        partial_obs: bool = False
        self.info = {}

        agents = []
        for i in self.agents_index:
            agents.append(Agent(self.world, i))

        super().__init__(
            grid_size=self.size,
            width=None,
            height=None,
            max_steps=100,
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

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        # generate inner walls of rooms
        self.grid.horz_wall(0, width // 2)
        self.grid.vert_wall(width // 2, 0)

        # place goal
        goal = self.layout_config.spawn_configs[self.spawn_type].goal
        goal_obj = Goal(
            self.world,
            0,
            color="green",
            reward=goal.reward,
            absorbing=goal.absorbing,
        )
        self.place_object(goal_obj, **self._parse_init_pos(goal.pos))
        self.goal_pos: Position = goal_obj.pos

        # Place Waypoints
        self.waypoint_pos: list[Position] = []
        for waypoint in self.layout_config.spawn_configs[self.spawn_type].waypoints:
            waypoint_obj = Goal(
                self.world,
                color="dark_grey",
                reward=waypoint.reward,
                absorbing=waypoint.absorbing,
            )
            self.place_object(waypoint_obj, **self._parse_init_pos(waypoint.pos))
            self.waypoint_pos.append(waypoint_obj.pos)
        
        for number, index, reward in zip(
            self.num_fruits, self.fruits_index, self.fruits_reward
        ):
            for _ in range(number):
                level = 1
                temp_fruit_config: FruitTypeDict = {
                    "capture_reward": reward,
                    "level": level,
                    "color": IDX_TO_COLOR[index]
                }
                self.place_obj(Fruit(temp_fruit_config, self.world))
                level += 1
        
        self.init_grid: Grid = self.grid.copy()
    
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.int_], dict[str, Any]]:
        self.step_count: int = 0
        self.collected_fruit: int = 0

        self._reset_gym(seed=seed)
        self._gen_grid(self.width, self.height)

        # Reset the agents. If agent_pos_list is provided, use it to reset the agents
        agent_pos_list: list[tuple[int, int]] | None = None
        if options is not None:
            agent_pos_list = options.get("agent_pos_list", None)
        else:
            pass
        self._reset_agents(agent_pos_list=agent_pos_list)

        obs: NDArray[np.int_] = self._get_obs()
        info: dict[str, Any] = self._get_info()

        return obs, info
        
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
        reward: float = 0

        actions: list[int] = np.array(action).flatten().astype(np.int_).tolist()
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

                if isinstance(next_cell, WorldObj) and not next_cell.can_overlap():
                    continue
                else:
                    # Move agent
                    self.grid.set(*next_pos, agent)
                    self.grid.set(*agent.pos, self.init_grid.get(*agent.pos))
                    # Change the dir of the agent
                    if agent.pos != next_pos:
                        dir_vec: NDArray[np.int_] = np.array(next_pos) - np.array(
                            agent.pos
                        )
                        agent.dir = agent.vec2dir(dir_vec)
                    agent.pos = next_pos

                    # Update the agent's bg_color
                    init_grid_cell: WorldObj | None = self.init_grid.get(*agent.pos)
                    if init_grid_cell is not None:
                        agent.bg_color = init_grid_cell.color

                    # Determine rewards and remove collected fruit from the grid

        # Terminate the episode if all the fruit are captured
        terminated = self.collected_fruit == self.num_fruits
        truncated = self.step_count >= self.max_steps

        self.step_count += 1

        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def _get_next_pos(self, agent, action: int) -> tuple[int, int]:
        self.actions: Type[NavigationActions]

        match action:
            case self.actions.STAY:
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
            fwd_cell.pos = Position([-1, -1])
            ball_idx = COLOR_TO_IDX[fwd_cell.color]
            self.grid.set(*fwd_pos, None)
            if self.respawn:
                self._respawn(ball_idx)
            self.collected_fruit += 1
            self._reward(i, rewards, fwd_cell.reward)
    
    def _reward(
        self, current_agent: int, rewards: NDArray[np.float64], reward: float = 1
    ) -> None:
        """
        Compute the reward to be given upon success
        """
        rewards[current_agent] += reward
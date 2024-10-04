from itertools import chain
from typing import Final, Literal, TypeAlias, TypedDict
from typing import Any, Iterable, SupportsFloat, TypeVar

from gymnasium import spaces
from gymnasium.core import ActType, ObsType
import numpy as np
import random
from numpy.typing import NDArray

from gym_multigrid.core.constants import *
from gym_multigrid.utils.window import Window
from gym_multigrid.core.agent import Agent, PolicyAgent, AgentT, FRActions
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.object import Goal
from gym_multigrid.core.world import FRWorld
from gym_multigrid.multigrid import MultiGridEnv
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


class FourRooms(MultiGridEnv):
    """
    Environment for capture the flag with multiple agents with N blue agents and M red agents.
    """

    def __init__(
        self,
        agent_pos=None,
        goal_pos=None,
        grid_size: tuple = (19, 19),
        agent_view_size: int = 7,
        max_steps: int = 100,
        highlight_visible_cells: bool | None = True,
        tile_size: int = 20,
        partial_observability: bool = False,
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
    ) -> None:
        """
        Initialize a new capture the flag environment.

        Parameters
        ----------

        """
        self.width = grid_size[0]
        self.height = grid_size[1]
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.world = FRWorld
        self.actions_set = FRActions

        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        see_through_walls: bool = False

        self.agents = [
            Agent(
                self.world,
                color="blue",
                bg_color="light_blue",
                view_size=agent_view_size,
                actions=self.actions_set,
                type="agent",
            )
        ]

        self.grids = {}
        self.grid_imgs = {}

        super().__init__(
            width=self.width,
            height=self.height,
            max_steps=max_steps,
            see_through_walls=see_through_walls,
            agents=self.agents,
            agent_view_size=agent_view_size,
            actions_set=self.actions_set,
            partial_obs=partial_observability,
            world=self.world,
            render_mode=render_mode,
            highlight_visible_cells=highlight_visible_cells,
            tile_size=tile_size,
        )

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height, self.world)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # # For each row of rooms
        # for j in range(0, 2):
        #     # For each column
        #     for i in range(0, 2):
        #         xL = i * room_w
        #         yT = j * room_h
        #         xR = xL + room_w
        #         yB = yT + room_h

        #         # Bottom wall and door
        #         if i + 1 < 2:
        #             self.grid.vert_wall(xR, yT, room_h)
        # pos = (xR, self._rand_int(yT + 1, yB - 1))
        # self.grid.set(*pos, None)

        # # Bottom wall and door
        # if j + 1 < 2:
        #     self.grid.horz_wall(xL, yB, room_w)
        # pos = (self._rand_int(xL + 1, xR - 1), yB)
        # self.grid.set(*pos, None)

        doorway_positions = [(3, 6), (6, 2), (10, 6), (7, 9)]
        vert_wall_positions = [(6, 0), (7, 6)]
        hor_wall_positions = [(0, 6), (6, 6)]

        # Bottom wall and door
        for coord in vert_wall_positions:
            self.grid.vert_wall(coord[0], coord[1], room_h)

        # Bottom wall and door
        for coord in hor_wall_positions:
            self.grid.horz_wall(coord[0], coord[1], room_w)

        for pos in doorway_positions:
            self.grid.set(*pos, None)

        # Randomize the player start position and orientation
        if self._goal_default_pos is not None:
            goal = Goal(self.world, 0)
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.place_obj(Goal(self.world, 0))

        if self._agent_default_pos is not None:
            for agent in self.agents:
                self.place_agent(agent, pos=self._agent_default_pos)
            # assuming random start direction
            self.agent_dir = self._rand_int(0, 4)
        else:
            for agent in self.agents:
                self.place_agent(agent)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        ### intentional to not feed seed since the grid and agent are fixed
        # obs = super().reset(seed=seed, options=options)
        obs, info = super().reset(options=options)

        ### NOTE: not multiagent setting
        self.agent_dir = self.agents[0].dir
        self.agent_pos = self.agents[0].pos

        ### NOTE: NOT MULTIAGENT SETTING
        observations = {"image": obs[0], "direction": self.agent_dir}
        return observations, info

    def step(self, actions):
        self.step_count += 1

        ### NOTE: MULTIAGENT SETTING NOT IMPLEMENTED
        actions = [actions]
        order = np.random.permutation(len(actions))

        rewards = np.zeros(len(actions))

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
                    elif fwd_cell.type == "switch":
                        self._handle_switch(i, rewards, fwd_pos, fwd_cell)
                    elif fwd_cell.type == "ball":
                        rewards = self._handle_pickup(i, rewards, fwd_pos, fwd_cell)
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
                    elif fwd_cell.type == "switch":
                        self._handle_switch(i, rewards, fwd_pos, fwd_cell)
                    elif fwd_cell.type == "ball":
                        rewards = self._handle_pickup(i, rewards, fwd_pos, fwd_cell)
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
                    elif fwd_cell.type == "switch":
                        self._handle_switch(i, rewards, fwd_pos, fwd_cell)
                    elif fwd_cell.type == "ball":
                        rewards = self._handle_pickup(i, rewards, fwd_pos, fwd_cell)
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
                    elif fwd_cell.type == "switch":
                        self._handle_switch(i, rewards, fwd_pos, fwd_cell)
                    elif fwd_cell.type == "ball":
                        rewards = self._handle_pickup(i, rewards, fwd_pos, fwd_cell)
                elif fwd_cell is None or fwd_cell.can_overlap():
                    self.grid.set(*self.agents[i].pos, None)
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.agents[i].pos = fwd_pos
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)
            else:
                assert False, "unknown action"

        ### NOTE: not multiagent setting
        self.agent_dir = self.agents[0].dir
        self.agent_pos = self.agents[0].pos

        terminated = done
        truncated = True if self.step_count >= self.max_steps else False

        if self.partial_obs:
            obs = self.gen_obs()
        else:
            obs = [
                self.grid.encode_for_agents(agent_pos=self.agents[i].pos)
                for i in range(len(actions))
            ]

        obs = [self.world.normalize_obs * ob for ob in obs]

        ### NOTE: not multiagent
        observations = {"image": obs[0], "direction": self.agent_dir}

        return observations, rewards, terminated, truncated, {}

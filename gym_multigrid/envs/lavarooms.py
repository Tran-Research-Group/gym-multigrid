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
from gym_multigrid.core.object import Lava
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


class LavaRooms(MultiGridEnv):
    """
    Environment for capture the flag with multiple agents with N blue agents and M red agents.
    """

    def __init__(
        self,
        env_seed: int = 0,
        grid_size: tuple = (9, 9),
        agent_view_size: int = 7,
        max_steps: int = 100,
        tile_size: int = 20,
        highlight_visible_cells: bool | None = True,
        partial_observability: bool = False,
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
    ) -> None:
        """
        Initialize a new capture the flag environment.

        Parameters
        ----------

        """
        if env_seed < 0 or env_seed >= 2:
            raise ValueError(
                f"The Lavaroom only accepts env_seed of 0 and 1, given {env_seed}"
            )
        else:
            self.env_seed = env_seed

        self.width = grid_size[0]
        self.height = grid_size[1]
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.world = FRWorld
        self.actions_set = FRActions

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

        self.doorway_positions = [(10, 6), (2, 6)]
        self.vert_wall_positions = [(6, 0), (7, 6)]
        self.hor_wall_positions = [(0, 6), (6, 6)]

        self.goal_positions = [(6, 3), (1, 1)]  # (7, 1)
        self.agent_positions = [(6, 9), (1, 11)]
        self.lava_positions = [
            [
                (2, 6),
                (2, 7),
                (2, 8),
                (3, 7),
                (3, 8),
                (4, 7),
                (4, 8),
            ],
            [
                (10, 6),
                (10, 7),
                (10, 8),
                (9, 7),
                (9, 8),
                (8, 7),
                (8, 8),
            ],
        ]

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

        # # Bottom wall and door
        # for coord in vert_wall_positions:
        #     self.grid.vert_wall(coord[0], coord[1], room_h)

        # Bottom wall and door
        for coord in self.hor_wall_positions:
            self.grid.horz_wall(coord[0], coord[1], room_w)

        for pos in self.doorway_positions:
            self.grid.set(*pos, None)

        # goal allocation
        # place goal
        goal = Goal(self.world, 0)
        self.put_obj(goal, *self.goal_positions[self.env_seed])
        goal.init_pos, goal.cur_pos = self.goal_positions[self.env_seed]

        # lava allocation
        lava_spawn_location = random.sample([0, 1], 1)[0]
        random_lava_positions = random.sample(
            self.lava_positions[lava_spawn_location], 3
        )
        for lava_pos in random_lava_positions:
            lava = Lava(self.world)
            self.put_obj(lava, *lava_pos)

        # agent allocation
        self.place_agent(self.agents[0], pos=self.agent_positions[self.env_seed])

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        ### intentional to not feed seed since the grid and agent are fixed
        obs, info = super().reset(options=options)

        ### NOTE: not multiagent setting
        self.agent_pos = self.agents[0].pos

        ### NOTE: NOT MULTIAGENT SETTING
        observations = {"image": obs[0]}
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
                        rewards += 1.0 - 0.5 * (self.step_count / self.max_steps)
                    elif fwd_cell.type == "switch":
                        self._handle_switch(i, rewards, fwd_pos, fwd_cell)
                    elif fwd_cell.type == "ball":
                        rewards = self._handle_pickup(i, rewards, fwd_pos, fwd_cell)
                    elif fwd_cell.type == "lava":
                        rewards[i] = -0.25
                elif fwd_cell is None or fwd_cell.can_overlap():
                    self.grid.set(*self.agents[i].pos, None)
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.agents[i].pos = fwd_pos
                else:
                    # rewards -= 0.001
                    rewards = 0
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)

            # Rotate right
            elif actions[i] == self.actions.right:
                # Get the contents of the cell in front of the agent
                fwd_pos = tuple(a + b for a, b in zip(curr_pos, (0, +1)))
                fwd_cell = self.grid.get(*fwd_pos)
                if fwd_cell is not None:
                    if fwd_cell.type == "goal":
                        done = True
                        rewards += 1.0 - 0.5 * (self.step_count / self.max_steps)
                    elif fwd_cell.type == "switch":
                        self._handle_switch(i, rewards, fwd_pos, fwd_cell)
                    elif fwd_cell.type == "ball":
                        rewards = self._handle_pickup(i, rewards, fwd_pos, fwd_cell)
                    elif fwd_cell.type == "lava":
                        rewards[i] = -0.25
                elif fwd_cell is None or fwd_cell.can_overlap():
                    self.grid.set(*self.agents[i].pos, None)
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.agents[i].pos = fwd_pos
                else:
                    rewards = 0
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)

            # Move forward
            elif actions[i] == self.actions.up:
                # Get the contents of the cell in front of the agent
                fwd_pos = tuple(a + b for a, b in zip(curr_pos, (-1, 0)))
                fwd_cell = self.grid.get(*fwd_pos)
                if fwd_cell is not None:
                    if fwd_cell.type == "goal":
                        done = True
                        rewards += 1.0 - 0.5 * (self.step_count / self.max_steps)
                    elif fwd_cell.type == "switch":
                        self._handle_switch(i, rewards, fwd_pos, fwd_cell)
                    elif fwd_cell.type == "ball":
                        rewards = self._handle_pickup(i, rewards, fwd_pos, fwd_cell)
                    elif fwd_cell.type == "lava":
                        rewards[i] = -0.25
                elif fwd_cell is None or fwd_cell.can_overlap():
                    self.grid.set(*self.agents[i].pos, None)
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.agents[i].pos = fwd_pos
                else:
                    rewards = 0
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)

            elif actions[i] == self.actions.down:
                # Get the contents of the cell in front of the agent
                fwd_pos = tuple(a + b for a, b in zip(curr_pos, (+1, 0)))
                fwd_cell = self.grid.get(*fwd_pos)
                if fwd_cell is not None:
                    if fwd_cell.type == "goal":
                        done = True
                        rewards += 1.0 - 0.5 * (self.step_count / self.max_steps)
                    elif fwd_cell.type == "switch":
                        self._handle_switch(i, rewards, fwd_pos, fwd_cell)
                    elif fwd_cell.type == "ball":
                        rewards = self._handle_pickup(i, rewards, fwd_pos, fwd_cell)
                    elif fwd_cell.type == "lava":
                        rewards[i] = -0.25
                elif fwd_cell is None or fwd_cell.can_overlap():
                    self.grid.set(*self.agents[i].pos, None)
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.agents[i].pos = fwd_pos
                else:
                    rewards = 0
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)
            else:
                assert False, "unknown action"

        ### NOTE: not multiagent setting
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
        observations = {"image": obs[0]}

        return observations, rewards, terminated, truncated, {}

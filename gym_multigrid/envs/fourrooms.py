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
        highlight: bool | None = True,
        img_tile_size: int = 20,
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
        self.img_tile_size = img_tile_size

        super().__init__(
            width=self.width,
            height=self.height,
            max_steps=max_steps,
            see_through_walls=see_through_walls,
            agents=self.agents,
            agent_view_size=agent_view_size,
            actions_set=self.actions_set,
            partial_obs=partial_observability,
            highlight=highlight,
            world=self.world,
            render_mode=render_mode,
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
        grid_img = self.grid.render(tile_size=self.img_tile_size)

        if self._agent_default_pos is not None:
            for agent in self.agents:
                self.place_agent(agent, pos=self._agent_default_pos)
            # assuming random start direction
            self.agent_dir = self._rand_int(0, 4)
        else:
            for agent in self.agents:
                self.place_agent(agent)

        return grid_img

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        ### intentional to not feed seed since the grid and agent are fixed
        # obs = super().reset(seed=seed, options=options)
        obs = super().reset(options=options)

        ### NOTE: not multiagent setting
        self.agent_dir = self.agents[0].dir
        self.agent_pos = self.agents[0].pos

        ### NOTE: NOT MULTIAGENT SETTING
        observations = {"image": obs[0], "direction": self.agent_dir}
        return observations, {}

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

    def render(
        self, close: bool = False, highlight: bool = False, tile_size: int | None = None
    ):
        """
        Render the whole-grid human view
        """

        if highlight is None:
            self.highlight = highlight

        if tile_size is None:
            tile_size = self.img_tile_size

        if close:
            if self.window:
                self.window.close()
            return

        if self.render_mode == "human" and not self.window:
            self.window = Window("gym_multigrid")
            self.window.show(block=False)

        if self.highlight:
            # Compute which cells are visible to the agent
            _, vis_masks = self.gen_obs_grid()

            highlight_masks = {
                (i, j): [] for i in range(self.width) for j in range(self.height)
            }

            for i, a in enumerate(self.agents):
                # Compute the world coordinates of the bottom-left corner
                # of the agent's view area
                f_vec = a.dir_vec
                r_vec = a.right_vec
                top_left = (
                    a.pos + f_vec * (a.view_size - 1) - r_vec * (a.view_size // 2)
                )

                # Mask of which cells to highlight

                # For each cell in the visibility mask
                for vis_j in range(0, a.view_size):
                    for vis_i in range(0, a.view_size):
                        # If this cell is not visible, don't highlight it
                        if not vis_masks[i][vis_i, vis_j]:
                            continue

                        # Compute the world coordinates of this cell
                        abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                        if abs_i < 0 or abs_i >= self.width:
                            continue
                        if abs_j < 0 or abs_j >= self.height:
                            continue

                        # Mark this cell to be highlighted
                        highlight_masks[abs_i, abs_j].append(i)

        # Render the whole grid
        img = self.grid.render(
            tile_size,
            highlight_masks=highlight_masks if self.highlight else None,
            uncached_object_types=self.uncahed_object_types,
        )

        if self.render_mode == "human":
            self.window.show_img(img)

        return img

    # def step(self, actions):
    #     self.step_count += 1

    #     ### NOTE: MULTIAGENT SETTING NOT IMPLEMENTED
    #     actions = [actions]
    #     order = np.random.permutation(len(actions))

    #     rewards = np.zeros(len(actions))

    #     for i in order:
    #         if (
    #             self.agents[i].terminated
    #             or self.agents[i].paused
    #             or not self.agents[i].started
    #         ):
    #             continue

    #         # Get the position in front of the agent
    #         fwd_pos = self.agents[i].front_pos

    #         # Get the contents of the cell in front of the agent
    #         fwd_cell = self.grid.get(*fwd_pos)

    #         # Rotate left
    #         if actions[i] == self.actions.left:
    #             self.agents[i].dir -= 1
    #             if self.agents[i].dir < 0:
    #                 self.agents[i].dir += 4

    #         # Rotate right
    #         elif actions[i] == self.actions.right:
    #             self.agents[i].dir = (self.agents[i].dir + 1) % 4

    #         # Move forward
    #         elif actions[i] == self.actions.forward:
    #             if fwd_cell is not None:
    #                 if fwd_cell.type == "goal":
    #                     done = True
    #                     rewards = self._reward(i, rewards, 1)
    #                 elif fwd_cell.type == "switch":
    #                     self._handle_switch(i, rewards, fwd_pos, fwd_cell)
    #                 elif fwd_cell.type == "ball":
    #                     rewards = self._handle_pickup(i, rewards, fwd_pos, fwd_cell)
    #             elif fwd_cell is None or fwd_cell.can_overlap():
    #                 self.grid.set(*self.agents[i].pos, None)
    #                 self.grid.set(*fwd_pos, self.agents[i])
    #                 self.agents[i].pos = fwd_pos
    #             self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)

    #         elif "build" in self.actions.available and actions[i] == self.actions.build:
    #             self._handle_build(i, rewards, fwd_pos, fwd_cell)

    #         # Pick up an object
    #         elif actions[i] == self.actions.pickup:
    #             self._handle_pickup(i, rewards, fwd_pos, fwd_cell)

    #         # Drop an object
    #         elif actions[i] == self.actions.drop:
    #             self._handle_drop(i, rewards, fwd_pos, fwd_cell)

    #         # Toggle/activate an object
    #         elif actions[i] == self.actions.toggle:
    #             if fwd_cell:
    #                 fwd_cell.toggle(self, fwd_pos)

    #         # Done action (not used by default)
    #         elif actions[i] == self.actions.done:
    #             pass

    #         else:
    #             assert False, "unknown action"

    #     ### NOTE: not multiagent setting
    #     self.agent_dir = self.agents[0].dir
    #     self.agent_pos = self.agents[0].pos
    #     terminated = self.agents[0].terminated

    #     truncated = True if self.step_count >= self.max_steps else False

    #     if self.partial_obs:
    #         obs = self.gen_obs()
    #     else:
    #         obs = [self.grid.encode_for_agents(agent_pos=self.agents[i].pos) for i in range(len(actions))]

    #     obs = [self.world.normalize_obs * ob for ob in obs]

    #     ### NOTE: not multiagent
    #     observations = {"image": obs[0], 'direction': self.agent_dir}

    #     return observations, rewards, terminated, truncated, {}

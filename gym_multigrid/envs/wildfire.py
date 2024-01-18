from gym_multigrid.multigrid import MultiGridEnv
from gym_multigrid.core.world import WildfireWorld
from gym_multigrid.core.agent import WildfireActions, Agent
from gym_multigrid.core.object import Tree
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.constants import (
    STATE_TO_IDX_WILDFIRE,
    TILE_PIXELS,
    STATE_IDX_TO_COLOR_WILDFIRE,
)
from gym_multigrid.utils.window import Window
from gym_multigrid.utils.misc import render_agent_tile
from collections import OrderedDict
from gymnasium.spaces import Box, Dict, Discrete
import numpy as np
from typing import Any


class WildfireEnv(MultiGridEnv):
    """
    Environment in which agents have to put out a fire
    """

    def __init__(
        self,
        alpha=0.05,
        beta=0.987,
        delta_beta=0.5,
        size=9,
        num_agents=4,
        agent_view_size=10,
        max_steps=300,
        partial_obs=False,
        actions_set=WildfireActions,
        render_mode="rgb_array",
    ):
        self.alpha = alpha
        self.beta = beta
        self.delta_beta = delta_beta
        self.num_agents = num_agents
        if num_agents > 4:
            raise ValueError("Number of agents cannot be greater than 4.")
        self.agent_view_size = agent_view_size
        self.max_steps = max_steps
        self.world = WildfireWorld
        self.grid_size = size
        self.grid_size_without_walls = size - 2
        self.burnt_trees = 0
        self.trees_on_fire = 0
        # add wildfire specific variables like num_burnt trees etc.

        agents = [
            Agent(
                world=self.world,
                index=i,
                view_size=agent_view_size,
                actions=actions_set,
                color="light_blue",
            )
            for i in range(self.num_agents)
        ]

        super().__init__(
            agents=agents,
            grid_size=size,
            max_steps=max_steps,
            partial_obs=partial_obs,
            agent_view_size=agent_view_size,
            actions_set=actions_set,
            world=self.world,
            render_mode=render_mode,
        )
        self.observation_space: Box | Dict = self._set_observation_space()
        self.action_space = Dict(
            {f"{a.index}": Discrete(len(self.actions)) for a in self.agents}
        )

    def _set_observation_space(self) -> Dict:
        low = np.full(self.grid_size_without_walls**2, 0)
        low = np.append(low, np.full(2 * self.num_agents, 1))
        high = np.full(self.grid_size_without_walls**2, 2)
        high = np.append(high, np.full(2 * self.num_agents, self.grid_size - 1))
        if (
            self.partial_obs
        ):  # right now partial obs is not supported. Modify to shorten low and high arrays.
            observation_space = Dict(
                {
                    f"{a.index}": Box(
                        low=low,
                        high=high,
                        dtype=np.int16,
                    )
                    for a in self.agents
                }
            )

        else:
            observation_space = Dict(
                {
                    f"{a.index}": Box(
                        low=low,
                        high=high,
                        dtype=np.int16,
                    )
                    for a in self.agents
                }
            )

        return observation_space

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height, self.world)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        # Insert trees in grid as per initial conditions of wildfire. Modify as needed.
        # Currently, center is on fire.
        if self.grid_size_without_walls % 2 == 0:
            trees_on_fire = [
                (self.grid_size_without_walls / 2, self.grid_size_without_walls / 2),
                (
                    self.grid_size_without_walls / 2 + 1,
                    self.grid_size_without_walls / 2,
                ),
                (
                    self.grid_size_without_walls / 2,
                    self.grid_size_without_walls / 2 + 1,
                ),
                (
                    self.grid_size_without_walls / 2 + 1,
                    self.grid_size_without_walls / 2 + 1,
                ),
            ]
            self.trees_on_fire += 4
        else:
            trees_on_fire = [
                (
                    (self.grid_size_without_walls + 1) / 2,
                    (self.grid_size_without_walls + 1) / 2,
                )
            ]
            self.trees_on_fire += 1

        num_healthy_trees = self.grid_size_without_walls**2 - len(trees_on_fire)

        for pos in trees_on_fire:
            self.put_obj(
                Tree(self.world, STATE_TO_IDX_WILDFIRE["on fire"]),
                int(pos[0]),
                int(pos[1]),
            )
        for _ in range(num_healthy_trees):
            self.place_obj(Tree(self.world, STATE_TO_IDX_WILDFIRE["healthy"]))

        # Helper grid is a work around for grid unable to store multiple objects at a single cell.
        # Helper grid is the grid without agents.
        self.helper_grid = self.grid.copy()

        # Place UAVs at start positions
        start_pos = [
            (1, 1),
            (1, self.grid.height - 2),
            (self.grid.width - 2, 1),
            (self.grid.width - 2, self.grid.height - 2),
        ]  # start positions for up to 4 agents. Modify as needed.
        for i, a in enumerate(self.agents):
            self.place_agent(a, pos=start_pos[i])

    def _get_obs(self) -> OrderedDict:
        local_obs = []
        for i in range(self.helper_grid.width):
            for j in range(self.helper_grid.height):
                o = self.helper_grid.get(i, j)
                if o is not None and o.type == "tree":
                    local_obs.append(o.state)
        for a in self.agents:
            local_obs.append(a.pos[0])
            local_obs.append(a.pos[1])
        local_obs = np.array(local_obs, dtype=np.int16)
        obs = OrderedDict({f"{a.index}": local_obs.copy() for a in self.agents})
        return obs

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        # zero out wildfire specific variables, if any
        self.burnt_trees = 0

        # reset the grid
        super().reset(seed=seed)

        obs = self._get_obs()

        info = {"burnt trees": self.burnt_trees}
        return obs, info

    def move_agent(self, i, next_cell, next_pos):
        if next_cell is None or next_cell.can_overlap():
            # Once reward function is decided, modify to add rewards here.
            self.grid.set(*next_pos, self.agents[i])
            self.grid.set(
                *self.agents[i].pos, self.helper_grid.get(*self.agents[i].pos)
            )
            self.agents[i].pos = next_pos
        else:
            # do nothing if next cell is a wall or another agent. Modify to add rewards here if needed.
            pass

    def neighbors_on_fire(self, i: int, j: int) -> int:
        """
        Args:
            i (int): first coordinate of tree position
            j (int): second coordinate of tree position
        Returns:
        int
            The number of neighboring trees on fire. A tree has upto 8 neighbors.
        """
        num = 0
        tree_pos = np.array([i, j])
        relative_pos = [
            np.array([1, 0]),
            np.array([-1, 0]),
            np.array([0, 1]),
            np.array([0, -1]),
        ]
        for r in relative_pos:
            neighbor_pos = tree_pos + r
            if neighbor_pos[0] >= 0 and neighbor_pos[0] < self.helper_grid.width:
                if neighbor_pos[1] >= 0 and neighbor_pos[1] < self.helper_grid.height:
                    o = self.helper_grid.get(*neighbor_pos)
                    if o is not None and o.type == "tree":
                        if o.state == 1:
                            num += 1
        return num

    def agent_above_tree(self, i: int, j: int) -> bool:
        """
        Args:
            i (int): first coordinate of tree position
            j (int): second coordinate of tree position
        Returns:
        bool
            True, if a UAV agent is located at (i,j), otherwise, False.
        """
        bool = False
        for a in range(self.num_agents):
            if np.array_equal(self.agents[a].pos, (i, j)):
                bool = True
                break
        return bool

    def step(self, actions):
        self.step_count += 1
        reward = 0
        # rewards = {}
        actions = [value for value in actions.values()]
        order = np.random.permutation(len(actions))

        done = False
        truncated = False

        for i in order:
            if actions[i] == self.actions.still:
                continue
            elif actions[i] == self.actions.north:
                next_pos = self.agents[i].north_pos()
                next_cell = self.grid.get(*next_pos)
                self.move_agent(i, next_cell, next_pos)
            elif actions[i] == self.actions.south:
                next_pos = self.agents[i].south_pos()
                next_cell = self.grid.get(*next_pos)
                self.move_agent(i, next_cell, next_pos)
            elif actions[i] == self.actions.east:
                next_pos = self.agents[i].east_pos()
                next_cell = self.grid.get(*next_pos)
                self.move_agent(i, next_cell, next_pos)
            elif actions[i] == self.actions.west:
                next_pos = self.agents[i].west_pos()
                next_cell = self.grid.get(*next_pos)
                self.move_agent(i, next_cell, next_pos)

        # Calculate reward.
        for a in self.agents:
            o = self.helper_grid.get(*a.pos)
            if (
                o is not None and o.type == "tree"
            ):  # this check is redundant. to be safe against future changes or oversight.
                if o.state == 1:
                    reward += 1
                else:
                    reward -= 0.1

        # Update tree states

        # Store number of neighboring trees on fire for each tree before updating tree states.
        on_fire_neighbors = np.zeros((self.helper_grid.width, self.helper_grid.height))
        for j in range(self.helper_grid.height):
            for i in range(self.helper_grid.width):
                on_fire_neighbors[i, j] = self.neighbors_on_fire(i, j)

        for j in range(self.helper_grid.height):
            for i in range(self.helper_grid.width):
                c = self.helper_grid.get(i, j)
                if c is not None and c.type == "tree":
                    # transition from healthy to on fire
                    if c.state == 0:
                        if (
                            np.random.rand()
                            < 1 - (1 - self.alpha) ** on_fire_neighbors[i, j]
                        ):
                            c.state = 1
                            c.color = STATE_IDX_TO_COLOR_WILDFIRE[c.state]
                            self.trees_on_fire += 1
                            # update self.grid if object at (i,j) is a tree
                            o = self.grid.get(i, j)
                            if o.type == "tree":
                                o.state = 1
                                o.color = STATE_IDX_TO_COLOR_WILDFIRE[o.state]
                    # transition from on fire to burnt
                    if c.state == 1:
                        if (
                            np.random.rand()
                            < 1
                            - self.beta
                            + self.agent_above_tree(i, j) * self.delta_beta
                        ):
                            c.state = 2
                            c.color = STATE_IDX_TO_COLOR_WILDFIRE[c.state]
                            self.burnt_trees += 1
                            self.trees_on_fire -= 1
                            # update self.grid if object at (i,j) is a tree
                            o = self.grid.get(i, j)
                            if o.type == "tree":
                                o.state = 2
                                o.color = STATE_IDX_TO_COLOR_WILDFIRE[o.state]

        if self.step_count >= self.max_steps:
            done = True
            truncated = True

        rewards = {f"{a.index}": reward for a in self.agents}

        next_obs = self._get_obs()
        info = {"burnt trees": self.burnt_trees}
        infos = {f"{a.index}": info for a in self.agents}
        return next_obs, rewards, done, truncated, infos

    def render(self, close=False, highlight=False, tile_size=TILE_PIXELS):
        """
        Render the whole-grid human view
        """

        if close:
            if self.window:
                self.window.close()
            return

        if self.render_mode == "human" and not self.window:
            self.window = Window("gym_multigrid")
            self.window.show(block=False)

        if highlight:
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

        # Render the grid with agents
        img = self.grid.render(
            tile_size,
            highlight_masks=highlight_masks if highlight else None,
            uncached_object_types=self.uncahed_object_types,
        )

        # Re-render the tiles containing agents to change background color. Agents are rendered in circular shape.
        for a in self.agents:
            img = render_agent_tile(img, a, self.helper_grid, self.world)

        if self.render_mode == "human":
            self.window.show_img(img)

        return img

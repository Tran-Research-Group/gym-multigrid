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
import numpy as np
from typing import Any


class WildfireEnv(MultiGridEnv):
    """
    Environment in which agents have to put out a fire
    """

    def __init__(
        self,
        alpha=0.2,
        beta=0.9,
        delta_beta=0.5,
        size=32,
        num_agents=2,
        agent_view_size=10,
        max_steps=10000,
        partial_obs=False,
        actions_set=WildfireActions,
        render_mode="rgb_array",
    ):
        self.alpha = alpha
        self.beta = beta
        self.delta_beta = delta_beta
        self.num_agents = num_agents
        self.agent_view_size = agent_view_size
        self.max_steps = max_steps
        self.world = WildfireWorld
        self.grid_size = size
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

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height, self.world)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        # Insert trees in grid as per initial conditions of wildfire
        trees_on_fire = [(15, 15), (16, 15), (15, 16), (16, 16)]
        num_healthy_trees = (
            self.grid_size**2
            - (2 * self.grid.width + 2 * (self.grid.height - 2))
            - len(trees_on_fire)
        )
        for pos in trees_on_fire:
            self.put_obj(Tree(self.world, STATE_TO_IDX_WILDFIRE["on fire"]), *pos)
        for _ in range(num_healthy_trees):
            self.place_obj(Tree(self.world, STATE_TO_IDX_WILDFIRE["healthy"]))

        # Helper grid is a work around for grid unable to store multiple objects at a single cell.
        # Helper grid is the grid without agents.
        self.helper_grid = self.grid.copy()

        # Place UAVs at start positions
        start_pos = [
            (1, 1),
            (self.grid.width - 2, self.grid.height - 2),
        ]  # change this if more agents are added
        for i, a in enumerate(self.agents):
            self.place_agent(a, pos=start_pos[i])

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        # zero out wildfire specific variables, if any

        # reset the grid
        obs = np.array(super().reset(seed=seed))
        info = {}
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
            np.array([1, 1]),
            np.array([-1, 1]),
            np.array([1, -1]),
            np.array([-1, -1]),
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

        order = np.random.permutation(len(actions))

        rewards = np.zeros(len(actions))
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
                    if c.state == 0:
                        # transition from healthy to on fire
                        if np.random.rand() < 1 - self.alpha ** on_fire_neighbors[i, j]:
                            c.state = 1
                            c.color = STATE_IDX_TO_COLOR_WILDFIRE[c.state]

                            # If self.grid doesn't contain an agent at (i,j), then update state of tree there.
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
                            # If self.grid doesn't contain an agent at (i,j), then update state of tree there.
                            o = self.grid.get(i, j)
                            if o.type == "tree":
                                o.state = 2
                                o.color = STATE_IDX_TO_COLOR_WILDFIRE[o.state]

        if self.step_count >= self.max_steps:
            done = True
            truncated = True

        next_obs = [
            self.grid.encode_for_agents(agent_pos=self.agents[i].pos)
            for i in range(len(self.agents))
        ]
        next_obs = np.array([self.world.normalize_obs * ob for ob in next_obs])
        info = {}
        return next_obs, rewards, done, truncated, info

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

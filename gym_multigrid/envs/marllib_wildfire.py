from marllib.envs.gym_multigrid.multigrid import MultiGridEnv
from marllib.envs.gym_multigrid.core.world import WildfireWorld
from marllib.envs.gym_multigrid.core.agent import WildfireActions, Agent
from marllib.envs.gym_multigrid.core.object import Tree
from marllib.envs.gym_multigrid.core.grid import Grid
from marllib.envs.gym_multigrid.core.constants import (
    STATE_TO_IDX_WILDFIRE,
    TILE_PIXELS,
    STATE_IDX_TO_COLOR_WILDFIRE,
    COLOR_NAMES,
)
from marllib.envs.gym_multigrid.utils.window import Window
from marllib.envs.gym_multigrid.utils.misc import (
    render_agent_tile,
    get_central_square_coordinates,
    render_rescue_tile,
    get_nxn_square_coordinates,
)
from collections import OrderedDict
from gym.spaces import Box, Dict, Discrete
import numpy as np
from typing import Any
import random


class WildfireEnv(MultiGridEnv):
    """
    Environment to simulate UAVs fighting a forest wildfire
    """

    def __init__(
        self,
        alpha=0.3,
        beta=0.9,
        delta_beta=0.5,
        size=12,
        num_agents=4,
        agent_view_size=10,
        initial_fire_size=2,
        max_steps=100,
        partial_obs=False,
        actions_set=WildfireActions,
        render_mode="rgb_array",
        reward_normalization=False,
        obs_normalization=False,
        cooperative_reward=False,
        log_selfish_region_metrics=False,
        selfish_region_xmin=None,
        selfish_region_xmax=None,
        selfish_region_ymin=None,
        selfish_region_ymax=None,
        two_initial_fires=False,
        search_and_rescue=False,
        num_rescues=None,
        irregular_shape=False,
    ):
        self.alpha = alpha
        self.beta = beta
        self.delta_beta = delta_beta
        self.num_agents = num_agents
        self.search_and_rescue = search_and_rescue
        self.irregular_shape = irregular_shape
        if search_and_rescue:
            self.obs_depth = (
                (self.num_agents - 1) + len(STATE_IDX_TO_COLOR_WILDFIRE) + 1
            )  # agent centered obs doesn't include agent's own position. +1 at end for people to rescue.
            self.cells_to_rescue = []
            self.num_rescues = num_rescues  # number of people to rescue
            self.num_rescued = 0  # number of people rescued
            self.time_to_rescue = np.zeros(len(self.cells_to_rescue) + 1)
        else:
            self.obs_depth = (self.num_agents - 1) + len(STATE_IDX_TO_COLOR_WILDFIRE)
        self.agent_view_size = agent_view_size
        self.max_steps = max_steps
        self.world = WildfireWorld
        self.grid_size = size
        self.grid_size_without_walls = size - 2
        self.initial_fire_size = initial_fire_size
        self.burnt_trees = 0
        self.burnt_tree_positions = []
        self.trees_on_fire = 0
        self.reward_normalization = reward_normalization  # Ensure correct rmin, rmax values are used in normalize reward method of WildfireEnv
        self.obs_normalization = obs_normalization  # Ensure correct omin, omax values are used in normalize observation method of WildfireEnv
        self.rmin = -1
        self.rmax = 0.5
        self.cooperative_reward = cooperative_reward
        self.two_initial_fires = two_initial_fires
        self.log_selfish_region_metrics = log_selfish_region_metrics
        if (
            self.log_selfish_region_metrics
        ):  # all selfish list elements are in ascending order of indices of selfish agents
            self.selfish_xmin = np.array(
                selfish_region_xmin
            )  # x-coordinate minimum of regions of interest for each selfish agent. List length = number of selfish agents.
            self.selfish_xmax = np.array(
                selfish_region_xmax
            )  # x-coordinate maximum of regions of interest for each selfish agent. List length = number of selfish agents.
            self.selfish_ymin = np.array(
                selfish_region_ymin
            )  # y-coordinate minimum of regions of interest for each selfish agent. List length = number of selfish agents.
            self.selfish_ymax = np.array(
                selfish_region_ymax
            )  # y-coordinate maximum of regions of interest for each selfish agent. List length = number of selfish agents.
            self.selfish_region_trees_on_fire = np.zeros(len(self.selfish_xmin))
            self.selfish_region_burnt_trees = np.zeros(len(self.selfish_xmin))
            self.selfish_region_size = (
                self.selfish_xmax - self.selfish_xmin + np.ones(self.num_agents)
            ) * (self.selfish_ymax - self.selfish_ymin + np.ones(self.num_agents))

        if self.cooperative_reward:
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
        else:
            if num_agents > 6:
                raise NotImplementedError("add more colors in constants.py")
            if num_agents == 5:
                agent_colors = [
                    "yellow",
                    "yellow",
                    "purple",
                    "purple",
                    "purple",
                ]
            if num_agents == 2:
                agent_colors = ["red", "blue"]
            agents = [
                Agent(
                    world=self.world,
                    index=i,
                    view_size=agent_view_size,
                    actions=actions_set,
                    color=agent_colors[i],
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
        self.observation_space = self._set_observation_space()
        self.action_space = Dict(
            {f"{a.index}": Discrete(len(self.actions)) for a in self.agents}
        )

    def _set_observation_space(self) -> Dict:
        low = np.full(self.obs_depth * (self.grid_size_without_walls**2), 0)
        high = np.full(self.obs_depth * (self.grid_size_without_walls**2), 1)
        if (
            self.partial_obs
        ):  # right now partial obs is not supported. Modify to shorten low and high arrays.
            observation_space = Dict(
                {
                    f"{a.index}": Box(
                        low=low,
                        high=high,
                        dtype=np.float32,
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
                        dtype=np.float32,
                    )
                    for a in self.agents
                }
            )

        return observation_space

    def _gen_grid(self, width, height, state=None):
        self.grid = Grid(width, height, self.world)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        # assign positions of trees on fire and agents. If state is not None, match them to provided state.
        start_pos = []
        if state is not None:
            state = state.reshape(
                (
                    self.grid_size_without_walls,
                    self.grid_size_without_walls,
                    self.obs_depth + 1,
                ),
                order="F",
            )
            trees_on_fire = []
            for i in range(self.grid_size_without_walls):
                for j in range(self.grid_size_without_walls):
                    if state[i, j, 1] == 1:
                        trees_on_fire.append((i + 1, j + 1))
                    for o in self.agents:
                        index = o.index
                        if state[i, j, len(STATE_IDX_TO_COLOR_WILDFIRE) + index] == 1:
                            start_pos.append((i + 1, j + 1))
        else:
            fire_square_center = (
                random.randint(
                    1 + ((self.initial_fire_size - 1) / 2),
                    self.grid_size_without_walls - ((self.initial_fire_size - 1) / 2),
                ),
                random.randint(
                    1 + ((self.initial_fire_size - 1) / 2),
                    self.grid_size_without_walls - ((self.initial_fire_size - 1) / 2),
                ),
            )
            trees_on_fire = get_nxn_square_coordinates(
                *fire_square_center,
                self.grid_size_without_walls,
                self.initial_fire_size,
            )
            for a in self.agents:  # random start positions for agents
                while True:
                    p = (
                        np.random.choice(range(1, self.grid.width - 1)),
                        np.random.choice(range(1, self.grid.height - 1)),
                    )
                    if p not in start_pos:
                        start_pos.append(p)
                        break

        # update counts of trees on fire and healthy trees
        self.trees_on_fire += self.initial_fire_size**2
        if self.log_selfish_region_metrics:
            for a in self.agents:
                for pos in trees_on_fire:
                    if self.in_selfish_region(pos[0], pos[1], a.index):
                        self.selfish_region_trees_on_fire[a.index] += 1

        num_healthy_trees = self.grid_size_without_walls**2 - len(trees_on_fire)

        # place on fire tree objects in grid
        for pos in trees_on_fire:
            self.put_obj(
                Tree(self.world, STATE_TO_IDX_WILDFIRE["on fire"]),
                int(pos[0]),
                int(pos[1]),
            )

        # place on fire trees in second region
        if state is None and self.two_initial_fires:
            self.trees_on_fire += 9
            num_healthy_trees -= 9
            cells_to_avoid = get_3x3_square_coordinates(
                *(trees_on_fire[0]), self.grid_size_without_walls
            )
            while True:
                fire_square_center = (
                    random.randint(2, self.grid_size_without_walls - 1),
                    random.randint(2, self.grid_size_without_walls - 1),
                )
                if fire_square_center not in cells_to_avoid:
                    break
            trees_on_fire_region2 = get_3x3_square_coordinates(
                *fire_square_center, self.grid_size_without_walls
            )
            if self.log_selfish_region_metrics:
                for a in self.agents:
                    for pos in trees_on_fire_region2:
                        if self.in_selfish_region(pos[0], pos[1], a.index):
                            self.selfish_region_trees_on_fire[a.index] += 1

            # place on fire tree objects in grid
            for pos in trees_on_fire_region2:
                self.put_obj(
                    Tree(self.world, STATE_TO_IDX_WILDFIRE["on fire"]),
                    int(pos[0]),
                    int(pos[1]),
                )

        # place healthy tree objects in grid
        for _ in range(num_healthy_trees):
            self.place_obj(Tree(self.world, STATE_TO_IDX_WILDFIRE["healthy"]))

        # Helper grid is a work around for grid unable to store multiple objects at a single cell.
        # Helper grid is the grid without agents.
        self.helper_grid = self.grid.copy()

        # Place UAVs at start positions
        for i, a in enumerate(self.agents):
            self.place_agent(a, pos=start_pos[i])

    def _get_obs(self, agent_pos, agent_index) -> OrderedDict:
        local_obs = np.zeros(
            (
                self.grid_size_without_walls,
                self.grid_size_without_walls,
                self.obs_depth,
            ),
            dtype=np.float32,
        )

        for i in range(self.helper_grid.width):
            for j in range(self.helper_grid.height):
                new_coords = [
                    i - agent_pos[0],
                    j - agent_pos[1],
                ]  # agent centered coords
                new_coords = [
                    new_coords[0] - 1,
                    new_coords[1] - 1,
                ]  # new coords in grid without walls. because obs dimensions are without walls.
                o = self.helper_grid.get(i, j)
                if new_coords[0] < 0:
                    new_coords[0] += self.grid_size_without_walls
                    # wrap around. assuming grid is square.
                if new_coords[1] < 0:
                    new_coords[1] += self.grid_size_without_walls
                if o is None:
                    continue
                elif o.type == "tree":
                    local_obs[new_coords[0], new_coords[1], o.state] = 1
                if self.search_and_rescue:
                    if (i, j) in self.cells_to_rescue:
                        local_obs[new_coords[0], new_coords[1], -1] = 1

        for o in self.agents:
            if o.index != agent_index:
                if o.index > agent_index:
                    id = o.index - 1
                else:
                    id = o.index

                new_coords = [
                    o.pos[0] - agent_pos[0],
                    o.pos[1] - agent_pos[1],
                ]  # agent centered coords
                new_coords = [
                    new_coords[0] - 1,
                    new_coords[1] - 1,
                ]  # new coords in grid without walls. because obs dimensions are without walls.

                local_obs[
                    new_coords[0],
                    new_coords[1],
                    len(STATE_IDX_TO_COLOR_WILDFIRE) + id,
                ] = 1

        if self.obs_normalization:
            raise NotImplementedError(
                "Observation normalization is not currently implemented because they are already normalized (1-hot encoded)."
            )
        # local_obs = np.array(local_obs, dtype=np.float32).reshape(-1)
        return local_obs.flatten("F")

    def get_state(self) -> OrderedDict:
        local_obs = np.zeros(
            (
                self.grid_size_without_walls,
                self.grid_size_without_walls,
                self.obs_depth + 1,
            ),
            dtype=np.float32,
        )

        for i in range(self.helper_grid.width):
            for j in range(self.helper_grid.height):
                o = self.helper_grid.get(i, j)
                if o is None:
                    continue
                elif o.type == "tree":
                    local_obs[i - 1, j - 1, o.state] = 1
                # search and rescue not implemented right now
                # if self.search_and_rescue:
                #     if (i, j) in self.cells_to_rescue:
                #         local_obs[new_coords[0], new_coords[1], -1] = 1

        for o in self.agents:
            local_obs[
                o.pos[0] - 1, o.pos[1] - 1, len(STATE_IDX_TO_COLOR_WILDFIRE) + o.index
            ] = 1

        if self.obs_normalization:
            raise NotImplementedError(
                "Observation normalization is not currently implemented because they are already normalized (1-hot encoded)."
            )
        # local_obs = np.array(local_obs, dtype=np.float32).reshape(-1)
        return local_obs.flatten("F")

    def get_state_interpretation(self, state):
        state = state.reshape(
            (
                self.grid_size_without_walls,
                self.grid_size_without_walls,
                self.obs_depth + 1,
            ),
            order="F",
        )
        print("-------------------------------------------------------------")
        print("Interpretable state:")
        for i in range(self.grid_size_without_walls):
            for j in range(self.grid_size_without_walls):
                if state[i, j, 1] == 1:
                    print(f"Tree at position {(i,j)} is on fire.")
                for o in self.agents:
                    index = o.index
                    if state[i, j, len(STATE_IDX_TO_COLOR_WILDFIRE) + index] == 1:
                        print(f"Agent {o.index} is at position {(i,j)}.")

    def construct_state(self, trees_on_fire, agent_pos):
        """
        Construct state from trees on fire and agent positions.
        Args:
            trees_on_fire (list): list of tuples containing positions of trees on fire. MUST be in grid without wall coordinates. Counting from 0 to grid_size_without_walls - 1.
            agent_pos (list): list of tuples containing positions of agents, in order of agent index. MUST be in grid without wall coordinates.
            Returns (ndarray): State representation of the environment.
        """
        state = np.zeros(
            (
                self.grid_size_without_walls,
                self.grid_size_without_walls,
                self.obs_depth + 1,
            ),
            dtype=np.float32,
        )
        for pos in trees_on_fire:
            state[pos[0], pos[1], 1] = 1
        for i, pos in enumerate(agent_pos):
            state[
                pos[0],
                pos[1],
                len(STATE_IDX_TO_COLOR_WILDFIRE) + i,
            ] = 1
        return state.flatten("F")

    def reset(self, seed=None, state=None):
        # zero out wildfire specific variables, if any
        self.burnt_trees = 0
        self.trees_on_fire = 0
        self.burnt_tree_positions = []
        if self.search_and_rescue:
            self.cells_to_rescue = []
            self.cells_to_rescue_chosen = False
            self.time_to_rescue = np.zeros(len(self.cells_to_rescue) + 1)
        if self.log_selfish_region_metrics:
            self.selfish_region_trees_on_fire = np.zeros(len(self.selfish_xmin))
            self.selfish_region_burnt_trees = np.zeros(len(self.selfish_xmin))

        # reset the grid
        if state is not None:
            super().reset(seed=seed, state=state)
        else:
            super().reset(seed=seed)

        obs = OrderedDict(
            {f"{a.index}": self._get_obs(a.pos, a.index) for a in self.agents}
        )

        info = {"burnt trees": self.burnt_trees}
        return obs, info

    def move_agent(self, i, next_cell, next_pos):
        if next_cell is None or next_cell.can_overlap():
            self.grid.set(*next_pos, self.agents[i])
            self.grid.set(
                *self.agents[i].pos, self.helper_grid.get(*self.agents[i].pos)
            )
            self.agents[i].pos = next_pos
        else:
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

    def in_selfish_region(self, i: int, j: int, agent_index: int) -> bool:
        """
        Args:
            i (int): first coordinate of tree position
            j (int): second coordinate of tree position
            agent_index (int): index of agent
        Returns:
        bool
            True, if tree at (i,j) is in the region of interest of the selfish agent, otherwise, False.
        """
        return (
            i >= self.selfish_xmin[agent_index]
            and i <= self.selfish_xmax[agent_index]
            and j >= self.selfish_ymin[agent_index]
            and j <= self.selfish_ymax[agent_index]
        )

    def step(self, actions):
        self.step_count += 1
        actions = [value for value in actions.values()]
        order = np.random.permutation(len(actions))

        done = False
        truncated = False

        # Move agents
        for i in order:
            if actions[i] == self.actions.still:
                continue
            if actions[i] == self.actions.north:
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
        rewards = {f"{a.index}": self._reward(a) for a in self.agents}

        # store number of neighboring trees on fire for each tree before updating tree states
        on_fire_neighbors = np.zeros((self.helper_grid.width, self.helper_grid.height))
        for j in range(self.helper_grid.height):
            for i in range(self.helper_grid.width):
                on_fire_neighbors[i, j] = self.neighbors_on_fire(i, j)

        # update tree states
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

                            # negative reward for tree transitioning to on fire state
                            if self.cooperative_reward:
                                for a in self.agents:
                                    rewards[f"{a.index}"] -= 0.5
                            else:
                                for a in self.agents:
                                    if self.in_selfish_region(i, j, a.index):
                                        rewards[f"{a.index}"] -= 0.5
                                    else:
                                        rewards[f"{a.index}"] -= 0.1
                                    # rewards[f"{a.index}"] -= 0.5

                            # update count of trees on fire
                            self.trees_on_fire += 1
                            if self.log_selfish_region_metrics:
                                for a in self.agents:
                                    if self.in_selfish_region(i, j, a.index):
                                        self.selfish_region_trees_on_fire[a.index] += 1

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

                            # update count of burnt trees and trees on fire
                            self.burnt_trees += 1
                            self.trees_on_fire -= 1
                            self.burnt_tree_positions.append((i, j))
                            if self.log_selfish_region_metrics:
                                for a in self.agents:
                                    if self.in_selfish_region(i, j, a.index):
                                        self.selfish_region_burnt_trees[a.index] += 1
                                        self.selfish_region_trees_on_fire[a.index] -= 1

                            # update self.grid if object at (i,j) is a tree
                            o = self.grid.get(i, j)
                            if o.type == "tree":
                                o.state = 2
                                o.color = STATE_IDX_TO_COLOR_WILDFIRE[o.state]

        if self.step_count >= self.max_steps:
            done = True
            truncated = True
            if self.search_and_rescue:
                self.num_rescued = (
                    self.num_rescues - len(self.cells_to_rescue)
                    if self.cells_to_rescue_chosen
                    else 0
                )  # zero if no rescue mission was initiated or if mission was initiated but no one was rescued.
        elif self.search_and_rescue:
            if self.burnt_trees >= self.num_rescues and not self.cells_to_rescue_chosen:
                possible_rescue_cells = self.burnt_tree_positions
                for a in self.agents:
                    if a.pos in self.burnt_tree_positions:
                        possible_rescue_cells.remove(a.pos)
                self.cells_to_rescue = random.sample(
                    possible_rescue_cells, self.num_rescues
                )
                self.time_to_rescue[0] = (
                    self.step_count
                )  # first element of self.time_to_rescue is the rescue mission start time
                self.cells_to_rescue_chosen = True

        next_obs = OrderedDict(
            {f"{a.index}": self._get_obs(a.pos, a.index) for a in self.agents}
        )
        info = {"burnt trees": self.burnt_trees}
        infos = {f"{a.index}": info for a in self.agents}
        return next_obs, rewards, done or truncated, infos

    def _reward(self, agent=None):
        reward = 0
        if self.cooperative_reward:
            if self.trees_on_fire > 0:
                for a in self.agents:
                    o = self.helper_grid.get(*a.pos)
                    if (
                        o is not None and o.type == "tree"
                    ):  # this check is redundant. to be safe against future changes or oversight.
                        if o.state == 1:
                            reward += 0.5
                        else:
                            pass
        else:
            if self.trees_on_fire > 0:
                o = self.helper_grid.get(*agent.pos)
                if (
                    o is not None and o.type == "tree"
                ):  # this check is redundant. to be safe against future changes or oversight.
                    if o.state == 1:
                        if self.in_selfish_region(
                            agent.pos[0], agent.pos[1], agent.index
                        ):
                            reward += 0.5
                        else:
                            reward += 0.1
                        # reward += 0.5
                    else:
                        pass

            if self.search_and_rescue:
                if tuple(agent.pos) in self.cells_to_rescue:
                    reward += 20
                    self.time_to_rescue = np.append(
                        self.time_to_rescue, self.step_count
                    )
                    self.cells_to_rescue.remove(tuple(agent.pos))

        if self.reward_normalization:
            reward = self._normalize_reward(reward, self.rmin, self.rmax)
        return reward

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

        if self.search_and_rescue:
            for cell in self.cells_to_rescue:
                img = render_rescue_tile(img, cell, self.helper_grid, self.world)

        if self.render_mode == "human":
            self.window.show_img(img)

        return img

    def _normalize_reward(self, rcurrent, rmin, rmax):
        """
        Normalize reward to be between 0 and 1.
        Args:
            reward (float): reward value
        Returns:
            float
                Normalized reward value
        """
        return (rcurrent - rmin) / (rmax - rmin)

    def _normalize_obs(self, obs, omin, omax):
        """
        Normalize observation to be between 0 and 1.
        Args:
            obs (ndarray): observation value
        Returns:
            ndarray
                Normalized observation value
        """
        return (obs - omin) / (omax - omin)

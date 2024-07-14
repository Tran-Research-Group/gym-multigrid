# pylint: disable=line-too-long, dangerous-default-value
"""Defines the WildfireEnv class, which simulates dynamics of unmanned aerial vehicles (UAVs) fighting a spreading wildfire
"""

from collections import OrderedDict
import random
from gymnasium.spaces import Box, Dict, Discrete
import numpy as np
from gym_multigrid.multigrid import MultiGridEnv
from gym_multigrid.core.world import WildfireWorld
from gym_multigrid.core.agent import WildfireActions, Agent
from gym_multigrid.core.object import Tree
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.constants import (
    STATE_TO_IDX_WILDFIRE,
    TILE_PIXELS,
    STATE_IDX_TO_COLOR_WILDFIRE,
    COLORS,
)
from gym_multigrid.utils.window import Window
from gym_multigrid.utils.misc import (
    render_agent_tiles,
    get_initial_fire_coordinates,
)


class WildfireEnv(MultiGridEnv):
    """Grid environment which simulates dynamics of unmanned aerial vehicles (UAVs) fighting a spreading wildfire"""

    def __init__(
        self,
        alpha=0.05,
        beta=0.99,
        delta_beta=0,
        size=17,
        num_agents=2,
        agent_start_positions=((1, 1), (15, 15)),
        agent_colors=("red", "blue"),
        agent_view_size=10,
        initial_fire_size=1,
        max_steps=100,
        partial_obs=False,
        actions_set=WildfireActions,
        render_mode="rgb_array",
        cooperative_reward=False,
        log_selfish_region_metrics=False,
        selfish_region_xmin=None,
        selfish_region_xmax=None,
        selfish_region_ymin=None,
        selfish_region_ymax=None,
        two_initial_fires=False,
    ):
        """Create a WildfireEnv environment

        Parameters
        ----------
        alpha : float, optional
            parameter for the wildfire dynamics model, by default 0.05
        beta : float, optional
            parameter for the wildfire dynamics model, by default 0.99
        delta_beta : float, optional
            parameter for the wildfire dynamics model, by default 0
        size : int, optional
            side of the square gridworld, by default 17
        num_agents : int, optional
            number of UAV agents, by default 2
        agent_start_positions : tuple[tuple[int,int]], optional
            tuple of tuples containing the start positions of the agents, in order of agent index. By default ((1, 1), (15, 15))
        agent_colors : tuple[str], optional
            tuple of strings of color names of all agents. The strings should be keys in the COLORS dictionary in constants.py. Only applicable if cooperative_reward is False. Fully cooperative agents have light_blue color by default. By default self-interested agents have red and blue colors
        agent_view_size : int, optional
            side of the square region visible to an agent with partial observability, by default 10. Only applicable if partial_obs is True
        initial_fire_size : int, optional
            side of the square shaped initial fire region, by default 1
        max_steps : int, optional
            maximum number of steps in an episode, by default 100
        partial_obs : bool, optional
            whether agents have partial observability, by default False
        actions_set : WildfireActions, optional
            action space of the agents. All agents have the same action space. By default WildfireActions.
        render_mode : str, optional
            mode of rendering the environment, by default "rgb_array"
        cooperative_reward : bool, optional
            whether the agents use a cooperative reward, by default False. If True, the agents are fully cooperative and receive the same reward.
        log_selfish_region_metrics : bool, optional
            whether to log metrics related to trees in selfish regions, by default False
        selfish_region_xmin : list, optional
            list containing x-coordinates of the leftmost boundaries of the regions of selfish interest for the agents. Regions of selfish interest are rectangular. List elements are in order of agent indices. Only applicable if cooperative_reward is False. By default None
        selfish_region_xmax : list, optional
            list containing x-coordinates of the rightmost boundaries of the regions of selfish interest for the agents. Regions of selfish interest are rectangular. List elements are in order of agent indices. Only applicable if cooperative_reward is False. By default None
        selfish_region_ymin : list, optional
            list containing y-coordinates of the topmost boundaries of the regions of selfish interest for the agents. Regions of selfish interest are rectangular. List elements are in order of agent indices. Only applicable if cooperative_reward is False. By default None
        selfish_region_ymax : list, optional
            list containing y-coordinates of the bottom-most boundaries of the regions of selfish interest for the agents. Regions of selfish interest are rectangular. List elements are in order of agent indices. Only applicable if cooperative_reward is False. By default None
        two_initial_fires : bool, optional
            whether the initial state has two disjoint square fire regions, by default False
        """
        self.alpha = alpha
        self.beta = beta
        self.delta_beta = delta_beta
        self.num_agents = num_agents
        self.agent_start_positions = agent_start_positions
        # observation vector of each agent is concatenation of one-hot encodings of tree states and other agents' positions. len(STATE_IDX_TO_COLOR_WILDFIRE) = the number of tree states
        self.obs_depth = (self.num_agents - 1) + len(STATE_IDX_TO_COLOR_WILDFIRE)
        self.max_steps = max_steps
        self.world = WildfireWorld
        self.grid_size = size
        self.grid_size_without_walls = size - 2
        self.initial_fire_size = initial_fire_size
        self.burnt_trees = 0
        self.unburnt_trees = []
        self.trees_on_fire = 0
        self.cooperative_reward = cooperative_reward
        self.two_initial_fires = two_initial_fires
        self.log_selfish_region_metrics = log_selfish_region_metrics
        if self.log_selfish_region_metrics:
            # initialize attributes for logging metrics related to trees in selfish regions
            self.selfish_xmin = np.array(selfish_region_xmin)
            self.selfish_xmax = np.array(selfish_region_xmax)
            self.selfish_ymin = np.array(selfish_region_ymin)
            self.selfish_ymax = np.array(selfish_region_ymax)
            self.selfish_region_trees_on_fire = np.zeros(len(self.selfish_xmin))
            self.selfish_region_burnt_trees = np.zeros(len(self.selfish_xmin))
            self.selfish_region_size = (
                self.selfish_xmax
                - self.selfish_xmin
                + np.ones(len(selfish_region_xmin))
            ) * (
                self.selfish_ymax
                - self.selfish_ymin
                + np.ones(len(selfish_region_ymin))
            )

        if self.cooperative_reward:
            # initialize cooperative agents
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
            # initialize self-interested agents with different colors
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
        self.observation_space: Box | Dict = self._set_observation_space()
        self.action_space = Dict(
            {f"{a.index}": Discrete(len(self.actions)) for a in self.agents}
        )

    def _set_observation_space(self) -> Dict:
        """Set the observation space for each agent in the environment. All agents possess the same observation space

        Returns
        -------
        observation_space : dict
            dictionary where each key is an agent's index and the value is the observation space
            for that agent
        """
        # observation vector of agent is the concatenation of obs_depth number of one-hot encodings where each encoding has grid_size_without_walls number of elements valued either 0 or 1. Additionally, the observation vector contains the normalized time step at the end
        low = np.full(self.obs_depth * (self.grid_size_without_walls**2) + 1, 0)
        high = np.full(self.obs_depth * (self.grid_size_without_walls**2) + 1, 1)
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
        """Generate the grid for the environment

        Parameters
        ----------
        width : int
            width of the grid
        height : int
            height of the grid
        state : ndarray, optional
            specifies the initial state of the environment, by default None.
            If none, it is chosen uniformly at random from the assumed initial state distribution
        """
        self.grid = Grid(width, height, self.world)

        # generate the walls of the grid
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        agent_start_pos = []
        if state is not None:
            # store positions of agents and trees on fire as per the specified initial state.
            state = state[:-1].reshape(
                (
                    self.obs_depth + 1,
                    self.grid_size_without_walls,
                    self.grid_size_without_walls,
                ),
            )
            initial_fire = []
            for i in range(self.grid_size_without_walls):
                for j in range(self.grid_size_without_walls):
                    if state[1, j, i]:
                        # increment by 1 is to convert positions to grid with wall coordinates. i and j are swapped because the element at first index of an 2d-array specifies the row, while the x-coordinate in gridworld positions specifies the column.
                        initial_fire.append((i + 1, j + 1))
                    for o in self.agents:
                        index = o.index
                        if state[len(STATE_IDX_TO_COLOR_WILDFIRE) + index, j, i]:
                            # increment by 1 is to convert positions to grid with wall coordinates. i and j are swapped because the element at first index of an 2d-array specifies the row, while the x-coordinate in gridworld positions specifies the column.
                            agent_start_pos.append((i + 1, j + 1))

        else:
            # choose location of initial fire uniformly at random
            if self.initial_fire_size % 2 == 0:
                # for even sized initial fires, choose top left corner of fire region uniformly at random
                top_left_corner = (
                    random.randint(
                        1,
                        self.grid_size_without_walls - (self.initial_fire_size),
                    ),
                    random.randint(
                        1,
                        self.grid_size_without_walls - (self.initial_fire_size),
                    ),
                )
                initial_fire = get_initial_fire_coordinates(
                    *top_left_corner,
                    self.grid_size_without_walls,
                    self.initial_fire_size,
                )
            else:
                # for odd sized initial fires, choose center of fire region uniformly at random
                fire_square_center = (
                    random.randint(
                        1 + ((self.initial_fire_size - 1) / 2),
                        self.grid_size_without_walls
                        - ((self.initial_fire_size - 1) / 2),
                    ),
                    random.randint(
                        1 + ((self.initial_fire_size - 1) / 2),
                        self.grid_size_without_walls
                        - ((self.initial_fire_size - 1) / 2),
                    ),
                )
                initial_fire = get_initial_fire_coordinates(
                    *fire_square_center,
                    self.grid_size_without_walls,
                    self.initial_fire_size,
                )
            # agent_start_pos is specified during environment initialization
            agent_start_pos = self.agent_start_positions

        for pos in initial_fire:
            region = "common"
            if self.log_selfish_region_metrics:
                # update count of trees on fire in selfish regions
                for a in self.agents:
                    if self.in_selfish_region(pos[0], pos[1], a.index):
                        # selfish region is identified by the lowest index among indices of the corresponding selfish agent(s)
                        region = f"{a.index}"
                        self.selfish_region_trees_on_fire[a.index] += 1
                        break
            # insert tree on fire in grid
            self.put_obj(
                Tree(self.world, STATE_TO_IDX_WILDFIRE["on fire"], region=region),
                int(pos[0]),
                int(pos[1]),
            )

        # update counts of trees on fire and healthy trees
        self.trees_on_fire += self.initial_fire_size**2
        num_healthy_trees = self.grid_size_without_walls**2 - len(initial_fire)

        # insert healthy tree in grid
        for _ in range(num_healthy_trees):
            tree_obj = Tree(self.world, STATE_TO_IDX_WILDFIRE["healthy"])
            self.place_obj(tree_obj)
            if self.log_selfish_region_metrics:
                # check if tree is in a selfish region, and update region attribute of tree if it is
                for a in self.agents:
                    if self.in_selfish_region(*(tree_obj.pos), a.index):
                        tree_obj.region = f"{a.index}"
                        break

        # helper grid is a work around for grid being unable to store multiple objects at a single cell. It does not contain agents
        self.helper_grid = self.grid.copy()

        # create list of unburnt trees. initial state does not have burnt trees.
        for c in self.helper_grid.grid:
            if c is not None and c.type == "tree":
                self.unburnt_trees.append(c)

        # insert agents in grid
        for i, a in enumerate(self.agents):
            self.place_agent(a, pos=agent_start_pos[i])
            self.helper_grid.get(*agent_start_pos[i]).agent_above = True

    def _get_obs(self) -> list[np.typing.NDArray]:
        """Get observation vectors of all agents in the environment

        Returns
        -------
        agent_obs: list(ndarray)
            list of agent observations where the element at i^th list index is the observation vector for the agent with index 'i'
        """
        # initialize list of observation vector of each agent
        agent_obs = [
            np.zeros(
                (
                    self.obs_depth,
                    self.grid_size_without_walls,
                    self.grid_size_without_walls,
                ),
                dtype=np.float32,
            )
            for _ in range(self.num_agents)
        ]

        # update tree states in agent observations
        for obj in self.helper_grid.grid:
            if obj is None or obj.type == "wall":
                # agent observations do not contain walls
                continue
            i, j = obj.pos
            for a in self.agents:
                # convert to agent centered coordinates
                nc = [i - a.pos[0], j - a.pos[1]]
                # wrap around to get agent centered toroidal coordinates
                if nc[0] < 0:
                    nc[0] += self.grid_size_without_walls
                if nc[1] < 0:
                    nc[1] += self.grid_size_without_walls
                # update agent's observation
                agent_obs[a.index][obj.state, nc[1], nc[0]] = 1

        # for each agent, update other agents' positions in agent observations
        for a in self.agents:
            for o in self.agents:
                if o.index != a.index:
                    if o.index > a.index:
                        idx = o.index - 1
                    else:
                        idx = o.index
                    # convert to agent centered coordinates
                    nc = [
                        o.pos[0] - a.pos[0],
                        o.pos[1] - a.pos[1],
                    ]
                    # wrap around to get agent centered toroidal coordinates
                    if nc[0] < 0:
                        nc[0] += self.grid_size_without_walls
                    if nc[1] < 0:
                        nc[1] += self.grid_size_without_walls
                    agent_obs[a.index][
                        len(STATE_IDX_TO_COLOR_WILDFIRE) + idx,
                        nc[1],
                        nc[0],
                    ] = 1

        # flatten, and append normalized time step at the end of, each agent observation
        for a in self.agents:
            agent_obs[a.index] = np.append(
                agent_obs[a.index].flatten(),
                np.array(self.step_count / self.max_steps, dtype=np.float32),
            )
        return agent_obs

    def get_state(self):
        """Get the state representation of the environment.

        Returns
        -------
        ndarray
            state representation of the environment
        """
        s = np.zeros(
            (
                self.obs_depth + 1,
                self.grid_size_without_walls,
                self.grid_size_without_walls,
            ),
            dtype=np.float32,
        )

        # update tree states in state representation
        for o in self.helper_grid.grid:
            if o.type == "tree":
                # state representation does not contain walls, so decrement by 1 to convert positions to grid without wall coordinates
                s[o.state, o.pos[1] - 1, o.pos[0] - 1] = 1

        # update agent positions in state representation
        for a in self.agents:
            # state representation does not contain walls, so decrement by 1 to convert positions to grid without wall coordinates
            s[
                len(STATE_IDX_TO_COLOR_WILDFIRE) + a.index, a.pos[1] - 1, a.pos[0] - 1
            ] = 1

        # flatten, and append normalized time step at the end of, state representation
        s = np.append(
            s.flatten(),
            np.array(self.step_count / self.max_steps, dtype=np.float32),
        )
        return s

    def get_state_interpretation(self, state, print_interpretation=True):
        """Get human readable interpretation of the state of the environment

        Parameters
        ----------
        state : ndarray
            state representation of the environment
        print_interpretation : bool, optional
           whether to print the interpretation of the state, by default True. Human readable interpretation refers to printing the positions of trees on fire, agents, and the time step.

        Returns
        -------
        on_fire_trees : list[tuple[int,int]]
            list of tuples containing position coordinates (x,y) of trees on fire. Coordinates are in grid without wall coordinates
        time_step : float
            normalized time step of the episode at which time the state was recorded
        """
        time_step = state[-1]
        state = state[:-1].reshape(
            (
                self.obs_depth + 1,
                self.grid_size_without_walls,
                self.grid_size_without_walls,
            ),
        )
        if print_interpretation:
            print("-------------------------------------------------------------")
            print("Interpretable state in grid without wall coordinates:")
        on_fire_trees = []
        for i in range(self.grid_size_without_walls):
            for j in range(self.grid_size_without_walls):
                if state[1, j, i] == 1:
                    if print_interpretation:
                        print(f"Tree at position {(i,j)} is on fire.")
                    on_fire_trees.append((i, j))
                for o in self.agents:
                    index = o.index
                    if state[len(STATE_IDX_TO_COLOR_WILDFIRE) + index, j, i] == 1:
                        if print_interpretation:
                            print(f"Agent {o.index} is at position {(i,j)}.")
        if print_interpretation:
            print(f"Time step: {time_step}")
            print("-------------------------------------------------------------")
        return on_fire_trees, time_step

    def construct_state(self, trees_on_fire, agent_pos, time_step: int):
        """Construct the state representation vector of the environment for given positions of trees on fire and agents

        Parameters
        ----------
        trees_on_fire : list
            list of tuples containing position coordinates (x,y) of trees on fire. Coordinates should be in grid without wall coordinates
        agent_pos : list
            list of tuples containing position coordinates (x,y) of agents, in order of agent index. Coordinates should be in grid without wall coordinates
        time_step : int
            normalized time step of the episode at which time the state was recorded

        Returns
        -------
        state : ndarray
            state representation of the environment
        """
        state = np.zeros(
            (
                self.obs_depth + 1,
                self.grid_size_without_walls,
                self.grid_size_without_walls,
            ),
            dtype=np.float32,
        )
        # update tree states in state representation. There are no burnt trees in the state
        state[0, :, :] = 1
        for pos in trees_on_fire:
            state[1, pos[1], pos[0]] = 1
            state[0, pos[1], pos[0]] = 0

        # update agent positions in state representation
        for i, pos in enumerate(agent_pos):
            state[
                len(STATE_IDX_TO_COLOR_WILDFIRE) + i,
                pos[1],
                pos[0],
            ] = 1

        # flatten, and append normalized time step at the end of, state representation
        state = np.append(
            state.flatten(),
            np.array(time_step / self.max_steps, dtype=np.float32),
        )
        return state

    def reset(self, seed: int | None = None, state=None):
        """Reset the state of the environment

        Parameters
        ----------
        seed : int, optional
            seed for random number generator, by default None
        state : ndarray, optional
            specifies the initial state of the environment upon reset, by default None.
            If none, initial state is chosen uniformly at random from initial state distribution.

        Returns
        -------
        obs : OrderedDict
            dictionary where each key is the agent index and the value is the observation vector for that agent.
        info : dict
            dictionary containing additional information about the environment.
            Here, it contains the number of burnt trees in the environment after reset
        """
        # reset environment attributes
        self.burnt_trees = 0
        self.trees_on_fire = 0
        self.unburnt_trees = []
        if self.log_selfish_region_metrics:
            self.selfish_region_trees_on_fire = np.zeros(len(self.selfish_xmin))
            self.selfish_region_burnt_trees = np.zeros(len(self.selfish_xmin))

        # reset the grid
        if state is not None:
            super().reset(seed=seed, state=state)
        else:
            super().reset(seed=seed)

        # get agent observations
        agent_obs = self._get_obs()
        obs = OrderedDict({f"{a.index}": agent_obs[a.index] for a in self.agents})

        # create info dictionary
        info = {"burnt trees": self.burnt_trees}
        return obs, info

    def move_agent(self, i, next_pos):
        """Move agent to a new position in the grid

        Parameters
        ----------
        i : int
            index of agent to be moved
        next_pos : tuple[int, int]
            coordinates of new position
        """
        # add agent to grid in new position
        self.grid.set(*next_pos, self.agents[i])

        # get tree in agent's old position from helper grid and add tree to grid
        tree = self.helper_grid.get(*self.agents[i].pos)
        tree.agent_above = False
        self.grid.set(*self.agents[i].pos, tree)

        # update attributes
        next_tree = self.helper_grid.get(*next_pos)
        next_tree.agent_above = True
        self.agents[i].pos = next_pos

    def neighbors_on_fire(self, tree_pos) -> int:
        """Get the number of neighboring trees on fire for a given tree.
           Neighbors are adjacent trees in the cardinal directions. A tree can have at most 4 neighbors

        Parameters
        ----------
        tree_pos : tuple[int, int]
            position coordinates of the tree whose neighbors are to be checked

        Returns
        -------
        num : int
            the number of neighboring trees on fire
        """
        num = 0
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

    def in_selfish_region(self, i: int, j: int, agent_index: int) -> bool:
        """Check if given tree is in region of selfish interest of given agent

        Parameters
        ----------
        i : int
            x-coordinate of tree position
        j : int
            y-coordinate of tree position
        agent_index : int
            index of selfish agent

        Returns
        -------
        bool
            True, if tree is in the region of selfish interest of the given agent. Otherwise, False
        """
        return (
            i >= self.selfish_xmin[agent_index]
            and i <= self.selfish_xmax[agent_index]
            and j >= self.selfish_ymin[agent_index]
            and j <= self.selfish_ymax[agent_index]
        )

    def step(self, actions):
        """Take a step in the environment. Wildfire dynamics are propagated by one time step, and agents move according to their actions.

        Parameters
        ----------
        actions : dict
            dictionary containing actions for each agent. The key is the agent index and the value is the action.

        Returns
        -------
        next_obs : OrderedDict
            dictionary where each key is the agent index and the value is the new observation vector (ndarray) for that agent after the environment step.
        rewards : dict
            dictionary where each key is the agent index and the value is the reward given to that agent after the environment step.
        terminated : bool
            True, if the episode is done, otherwise, False. Episode is done if maximum number of steps is reached or there are zero trees on fire.
        info : dict
            dictionary where each key is the agent index and the value is an info dictionary containing additional information about the environment. Here, each agent's info dictionary contains the same information, viz., the number of burnt trees.
        """
        self.step_count += 1
        actions = [value for value in actions.values()]
        terminated = False
        truncated = False

        # Move agents sequentially, in random order
        order = np.random.permutation(len(actions))
        for i in order:
            if actions[i] == self.actions.still:
                continue
            if actions[i] == self.actions.north:
                next_pos = self.agents[i].north_pos()
                next_cell = self.grid.get(*next_pos)
                if next_cell is None or next_cell.can_overlap():
                    self.move_agent(i, next_pos)
            elif actions[i] == self.actions.south:
                next_pos = self.agents[i].south_pos()
                next_cell = self.grid.get(*next_pos)
                if next_cell is None or next_cell.can_overlap():
                    self.move_agent(i, next_pos)
            elif actions[i] == self.actions.east:
                next_pos = self.agents[i].east_pos()
                next_cell = self.grid.get(*next_pos)
                if next_cell is None or next_cell.can_overlap():
                    self.move_agent(i, next_pos)
            elif actions[i] == self.actions.west:
                next_pos = self.agents[i].west_pos()
                next_cell = self.grid.get(*next_pos)
                if next_cell is None or next_cell.can_overlap():
                    self.move_agent(i, next_pos)

        # rewards for moving over tree on fire
        agent_rewards = self.partial_reward()

        # propagate wildfire dynamics by one time step
        # initialize lists to store trees transitioning to on fire and burnt state in the current time step
        trees_to_fire_state = []
        trees_to_burnt_state = []
        # loop over all unburnt trees. Burnt trees remain burnt
        for c in self.unburnt_trees:
            if c.state == 0:
                pos = np.array(c.pos)
                # transition from healthy to on fire with probability 1 - (1 - alpha)^n
                if np.random.rand() < 1 - (1 - self.alpha) ** self.neighbors_on_fire(
                    pos
                ):
                    # update relevant attributes and list
                    trees_to_fire_state.append(c)
                    self.trees_on_fire += 1
                    if self.log_selfish_region_metrics:
                        if c.region != "common":
                            self.selfish_region_trees_on_fire[int(c.region)] += 1
            if c.state == 1:
                # transition from on fire to burnt with probability 1 - beta + delta_beta * agent_above
                if np.random.rand() < 1 - self.beta + c.agent_above * self.delta_beta:
                    # update relevant attributes and list
                    trees_to_burnt_state.append(c)
                    self.burnt_trees += 1
                    self.trees_on_fire -= 1
                    if self.log_selfish_region_metrics:
                        if c.region != "common":
                            self.selfish_region_burnt_trees[int(c.region)] += 1
                            self.selfish_region_trees_on_fire[int(c.region)] -= 1

        # update tree objects in helper grid and grid. This update is done after the loop to avoid affecting
        # the transition probabilities of trees later in the loop due to trees that have already transitioned
        # earlier in the loop
        for c in trees_to_fire_state:
            c.state = 1
            c.color = STATE_IDX_TO_COLOR_WILDFIRE[c.state]
            # update tree in grid
            o = self.grid.get(c.pos[0], c.pos[1])
            if o.type == "tree":
                o.state = 1
                o.color = STATE_IDX_TO_COLOR_WILDFIRE[o.state]
        for c in trees_to_burnt_state:
            self.unburnt_trees.remove(c)
            c.state = 2
            c.color = STATE_IDX_TO_COLOR_WILDFIRE[c.state]
            # update tree in grid
            o = self.grid.get(c.pos[0], c.pos[1])
            if o.type == "tree":
                o.state = 2
                o.color = STATE_IDX_TO_COLOR_WILDFIRE[o.state]

        # include rewards due to trees on fire
        if self.cooperative_reward:
            agent_rewards -= 0.5 * self.trees_on_fire
        else:
            for a in self.agents:
                agent_rewards[a.index] -= 0.5 * self.selfish_region_trees_on_fire[
                    a.index
                ] + 0.1 * (
                    self.trees_on_fire - self.selfish_region_trees_on_fire[a.index]
                )
        # rewards dictionary
        rewards = {f"{a.index}": agent_rewards[a.index] for a in self.agents}

        # check if episode is done
        if self.trees_on_fire == 0:
            terminated = True
        elif self.step_count >= self.max_steps:
            truncated = True

        # get agent observations after the environment step
        agent_obs = self._get_obs()
        next_obs = OrderedDict({f"{a.index}": agent_obs[a.index] for a in self.agents})

        # info dictionary
        info = {"burnt trees": self.burnt_trees}
        infos = {f"{a.index}": info for a in self.agents}

        return next_obs, rewards, terminated, truncated, infos

    def partial_reward(self):
        """Compute partial reward arising from being over trees on fire, for each agent

        Returns
        -------
        reward: ndarray[float]
            reward for each agent
        """
        if self.cooperative_reward:
            reward = 0.0
            # non-zero reward possible only if there are trees on fire
            if self.trees_on_fire > 0:
                for a in self.agents:
                    o = self.helper_grid.get(*a.pos)
                    # check if agent is over a tree on fire
                    if o.state == 1:
                        reward += 0.5
            return np.array([reward for _ in range(self.num_agents)])
        reward = [0.0 for _ in range(self.num_agents)]
        # non-zero reward possible only if there are trees on fire
        if self.trees_on_fire > 0:
            for agent in self.agents:
                o = self.helper_grid.get(*agent.pos)
                # check if agent is over a tree on fire
                if o.state == 1:
                    reward[agent.index] += 0.5 if o.region == f"{agent.index}" else 0.1
            return np.array(reward)
        return reward

    def render(self, close=False, highlight=False, tile_size=TILE_PIXELS):
        """Render the whole-grid human view

        Parameters
        ----------
        close : bool, optional
            close the rendering window, by default False. Only applicable if render_mode is "human"
        highlight : bool, optional
            highlight the cells visible to the agent, by default False
        tile_size : int, optional
            size of each tile in pixels, by default TILE_PIXELS (defined in constants.py)

        Returns
        -------
        img : ndarray
            image of the grid
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

        # Render the grid
        if self.cooperative_reward:
            img = self.grid.render(
                tile_size,
                highlight_masks=highlight_masks if highlight else None,
                uncached_object_types=self.uncahed_object_types,
            )
        else:
            # include selfish region boundaries in render
            colors = [COLORS[a.color] for a in self.agents]
            img = self.grid.render(
                tile_size,
                highlight_masks=highlight_masks if highlight else None,
                uncached_object_types=self.uncahed_object_types,
                x_min=self.selfish_xmin,
                y_min=self.selfish_ymin,
                x_max=self.selfish_xmax,
                y_max=self.selfish_ymax,
                colors=colors,
            )

        # Re-render the tiles containing agents to include trees below agent
        if self.cooperative_reward:
            for a in self.agents:
                img = render_agent_tiles(img, a, self.helper_grid, self.world)
        else:
            # include selfish region boundaries in render
            for a in self.agents:
                img = render_agent_tiles(
                    img,
                    a,
                    self.helper_grid,
                    self.world,
                    x_min=self.selfish_xmin,
                    y_min=self.selfish_ymin,
                    x_max=self.selfish_xmax,
                    y_max=self.selfish_ymax,
                    colors=colors,
                )

        if self.render_mode == "human":
            self.window.show_img(img)

        return img

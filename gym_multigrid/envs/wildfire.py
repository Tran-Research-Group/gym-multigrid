from gym_multigrid.multigrid import MultiGridEnv
from gym_multigrid.core.world import WildfireWorld
from gym_multigrid.core.agent import WildfireActions, Agent
from gym_multigrid.core.object import Tree
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.constants import STATE_TO_IDX_WILDFIRE
import numpy as np


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

        # Insert trees to grid as per initial conditions of wildfire
        for _ in range(self.grid_size**2):
            self.place_obj(Tree(self.world, STATE_TO_IDX_WILDFIRE["healthy"]))

        # Randomize the UAV start position
        for a in self.agents:
            self.place_agent(a)

    def reset(self, seed: int | None = None):
        # zero out wildfire specific variables, if any

        # reset the grid
        obs = super().reset(seed=seed)
        return obs

    def move_agent(self, i, next_cell, next_pos):
        if next_cell.can_overlap():
            # Once reward function is decided, modify to add rewards here.
            self.grid.set(*next_pos, self.agents[i])
            self.grid.set(*self.agents[i].pos, None)
            self.agents[i].pos = next_pos
        else:
            # do nothing if next cell is a Wall. Modify to add rewards here if needed.
            pass

    def neighbors_on_fire(self, i: int, j: int) -> int:
        """
        Args:
            i (int): first coordinate of tree position
            j (int): second coordinate of tree position
        Output:
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
            if neighbor_pos[0] >= 0 and neighbor_pos[0] < self.grid.width:
                if neighbor_pos[1] >= 0 and neighbor_pos[1] < self.grid.height:
                    o = self.grid.get(*neighbor_pos)
                    if o.type == "tree":
                        if o.state == 1:
                            num += 1
        return num

    def agent_above_tree(self, i: int, j: int) -> bool:
        """
        Args:
            i (int): first coordinate of tree position
            j (int): second coordinate of tree position
        Output:
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

        for j in range(self.grid.height):
            for i in range(self.grid.width):
                c = self.grid.get(i, j)

                if c.type == "tree":
                    if c.state == 0:
                        if np.random.rand() < 1 - self.alpha ** self.neighbors_on_fire(
                            i, j
                        ):
                            c.state == 1
                    if c.state == 1:
                        if (
                            np.random.rand()
                            < 1
                            - self.beta
                            + self.agent_above_tree(i, j) * self.delta_beta
                        ):
                            c.state == 2

        next_obs = [
            self.grid.encode_for_agents(agent_pos=self.agents[i].pos)
            for i in range(len(self.agents))
        ]
        next_obs = [self.world.normalize_obs * ob for ob in next_obs]
        info = {}
        return next_obs, rewards, done, truncated, info

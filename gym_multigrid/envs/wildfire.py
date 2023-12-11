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
        size=30,
        num_agents=2,
        agent_view_size=10,
        max_steps=10000,
        partial_obs=False,
        actions_set=WildfireActions,
        render_mode="rgb_array",
    ):
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

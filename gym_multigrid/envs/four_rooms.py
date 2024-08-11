from gym_multigrid.multigrid import MultiGridEnv
from gym_multigrid.core.world import DefaultWorld
from gym_multigrid.core.agent import Agent
from gym_multigrid.core.object import Goal
from gym_multigrid.core.grid import Grid
import numpy as np

class FourRoomsEnv(MultiGridEnv):
    """
    Environment with four connected rooms where agents navigate to reach goals.
    """

    def __init__(
        self,
        size=15,
        view_size=5,
        width=None,
        height=None,
        goal_pos=None,
        agent_start_pos=None,
        max_steps=1000,
        see_through_walls=False,
    ):
        self.goal_pos = goal_pos or [[1, 1]]
        self.agent_start_pos = agent_start_pos or [[3, 3]]
        self.world = DefaultWorld

        agents = [Agent(self.world, i, view_size=view_size) for i in range(len(self.agent_start_pos))]

        super().__init__(
            grid_size=size,
            width=width,
            height=height,
            max_steps=max_steps,
            see_through_walls=see_through_walls,
            agents=agents,
            agent_view_size=view_size,
        )

    def _gen_grid(self, width, height):
        """
        Generates the grid for the environment, creating four rooms with doors connecting them.
        """
        self.grid = Grid(width, height, self.world)

        self.grid.horz_wall(0, height // 2)  
        self.grid.horz_wall(0, 0)            
        self.grid.horz_wall(0, height - 1)   

        self.grid.vert_wall(0, 0)        
        self.grid.vert_wall(width - 1, 0)   
        self.grid.vert_wall(width // 2, 0)  

        self.grid.set(width // 2, height // 4, None)
        self.grid.set(width // 2, 3 * height // 4, None)
        self.grid.set(width // 4, height // 2, None)
        self.grid.set(3 * width // 4, height // 2, None)

        for i, pos in enumerate(self.goal_pos):
            self.place_obj(Goal(self.world, index=i), top=pos, size=[1, 1])

        for i, (agent, pos) in enumerate(zip(self.agents, self.agent_start_pos)):
            self.place_agent(agent, top=pos)

    def step(self, actions):
        obs, rewards, done, info = MultiGridEnv.step(self, actions)
        return obs, rewards, done, info
    
    def _reward(self, i, rewards, reward=1):
        for i, agent in enumerate(self.agents):
            if np.array_equal(agent.cur_pos, self.goal_pos[i]):
                rewards[i] += reward

    def step(self, actions):
        obs, rewards, done, info = super().step(self, actions)

        for i, agent in enumerate(self.agents):
            if np.array_equal(agent.cur_pos, self.goal_pos[i]):
                rewards[i] += 1
                done = True

        return obs, rewards, done, info


class FourRoomsEnv10x10(FourRoomsEnv):
    def __init__(self):
        super().__init__(
            size=None,
            width=10,
            height=10,
            goal_pos=[[8, 8]],
            agent_start_pos=[[1, 1]],
            max_steps=1000,
            see_through_walls=False,
        )


if __name__ == "__main__":
    env = FourRoomsEnv10x10()
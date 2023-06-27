from gym_multigrid.multigrid import MultiGridEnv
from gym_multigrid.core.world import CollectWorld
from gym_multigrid.core.agent import CollectActions, Agent
from gym_multigrid.core.object import Ball
from gym_multigrid.core.grid import Grid
import random
import numpy as np


class CollectGameEnv(MultiGridEnv):
    """
    Environment in which the agents have to collect the balls
    """

    def __init__(
        self,
        size=10,
        width=None,
        height=None,
        num_balls=[],
        agents_index=[],
        balls_index=[],
        balls_reward=[],
        zero_sum=False,
        partial_obs=False,
        view_size=7,
        actions_set=CollectActions,
        render_mode="rgb_array",
    ):
        self.num_balls = num_balls
        self.collected_balls = 0
        self.balls_index = balls_index
        self.balls_reward = balls_reward
        self.zero_sum = zero_sum
        self.agents_index = agents_index
        self.world = CollectWorld
        self.keys = [
            "agent1ball1",
            "agent1ball2",
            "agent1ball3",
            "agent2ball1",
            "agent2ball2",
            "agent2ball3",
        ]

        agents = []
        for i in agents_index:
            agents.append(Agent(self.world, i, view_size=view_size))

        super().__init__(
            grid_size=size,
            width=width,
            height=height,
            max_steps=100,
            world=self.world,
            see_through_walls=False,
            agents=agents,
            partial_obs=partial_obs,
            agent_view_size=view_size,
            actions_set=actions_set,
            render_mode="rgb_array",
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height, self.world)

        # Generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height - 1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width - 1, 0)

        for number, index, reward in zip(
            self.num_balls, self.balls_index, self.balls_reward
        ):
            for i in range(number):
                self.place_obj(Ball(self.world, index, reward))

        # Randomize the player start position
        for a in self.agents:
            self.place_agent(a)

    def reset(self, seed=None):
        self.collected_balls = 0
        self.info = {
            "agent1ball1": 0,
            "agent1ball2": 0,
            "agent1ball3": 0,
            "agent2ball1": 0,
            "agent2ball2": 0,
            "agent2ball3": 0,
        }
        super().reset()
        state = self.grid.encode()
        return state, {}

    def _reward(self, i, rewards, reward=1):
        """
        Compute the reward to be given upon success
        """
        rewards[i] += reward

    def _handle_pickup(self, i, rewards, fwd_pos, fwd_cell):
        if fwd_cell:
            if fwd_cell.can_pickup():
                fwd_cell.pos = np.array([-1, -1])
                ball_idx = self.world.COLOR_TO_IDX[fwd_cell.color]
                self.grid.set(*fwd_pos, None)
                self.collected_balls += 1
                if ball_idx == 0:
                    self._reward(i, rewards, fwd_cell.reward)
                elif ball_idx == 1:
                    self._reward(i, rewards, fwd_cell.reward)
                elif ball_idx == 2:
                    self._reward(i, rewards, fwd_cell.reward)
                self.info[self.keys[3 * i + ball_idx]] += 1

    def move_agent(self, rewards, i, next_cell, next_pos):
        if next_cell is not None:
            if next_cell.type == "ball":
                self._handle_pickup(i, rewards, next_pos, next_cell)
                self.grid.set(*next_pos, self.agents[i])
                self.grid.set(*self.agents[i].pos, None)
                self.agents[i].pos = next_pos
            else:
                self._reward(i, rewards, -0.01)
        elif next_cell is None or next_cell.can_overlap():
            self.grid.set(*next_pos, self.agents[i])
            self.grid.set(*self.agents[i].pos, None)
            self.agents[i].pos = next_pos
            self._reward(i, rewards, -0.01)

    def step(self, actions):
        order = np.random.permutation(len(actions))
        rewards = np.zeros(len(actions))
        done = False
        truncated = False
        self.step_count += 1
        for i in order:
            if actions[i] == self.actions.north:
                # print('up')
                next_pos = self.agents[i].north_pos()
                next_cell = self.grid.get(*next_pos)
                self.move_agent(rewards, i, next_cell, next_pos)
            elif actions[i] == self.actions.east:
                next_pos = self.agents[i].east_pos()
                next_cell = self.grid.get(*next_pos)
                self.move_agent(rewards, i, next_cell, next_pos)
            elif actions[i] == self.actions.south:
                next_pos = self.agents[i].south_pos()
                next_cell = self.grid.get(*next_pos)
                self.move_agent(rewards, i, next_cell, next_pos)
            elif actions[i] == self.actions.west:
                next_pos = self.agents[i].west_pos()
                next_cell = self.grid.get(*next_pos)
                self.move_agent(rewards, i, next_cell, next_pos)
            elif actions[i] == self.actions.still:
                self._reward(i, rewards, -0.01)
        if self.collected_balls == self.num_balls:
            done = True
        if self.step_count >= self.max_steps:
            done = True
            truncated = True

        obs = self.grid.encode()
        return obs, rewards, done, truncated, self.info

    def simulate(self, action):
        # need to simulate what happens without changing current env
        def dummy_pickup(dummy_grid, fwd_pos, fwd_cell):
            if fwd_cell:
                if fwd_cell.can_pickup():
                    fwd_cell.pos = np.array([-1, -1])
                    dummy_grid.set(*fwd_pos, None)

        def dummy_move(dummy_grid, next_cell, next_pos):
            if next_cell is not None:
                if next_cell.type == "ball":
                    dummy_pickup(dummy_grid, next_pos, next_cell)
                    dummy_grid.set(*next_pos, self.agents[0])
                    dummy_grid.set(*self.agents[0].pos, None)
            elif next_cell is None or next_cell.can_overlap():
                dummy_grid.set(*next_pos, self.agents[0])
                dummy_grid.set(*self.agents[0].pos, None)

        def phi_v(s, snew):
            phi = np.zeros((self.phi_dim(),))
            for i in range(self.grid.width):
                for j in range(self.grid.height):
                    obj = snew.get(i, j)
                    if obj is not None and obj.type == "agent":
                        if s.get(i, j) is not None and s.get(i, j).type == "ball":
                            color = self.world.COLOR_TO_IDX[s.get(i, j).color]
                            phi[color] = 1
            return phi

        dummy_grid = self.grid.copy()
        s = self.grid.copy()
        if action == self.actions.north:
            next_pos = self.agents[0].north_pos()
            next_cell = dummy_grid.get(*next_pos)
            dummy_move(dummy_grid, next_cell, next_pos)
        elif action == self.actions.east:
            next_pos = self.agents[0].east_pos()
            next_cell = dummy_grid.get(*next_pos)
            dummy_move(dummy_grid, next_cell, next_pos)
        elif action == self.actions.south:
            next_pos = self.agents[0].south_pos()
            next_cell = dummy_grid.get(*next_pos)
            dummy_move(dummy_grid, next_cell, next_pos)
        elif action == self.actions.west:
            next_pos = self.agents[0].west_pos()
            next_cell = dummy_grid.get(*next_pos)
            dummy_move(dummy_grid, next_cell, next_pos)

        return phi_v(s, dummy_grid)


class CollectGame3Obj2Agent(CollectGameEnv):
    def __init__(self, size=10):
        super().__init__(
            size=size,
            num_balls=15,
            agents_index=[3, 5],
            balls_index=[0],
            balls_reward=[1],
            zero_sum=True,
            render_mode="rgb_array",
        )
        self.num_ball_types = self.num_balls // 5
        self.sigma = 0.1

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height, self.world)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        index = 0
        for ball in range(self.num_balls):
            if ball % 5 == 0:
                index = ball // 5
            self.place_obj(
                Ball(self.world, index, 1)
            )
        for a in self.agents:
            self.place_agent(a)

    def get_obj_grid(self, ball_idx=[0, 1, 2]):
        # return width x height grid indicating ball locations
        arr = np.zeros((self.grid.width, self.grid.height))
        for j in range(self.grid.height):
            for i in range(self.grid.width):
                c = self.grid.get(i, j)
                if (
                    c is not None
                    and c.type == "ball"
                    and self.world.COLOR_TO_IDX[c.color] in ball_idx
                ):
                    arr[i][j] = self.world.COLOR_TO_IDX[c.color] + 1
                else:
                    arr[i][j] = 0
        return arr

    def find_closest_obj(self, arr, cur_pos):
        # Get the coordinates of all objects in the grid
        object_coords = np.argwhere(arr != -1)
        # Calculate the distances from the current position to all objects
        distances = np.linalg.norm(object_coords - cur_pos, axis=1)
        # Find the index of the closest object
        closest_index = np.argmin(distances)
        # Get the coordinates of the closest object
        closest_object_coords = object_coords[closest_index]
        return closest_object_coords.tolist()

    def move_to_obj(self, cur_pos, obj_coords):
        # return which action to take to get to desired object
        x_diff = obj_coords[0] - cur_pos[0]
        y_diff = obj_coords[1] - cur_pos[1]
        if x_diff == 0 and y_diff == 0:
            return 0
        elif x_diff == 0:
            if y_diff < 0:
                return 1
            else:
                return 3
        elif y_diff == 0:
            if x_diff < 0:
                return 4
            else:
                return 2
        else:
            if abs(x_diff) > abs(y_diff):
                if x_diff < 0:
                    return 4
                else:
                    return 2
            else:
                if y_diff < 0:
                    return 1
                else:
                    return 3

    def indicator(self):
        # indicator function for grid state
        # [wall, ball1, ball2, ball3, agent1, agent2]
        num_layers = self.num_ball_types + len(self.agents_index) + 1
        phi_obs = np.zeros(
            (num_layers, self.grid.width, self.grid.height), dtype="uint8"
        )
        for i in range(self.grid.width):
            for j in range(self.grid.height):
                obj = self.grid.get(i, j)
                if obj is None:
                    phi_obs[:, i, j] = 0
                elif obj.type == "agent":
                    idx = (
                        4
                        if self.agents_index[0] == (self.world.COLOR_TO_IDX[obj.color])
                        else 5
                    )
                    phi_obs[idx, i, j] = 1
                elif obj.type == "ball":
                    idx = self.world.COLOR_TO_IDX[obj.color] + 1
                    phi_obs[idx, i, j] = 1
                elif obj.type == "wall":
                    phi_obs[0, i, j] = 1
        return phi_obs

    def toroid(self, idx):
        # transform grid into toroidal, agent-centric obs
        pos = (idx // self.grid.width, idx % self.grid.width)
        depth = self.num_ball_types + len(self.agents)
        obs = np.zeros((self.grid.width, self.grid.height, depth), dtype="float32")
        for i in range(self.grid.width):
            for j in range(self.grid.height):
                new_coords = [i - pos[0], j - pos[1]]
                obj = self.grid.get(i, j)
                if new_coords[0] < 0:
                    new_coords[0] += self.grid.width
                if new_coords[1] < 0:
                    new_coords[1] += self.grid.height
                if obj is None:
                    continue
                elif obj.type == "wall":
                    obs[new_coords[1], new_coords[0], depth - 1] = 1
                elif obj.type == "ball":
                    obs[
                        new_coords[1], new_coords[0], self.world.COLOR_TO_IDX[obj.color]
                    ] = 1
                elif obj.type == "agent" and not np.array_equal(obj.pos, pos):
                    obs[new_coords[1], new_coords[0], depth - 2] = 1
        return obs

    def gaussian(self, idx):
        phi_obs = np.zeros((self.grid.width, self.grid.height), dtype="float32")
        pos = (idx // self.grid.width, idx % self.grid.width)
        for i in range(self.grid.width):
            for j in range(self.grid.height):
                obj = self.grid.get(i, j)
                if obj is None or obj.type == "wall" or obj.type == "agent":
                    phi_obs[i, j] = 0
                elif obj.type == "ball":
                    phi_obs[i, j] = np.exp(
                        -(
                            (((pos[0] - i) / self.grid.width) ** 2)
                            + (((pos[1] - j) / self.grid.height) ** 2)
                        )
                        / self.sigma
                    )
        return phi_obs.T

    def phi(self):
        # phi(s, a, s')
        phi = np.zeros((self.ac_dim, self.phi_dim()))
        for a in range(self.ac_dim):
            phi[a, :] += self.simulate(a)
        return phi

    def phi_dim(self):
        return self.num_ball_types


class CollectGame3ObjFixed2Agent(CollectGame3Obj2Agent):
    def __init__(self):
        super().__init__()

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height, self.world)

        # Generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height - 1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width - 1, 0)

        index = 0
        ball_pos = [
            (1, 1),
            (1, 2),
            (3, 2),
            (2, 4),
            (5, 3),
            (8, 7),
            (8, 6),
            (6, 5),
            (7, 4),
            (6, 8),
            (8, 3),
            (6, 3),
            (7, 2),
            (5, 5),
            (7, 6),
        ]
        random.shuffle(ball_pos)
        for ball in range(self.num_balls):
            if ball % 5 == 0:
                index = ball // 5
            loc = ball_pos[ball]
            self.put_obj(Ball(self.world, index, 1), loc[0], loc[1])
        agent_pos = (1, height - 2)
        for a in self.agents:
            self.place_agent(a, pos=agent_pos)
            agent_pos = (agent_pos[0] + 1, agent_pos[1])


class CollectGame3ObjSingleAgent(CollectGame3Obj2Agent):
    def __init__(self):
        super().__init__(agents_index=[3])


class CollectGameQuadrants(CollectGame3Obj2Agent):
    def __init__(self):
        super().__init__()

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height, self.world)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        # place balls
        partitions = [(0, 0), (width // 2 - 1, height // 2 - 1), (width // 2 - 1, 0)]
        partition_size = (width // 2 + 1, height // 2 + 1)
        index = 0
        for ball in range(self.num_balls):
            if ball % 5 == 0:
                top = partitions[ball // 5]
                index = ball // 5
            self.place_obj(Ball(self.world, index, 1), top=top, size=partition_size)

        # place agents
        agent_pos = (1, height - 2)
        for a in self.agents:
            self.place_agent(a, pos=agent_pos)
            agent_pos = (agent_pos[0] + 1, agent_pos[1])


class CollectGameRooms(CollectGame3Obj2Agent):
    def __init__(self):
        super().__init__(size=11)

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height, self.world)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        # generate inner walls of rooms
        wall_size = self.width // 2 - 1
        self.grid.horz_wall(0, width // 2, wall_size)
        self.grid.horz_wall(width - wall_size, width // 2, wall_size)
        self.grid.vert_wall(width // 2, 0, wall_size)
        self.grid.vert_wall(width // 2, width - wall_size, wall_size)

        # place agents
        possible_coords = [(width // 2, width // 2), 
                           (width // 2 - 1, width // 2 - 1),
                           (width // 2 - 1, width // 2 + 1),
                           (width // 2 + 1, width // 2 + 1),
                           (width // 2 + 1, width // 2 - 1)]
        for a in self.agents:
            location = self._rand_elem(possible_coords)
            self.place_agent(agent=a, pos=location)

        # place balls
        partitions = [(0, 0), (width // 2 + 1, width // 2 + 1), (width // 2 + 1, 0), (0, width // 2 + 1)]
        partition_size = (width // 2 - 1, width // 2 - 1)
        index = 0
        for ball in range(self.num_balls):
            if ball % 5 == 0:
                top = partitions[ball // 5]
                index = ball // 5
                self.place_obj(Ball(self.world, index, 1), top=partitions[3], size=partition_size)
            self.place_obj(Ball(self.world, index, 1), top=top, size=partition_size)

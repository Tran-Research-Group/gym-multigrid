import random
from typing import Literal, Type

import numpy as np
from numpy.typing import NDArray

from gym_multigrid.multigrid import MultiGridEnv
from gym_multigrid.core.world import CollectWorld
from gym_multigrid.core.agent import ActionsT, AgentT, CollectActions, Agent
from gym_multigrid.core.object import Ball, WorldObjT
from gym_multigrid.core.grid import Grid
from gym_multigrid.typing import Position


class CollectGameEnv(MultiGridEnv):
    """Environment in which the agents have to collect balls.

    Inherits MultiGridEnv
    """

    def __init__(
        self,
        size: int = 10,
        width: int | None = None,
        height: int | None = None,
        num_balls: list[int] | int = [],
        agents_index: list[int] = [],
        balls_index: list[int] = [],
        balls_reward: list[float] = [],
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
        max_steps: int = 100,
    ):
        """Initialize the CollectGameEnv.

        Parameters
        ----------
        size : int, optional
            Size of grid if square. Default 10
        width : int, optional
            Width of grid, used if not square. Default None
        height : int, optional
            Height of grid, used if not square. Default None
        num_balls : list[int]
            Total number of balls present in environment.
        agents_index : list[int]
            Colour index for each agent.
        balls_index : list[int]
            Colour index for each ball type.
        balls_reward : list[float]
            Reward given for collecting each ball type.
        render_mode : Literal["human", "rgb_array"]="rgb_array"
            Rendering mode. See utils.rendering for details
        max_steps : int=100
            Maximum number of steps per episode.
        """

        self.num_balls = num_balls
        self.collected_balls = 0
        self.balls_index = balls_index
        self.balls_reward = balls_reward
        self.agents_index = agents_index
        self.world = CollectWorld
        self.actions_set = CollectActions
        partial_obs: bool = False
        view_size: int = 10
        self.keys = [
            "agent1ball1",
            "agent1ball2",
            "agent1ball3",
            "agent2ball1",
            "agent2ball2",
            "agent2ball3",
        ]

        agents: list[AgentT] = []
        for i in agents_index:
            agents.append(Agent(self.world, i, view_size=view_size))

        super().__init__(
            grid_size=size,
            width=width,
            height=height,
            max_steps=max_steps,
            world=self.world,
            see_through_walls=False,
            agents=agents,
            partial_obs=partial_obs,
            agent_view_size=view_size,
            actions_set=self.actions_set,
            render_mode=render_mode,
        )

    def _gen_grid(self, width: int, height: int):
        """Generate grid and place all the balls and agents.

        Parameters
        ----------
        width : int
            width of grid
        height : int
            height of grid
        """

        self.grid = Grid(width, height, self.world)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        assert type(self.num_balls) is list

        for number, index, reward in zip(
            self.num_balls, self.balls_index, self.balls_reward
        ):
            for i in range(number):
                self.place_obj(Ball(self.world, index, reward))

        # Randomize the player start position
        for a in self.agents:
            self.place_agent(a)

    def reset(self, seed: int | None = None):
        self.collected_balls = 0
        self.info = {
            "agent1ball1": 0,
            "agent1ball2": 0,
            "agent1ball3": 0,
            "agent2ball1": 0,
            "agent2ball2": 0,
            "agent2ball3": 0,
        }
        super().reset(seed=seed)
        state = self.grid.encode()
        return state, {}

    def _reward(
        self, agent_index: int, rewards: NDArray[np.float_], reward: float = 1
    ) -> None:
        """
        Compute the reward to be given upon success
        """
        rewards[agent_index] += reward

    def _handle_pickup(
        self,
        i,
        rewards: NDArray[np.float_],
        fwd_pos: Position,
        fwd_cell: WorldObjT | None,
    ) -> None:
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

    def move_agent(
        self,
        rewards: NDArray[np.float_],
        agent_index: int,
        next_cell: WorldObjT | None,
        next_pos: Position,
    ) -> None:
        if next_cell is not None:
            if next_cell.type == "ball":
                self._handle_pickup(agent_index, rewards, next_pos, next_cell)
                self.grid.set(*next_pos, self.agents[agent_index])
                self.grid.set(*self.agents[agent_index].pos, None)
                self.agents[agent_index].pos = next_pos
            else:
                self._reward(agent_index, rewards, -0.01)
        elif next_cell is None or next_cell.can_overlap():
            self.grid.set(*next_pos, self.agents[agent_index])
            self.grid.set(*self.agents[agent_index].pos, None)
            self.agents[agent_index].pos = next_pos
            self._reward(agent_index, rewards, -0.01)

    def step(
        self, actions: list[int] | NDArray[np.int_]
    ) -> tuple[NDArray[np.int_], NDArray[np.float_], bool, bool, dict]:
        order: list[int] = np.random.permutation(len(actions)).tolist()
        rewards: NDArray[np.float_] = np.zeros(len(actions))
        done: bool = False
        truncated: bool = False
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


class CollectGame3Obj2Agent(CollectGameEnv):
    def __init__(
        self,
        size: int = 10,
        num_balls: int = 15,
        agents_index: list[int] = [3, 5],
        balls_index: list[int] = [0, 1, 2],
        balls_reward: list[float] = [1, 1, 1],
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
        max_steps: int = 100,
    ):
        super().__init__(
            size=size,
            num_balls=num_balls,
            agents_index=agents_index,
            balls_index=balls_index,
            balls_reward=balls_reward,
            render_mode=render_mode,
            max_steps=max_steps,
        )
        assert type(self.num_balls) is int
        self.num_ball_types = self.num_balls // 5

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height, self.world)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        index = 0
        assert type(self.num_balls) is int
        num_colors: int = len(self.balls_index)
        assert len(self.balls_reward) == num_colors
        num_ball: int = round(self.num_balls / num_colors)
        for ball in range(self.num_balls):
            if ball % num_ball == 0:
                index = ball // num_ball
            self.place_obj(
                Ball(self.world, self.balls_index[index], self.balls_reward[index])
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

    def get_toroids(self):
        obs = []
        for a in self.agents:
            agent_pos = a.pos
            idx = self.grid.width * agent_pos[0] + agent_pos[1]
            obs.append(self.toroid(idx))
        return obs

    def phi_dim(self):
        return self.num_ball_types


class CollectGame3ObjSingleAgent(CollectGame3Obj2Agent):
    def __init__(self):
        super().__init__(agents_index=[3])


class CollectGameQuadrants(CollectGame3Obj2Agent):
    def __init__(self):
        super().__init__()

    def _gen_grid(self, width: int, height: int):
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
    def __init__(self, size: int = 11, *args, **kwargs):
        super().__init__(size=size, *args, **kwargs)

    def _gen_grid(self, width: int, height: int):
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
        possible_coords = [
            (width // 2, width // 2),
            (width // 2 - 1, width // 2 - 1),
            (width // 2 - 1, width // 2 + 1),
            (width // 2 + 1, width // 2 + 1),
            (width // 2 + 1, width // 2 - 1),
        ]
        for a in self.agents:
            location = self._rand_elem(possible_coords)
            self.place_agent(agent=a, pos=location)

        # place balls
        partitions = [
            (0, 0),
            (width // 2 + 1, width // 2 + 1),
            (width // 2 + 1, 0),
            (0, width // 2 + 1),
        ]
        partition_size = (width // 2 - 1, width // 2 - 1)
        index = 0
        assert type(self.num_balls) is int
        num_colors: int = len(self.balls_index)
        assert len(self.balls_reward) == num_colors
        num_ball: int = round(self.num_balls / num_colors)
        for ball in range(self.num_balls):
            if ball % num_ball == 0:
                top = partitions[ball // num_ball]
                index = ball // num_ball
                self.place_obj(
                    Ball(self.world, self.balls_index[index], self.balls_reward[index]),
                    top=partitions[3],
                    size=partition_size,
                )
            self.place_obj(
                Ball(self.world, self.balls_index[index], self.balls_reward[index]),
                top=top,
                size=partition_size,
            )


class CollectGameRoomsFixedHorizon(CollectGameRooms):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, actions):
        order: list[int] = np.random.permutation(len(actions)).tolist()
        rewards: NDArray[np.float_] = np.zeros(len(actions))
        done: bool = False
        truncated: bool = False
        self.step_count += 1
        for i in order:
            if actions[i] == self.actions.north:
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

        obs = self.grid.encode()
        return obs, rewards, done, truncated, self.info


class CollectGameRoomsRespawn(CollectGameRoomsFixedHorizon):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _respawn(self, color):
        self.place_obj(Ball(self.world, color, 1))

    def _handle_pickup(self, i, rewards, fwd_pos, fwd_cell):
        if fwd_cell:
            if fwd_cell.can_pickup():
                fwd_cell.pos = np.array([-1, -1])
                ball_idx = self.world.COLOR_TO_IDX[fwd_cell.color]
                self.grid.set(*fwd_pos, None)
                self._respawn(ball_idx)
                self.collected_balls += 1
                self._reward(i, rewards, fwd_cell.reward)
                self.info[self.keys[3 * i + ball_idx]] += 1

    def move_agent(self, rewards, i, next_cell, next_pos):
        if next_cell is not None:
            if next_cell.type == "ball":
                self._handle_pickup(i, rewards, next_pos, next_cell)
                # move agent to cell
                self.grid.set(*next_pos, self.agents[i])
                self.grid.set(*self.agents[i].pos, None)
                # update agent position variable
                self.agents[i].pos = next_pos
            # self._reward(i, rewards, -0.01)
        elif next_cell is None or next_cell.can_overlap():
            self.grid.set(*next_pos, self.agents[i])
            self.grid.set(*self.agents[i].pos, None)
            self.agents[i].pos = next_pos


class CollectGameRespawn(CollectGame3Obj2Agent):
    def __init__(self):
        super().__init__()
        self.max_steps = 50

    def _respawn(self, color):
        self.place_obj(Ball(self.world, color, 1))

    def _handle_pickup(self, i, rewards, fwd_pos, fwd_cell):
        if fwd_cell:
            if fwd_cell.can_pickup():
                fwd_cell.pos = np.array([-1, -1])
                ball_idx = self.world.COLOR_TO_IDX[fwd_cell.color]
                self.grid.set(*fwd_pos, None)
                self._respawn(ball_idx)
                self.collected_balls += 1
                self._reward(i, rewards, fwd_cell.reward)
                self.info[self.keys[3 * i + ball_idx]] += 1

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
        if self.step_count >= self.max_steps:
            done = True
            truncated = True

        obs = self.grid.encode()
        return obs, rewards, done, truncated, self.info


class CollectGameRespawnClustered(CollectGameRespawn):
    def __init__(self):
        super().__init__()

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height, self.world)

        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        # place balls
        partitions = [(0, 0), (width // 2 - 1, height // 2 - 1), (width // 2 - 1, 0)]
        partition_size = (width // 2 + 1, height // 2 + 1)
        num_ball_per_type = self.num_balls // len(partitions)
        index = 0
        for ball in range(self.num_balls):
            if ball % num_ball_per_type == 0:
                top = partitions[ball // num_ball_per_type]
                index = ball // num_ball_per_type
            self.place_obj(Ball(self.world, index, 1), top=top, size=partition_size)

        # place agents
        agent_pos = (1, height - 2)
        for a in self.agents:
            self.place_agent(a, pos=agent_pos)
            agent_pos = (agent_pos[0] + 1, agent_pos[1])

    def _respawn(self, color):
        partitions = [
            (0, 0),
            (self.width // 2 - 1, self.height // 2 - 1),
            (self.width // 2 - 1, 0),
        ]
        partition_size = (self.width // 2 + 1, self.height // 2 + 1)
        top = partitions[color]
        self.place_obj(Ball(self.world, color, 1), top=top, size=partition_size)

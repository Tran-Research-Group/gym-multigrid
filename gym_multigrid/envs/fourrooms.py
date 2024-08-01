from itertools import chain
from typing import Final, Literal, TypeAlias, TypedDict
from typing import Any, Iterable, SupportsFloat, TypeVar

from gymnasium import spaces
from gymnasium.core import ActType, ObsType
import numpy as np
from numpy.typing import NDArray

from gym_multigrid.core.agent import Agent, PolicyAgent, AgentT, FRActions
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.object import Goal
from gym_multigrid.core.object import Floor, Flag, Obstacle, WorldObjT
from gym_multigrid.core.world import FRWorld, World
from gym_multigrid.multigrid import MultiGridEnv
from gym_multigrid.policy.ctf.heuristic import RwPolicy, CtfPolicyT
from gym_multigrid.typing import Position
from gym_multigrid.utils.map import distance_area_point, distance_points, load_text_map


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
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
    ) -> None:
        """
        Initialize a new capture the flag environment.

        Parameters
        ----------

        """
        partial_obs = True

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

        super().__init__(
            width=self.width,
            height=self.height,
            max_steps=max_steps,
            see_through_walls=see_through_walls,
            agents=self.agents,
            agent_view_size=agent_view_size,
            actions_set=self.actions_set,
            partial_obs=partial_obs,
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

        # For each row of rooms
        for j in range(0, 2):
            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    pos = (xR, self._rand_int(yT + 1, yB))
                    self.grid.set(*pos, None)

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    pos = (self._rand_int(xL + 1, xR), yB)
                    self.grid.set(*pos, None)

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            # assuming random start direction
            self.agent_dir = self._rand_int(0, 4)
        else:
            self.place_agent(self.agents[0])

        if self._goal_default_pos is not None:
            goal = Goal(self.world, 0)
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.place_obj(Goal(self.world, 0))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):

        super().reset(seed=seed)
        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        for a in self.agents:
            assert a.pos is not None
            assert a.dir is not None

        # Item picked up, being carried, initially nothing
        for a in self.agents:
            a.carrying = None

        # Step count since episode start
        self.step_count: int = 0

        # Return first observation
        if self.partial_obs:
            obs = self.gen_obs()
        else:
            obs = [
                self.grid.encode_for_agents(agent_pos=self.agents[i].pos)
                for i in range(len(self.agents))
            ]
        obs = [self.world.normalize_obs * ob for ob in obs]
        return obs[0]

    def step(self, actions):
        """
        Multi-Agent is not implemented
        """
        self.step_count += 1
        actions = [actions]  # b/c single agent

        order = np.random.permutation(len(actions))

        rewards = np.zeros(len(actions))
        terminated = False
        truncated = False

        for i in order:
            if (
                self.agents[i].terminated
                or self.agents[i].paused
                or not self.agents[i].started
            ):
                continue

            # Get the position in front of the agent
            fwd_pos = self.agents[i].front_pos

            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)

            # Rotate left
            if actions[i] == self.actions.left:
                self.agents[i].dir -= 1
                if self.agents[i].dir < 0:
                    self.agents[i].dir += 4

            # Rotate right
            elif actions[i] == self.actions.right:
                self.agents[i].dir = (self.agents[i].dir + 1) % 4

            # Move forward
            elif actions[i] == self.actions.forward:
                if fwd_cell is not None:
                    if fwd_cell.type == "goal":
                        done = True
                        rewards = self._reward(i, rewards, 1)
                    elif fwd_cell.type == "switch":
                        self._handle_switch(i, rewards, fwd_pos, fwd_cell)
                    elif fwd_cell.type == "ball":
                        rewards = self._handle_pickup(i, rewards, fwd_pos, fwd_cell)
                elif fwd_cell is None or fwd_cell.can_overlap():
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.grid.set(*self.agents[i].pos, None)
                    self.agents[i].pos = fwd_pos
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)

            else:
                raise ValueError(f"Unknown action: {actions[i]}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        if self.partial_obs:
            obs = self.gen_obs()
        else:
            obs = [
                self.grid.encode_for_agents(agent_pos=self.agents[i].pos)
                for i in range(len(actions))
            ]

        obs = [self.world.normalize_obs * ob for ob in obs]  # b/c single agent

        return obs[0], rewards[0], terminated, truncated, {}  # b/c single agent

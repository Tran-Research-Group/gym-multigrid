import numpy as np
from numpy.typing import NDArray

from gym_multigrid.multigrid import MultiGridEnv
from gym_multigrid.core.world import SaveTheCityWorld
from gym_multigrid.core.agent import SaveTheCityActions, Firefighter, Builder, Generalist
from gym_multigrid.core.object import Building, WorldObjT
from gym_multigrid.core.grid import Grid
from gym_multigrid.typing import Position


class SaveTheCityEnv(MultiGridEnv):
    """
    Environment in which agents must build buildings and extinguish fires.
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30  # or 15/60/etc depending on your use case
    }

    def __init__(self, size: int = 10, num_buildings: int = 3, agent_types: list[str] = None,
                 actions_set=SaveTheCityActions, *args, **kwargs):
        """
        Initialize the SaveTheCity environment.

        Parameters
        ----------
        size : int
            Grid size (square).
        num_buildings : int
            Total number of buildings in the environment.
        agent_types : list[str]
            List of agent types: "firefighter", "builder", "generalist".
        actions_set : Enum
            Action space enum.
        respawn_fires : bool
            Whether fires randomly reignite (default: False).
        partial_obs : bool
            Whether agents have partial observability (default: False).
        """
        self.size = size
        self.num_buildings = num_buildings
        self.world = SaveTheCityWorld
        self.actions_set = actions_set
        self.respawn_fires = kwargs.get("respawn_fires", False)
        self.partial_obs: bool = kwargs.get("partial_obs", False)

        # Default agent types if not provided
        if agent_types is None:
            agent_types = ["firefighter", "builder", "generalist"]

        # Initialize agents
        self.agents = []
        for agent_type in agent_types:
            if agent_type == "firefighter":
                self.agents.append(Firefighter(self.world))
            elif agent_type == "builder":
                self.agents.append(Builder(self.world))
            elif agent_type == "generalist":
                self.agents.append(Generalist(self.world))
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")

        # Initialize per-agent tracking info
        self.info = {
            f"agent{i+1}": {"fires_extinguished": 0, "buildings_completed": 0}
            for i in range(len(self.agents))
        }

        super().__init__(
            grid_size=self.size,
            width=None,
            height=None,
            max_steps=100,
            world=self.world,
            see_through_walls=False,
            agents=self.agents,
            partial_obs=self.partial_obs,
            actions_set=self.actions_set,
            render_mode="human",
        )

    def _gen_grid(self, width: int, height: int):
        """
        Generate the grid and place buildings and agents.
        """
        self.grid = Grid(width, height, self.world)

        # Generate surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        # Place buildings
        for i in range(self.num_buildings):
            # Alternate between slow and fast-burning buildings
            fast_burning = (i % 2 == 0)  # every other building is fast-burning
            burn_rate = 1

            building = Building(
                world=self.world,
                burn_rate=burn_rate,
                build_speed=2,
                firefight_speed=2,
                fast_burning=fast_burning
            )
            self.place_obj(building)

        # Place agents randomly
        for agent in self.agents:
            self.place_agent(agent)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.collected_balls = 0
        self.info = {f"agent{i+1}": {"fires_extinguished": 0, "buildings_completed": 0} for i in range(len(self.agents))}
        super().reset(seed=seed)
        state = self.grid.encode()
        return state, self.info

    def _reward(
        self, current_agent: int, rewards: NDArray[np.float64], reward: float = 1
    ) -> None:
        """
        Compute the reward to be given upon success
        """
        rewards[current_agent] += reward

    def _respawn(self, color):
        self.place_obj(Ball(self.world, color, self.balls_reward[color]))

    def _handle_pickup(
        self,
        i,
        rewards: NDArray[np.float64],
        fwd_pos: Position,
        fwd_cell: WorldObjT | None,
    ) -> None:
        if fwd_cell and fwd_cell.can_pickup():
            fwd_cell.pos = np.array([-1, -1])
            ball_idx = self.world.COLOR_TO_IDX[fwd_cell.color]
            self.grid.set(*fwd_pos, None)
            if self.respawn:
                self._respawn(ball_idx)
            self.collected_balls += 1
            self._reward(i, rewards, fwd_cell.reward)
            self.info[self.keys[self.num_ball_types * i + ball_idx]] += 1

    def move_agent(
        self,
        rewards: NDArray[np.float64],
        agent_index: int,
        next_cell: WorldObjT | None,
        next_pos: Position,
    ) -> None:
        """
        Method to move given agent to given next position

        Parameters
        ----------
        rewards : NDArray[np.float64]
            array of rewards for each agent
        agent_index : int
            index of agent to move
        next_cell : WorldObjT | None
            object corresponding to next position
        next_pos : Position
            position coordinates to move agent to
        """
        if next_cell is not None:
            if next_cell.type == "ball":
                self._handle_pickup(agent_index, rewards, next_pos, next_cell)
                # move agent to cell
                self.grid.set(*next_pos, self.agents[agent_index])
                self.grid.set(*self.agents[agent_index].pos, None)
                # update agent position variable
                self.agents[agent_index].pos = next_pos
        elif next_cell is None or next_cell.can_overlap():
            self.grid.set(*next_pos, self.agents[agent_index])
            self.grid.set(*self.agents[agent_index].pos, None)
            self.agents[agent_index].pos = next_pos

    def step(
        self, actions: list[int] | NDArray[np.int_]
    ) -> tuple[NDArray[np.int_], NDArray[np.float64], bool, bool, dict]:
        order: list[int] = np.random.permutation(len(actions)).tolist()
        rewards: NDArray[np.float64] = np.zeros(len(actions))
        terminated: bool = False
        truncated: bool = False
        self.step_count += 1
        for i in order:
            if actions[i] == self.actions.NORTH:
                next_pos = self.agents[i].north_pos()
                next_cell = self.grid.get(*next_pos)
                self.move_agent(rewards, i, next_cell, next_pos)
            elif actions[i] == self.actions.EAST:
                next_pos = self.agents[i].east_pos()
                next_cell = self.grid.get(*next_pos)
                self.move_agent(rewards, i, next_cell, next_pos)
            elif actions[i] == self.actions.SOUTH:
                next_pos = self.agents[i].south_pos()
                next_cell = self.grid.get(*next_pos)
                self.move_agent(rewards, i, next_cell, next_pos)
            elif actions[i] == self.actions.WEST:
                next_pos = self.agents[i].west_pos()
                next_cell = self.grid.get(*next_pos)
                self.move_agent(rewards, i, next_cell, next_pos)
        if not self.respawn and self.collected_balls == self.num_balls:
            terminated = True
        if self.step_count >= self.max_steps:
            truncated = True

        obs = self.grid.encode()
        return obs, rewards, terminated, truncated, self.info

    def phi_dim(self) -> int:
        """
        Helper method to get feature vector dimension

        Returns
        -------
        int
            length of feature vector = number of ball types
        """
        return self.num_ball_types

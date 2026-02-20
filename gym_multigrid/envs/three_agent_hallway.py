from typing import Literal, Optional, TypedDict, TypeAlias
from dataclasses import asdict

import numpy as np
from numpy.typing import NDArray
from gymnasium import spaces

from gym_multigrid.core.agent import HallwayActions, ActionsT, Agent
from gym_multigrid.core.agent import HALL_DIR_TO_VEC
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.object import AgentGoal, Wall, WorldObjT
from gym_multigrid.core.world import WorldT, TeamNavigationWorld
from gym_multigrid.multigrid import MultiGridEnv
from gym_multigrid.typing import Position
from gym_multigrid.core.object_group import ObjectGroup, EnvObjectGroup, ObjGroupT


# Classes used to interface with the env
class EnvInfo(TypedDict):
    """info about the environment used for PyMARL training"""

    state_shape: int
    obs_shape: int
    n_actions: int
    n_agents: int


class StepInfo(TypedDict):
    """info used in the env's step function"""

    success: bool
    position: tuple


State: TypeAlias = NDArray[np.int_] | NDArray[np.float32] | NDArray[np.float64]

Observation: TypeAlias = (
    dict[str, NDArray[np.int_]] | NDArray[np.int_] | NDArray[np.float32]
)


class ThreeAgentHallwaysEnv(MultiGridEnv):
    """based on the "join1" 3-agent hallway environment from MAIC
    https://github.com/mansicer/MAIC/blob/main/src/envs/join1.py, originally from NDQ
    https://github.com/TonghanWang/NDQ
    """

    metadata = {"render_fps": 10, "render_modes": ["human", "rgb_array"]}

    # setup and env properties
    def __init__(
        self,
        num_agents: int = 3,
        hall_lengths: list[int] = [2, 6, 10],
        p_intended_movement: float = 1.0,
        actions_set: type[ActionsT] = HallwayActions,
        world: WorldT = TeamNavigationWorld,
        observation_option: Literal["x_position"] = "x_position",
        obs_type: Literal["array", "array_scaled"] = "array_scaled",
        agent_dir_to_vec: list[NDArray[np.int_]] = HALL_DIR_TO_VEC,
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
        init_state_dist: None = None,
        subtask_type: None = None,
        subtask_idx: None = None,
    ) -> None:
        """
        Constructor for the TeamNavigationEnv class.

        Parameters
        ----------
        num_agents : int = 3
            number of agents in the environment.
        p_intended_movement : float = 0.95
            Probability of the intended movement.
            Should be in the range [0, 1].
        observation_option : Literal["goal"] = "goal"
            Observation option.
            - "goal": The observation includes agent positions and position of the assigned goal.
        actions_set : type[ActionsT] = HallwayActions
            Set of actions for the agents.
            By default, there are five actions: "stay", "up", "right", "down", and "left".
        agent_dir_to_vec : list[NDArray[np.int_]] = HALL_DIR_TO_VEC
            Direction vectors for the agents.
            The length of the list should be equal to the number of actions in the actions set.
        world : WorldT = LabyrinthWorld
            World for the environment.
        render_mode : Literal["human", "rgb_array"] = "rgb_array"
            Render mode for the environment.
        """

        self.num_agents: int = num_agents
        self.p_intended_movement: float = p_intended_movement

        # set env height and width
        height = 7
        width = 13
        self.goal_x = width - 2

        self.goal_state_set = ((self.goal_x, 1), (self.goal_x, 3), (self.goal_x, 5))

        # hall_length = number of empty states in each agent's hall
        # hall length in join1 was [2, 6, 10], but the goal state at the
        # end of the hall adds 1 more length to the total hall length
        # total hall length should be x+1 for each x in state_numbers from join1
        self.hall_lengths = np.array(hall_lengths)

        # observation config
        self.observation_option = observation_option
        self.obs_type = obs_type

        # agent config
        agent_view_size: int | None = None
        agents: list[Agent] = [
            Agent(
                world=world,
                index=i,
                view_size=agent_view_size,
                actions=actions_set,
                dir_to_vec=agent_dir_to_vec,
            )
            for i in range(num_agents)
        ]
        uncached_object_types: list[str] = ["agent"]

        # have to init this after the set of agents is constructed
        self.agent_goal_state_sets: dict[int, NDArray]
        self.termination_condition = "reach_assigned_goal_state"
        self.build_goal_state_sets(agents)

        # basic grid init
        self.object_options: dict[str, WorldObjT] = {
            "goal": AgentGoal,
            "wall": Wall,
        }
        self.obj_group_dict: dict[str, dict[int, ObjGroupT]]
        self.init_grid: Grid

        # scalings so entries in these tensors are in [0, 1]
        obj_encoding_scaling = len(world.OBJECT_TO_IDX) - 1
        if world.encode_dim == 2:
            # state[:, :, 1] captures agent indices, so scale by num_agents-1 since 0-indexed
            self.state_scaling = np.array([obj_encoding_scaling, self.num_agents - 1])

        self.obs_scaling = width

        super().__init__(
            agents=agents,
            width=width,
            height=height,
            actions_set=actions_set,
            world=world,
            render_mode=render_mode,
            uncached_object_types=uncached_object_types,
        )

        # define the action space
        self.action_space = spaces.MultiDiscrete(
            [len(self.actions) for _ in range(self.num_agents)]
        )


    def _set_observation_space(self) -> spaces.Box:
        max_x: int = self.width - 1
        max_y: int = self.height - 1

        obs_shape = self.reset()[0].shape

        if self.obs_type in ["array", "array_scaled"]:
            if self.obs_type == "array":
                max_val = np.max((max_x, max_y))
            else:
                max_val = 1

            observation_space = spaces.Box(
                low=np.zeros(obs_shape),
                high=max_val * np.ones(obs_shape),
                dtype=np.float32,
            )

        else:
            raise ValueError(f"Invalid observation type: {self.obs_type}")

        return observation_space

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> NDArray[np.int_]:

        super().reset(seed=seed, options=options)

        obs: Observation = self._get_obs()

        return obs

    def _gen_grid(self, width, height) -> None:
        self.grid = Grid(width, height, self.world)

        # make a list of all the objects you want to spawn in the env
        n_wall_groups = 0
        env_object_config = []
        env_object_config.append(
            # surrounding wall
            EnvObjectGroup(
                obj_type="wall",
                group_index=n_wall_groups,
                pos=((0, 0, self.width, self.height),),
                color="grey",
                spawned_subtask_indices=(0,),
                fill_mode="empty",
            ),
        )
        n_wall_groups += len(env_object_config)

        # middle walls
        for x, y in self.goal_state_set:
            if y > self.height:
                pass
            else:
                env_object_config.append(
                    EnvObjectGroup(
                        obj_type="wall",
                        group_index=n_wall_groups,
                        pos=((0, y+1, self.width, 1),),
                        color="grey",
                        spawned_subtask_indices=(0,),
                        fill_mode="filled",
                    )
                )
                n_wall_groups += 1

        # fill in agent hallways so they are different lengths
        for i, (goal_x, goal_y) in enumerate(self.goal_state_set):
            hall_length = self.hall_lengths[i]
            wall_length = goal_x - hall_length

            if wall_length > 0:
                env_object_config.append(
                    EnvObjectGroup(
                        obj_type="wall",
                        group_index=n_wall_groups,
                        pos=((0, goal_y, wall_length, 1),),
                        color="grey",
                        spawned_subtask_indices=(0,),
                        fill_mode="filled",
                    )
                )
                n_wall_groups += 1

        # place the goals
        env_object_config.append(
            EnvObjectGroup(
                obj_type="goal",
                group_index=0,
                pos=self.goal_state_set,
                color="green",
                spawned_subtask_indices=(0,),
                fill_mode="empty",
            )
        )

        obj_group_dict: dict[str, dict[int, ObjGroupT]] = {}

        # Place objects
        for obj_group_config in env_object_config:
            obj_type: str = obj_group_config.obj_type
            group_index: int = obj_group_config.group_index

            if obj_type not in obj_group_dict:
                obj_group_dict[obj_type] = {}
            else:
                pass

            obj_group_dict[obj_type][group_index] = ObjectGroup(
                object_options=self.object_options, **asdict(obj_group_config)
            )
            obj_group_dict[obj_type][group_index].put_objects(self.grid, self.world)

        self.obj_group_dict = obj_group_dict

        # init_grid has to be defined before the agents are placed
        self.init_grid: Grid = self.grid.copy()

        # Place the agents by sampling a state from the initial state distribution
        init_pos: list[tuple[int, int]] = []
        for i, agent in enumerate(self.agents):
            # uniformly sample x positions in non-goal states in each agent's hallway
            x_pos = self.np_random.integers(
                low=self.goal_x - self.hall_lengths[i], high=self.goal_x - 1
            ).item()
            init_pos.append((x_pos, self.goal_state_set[i][1]))

        assert len(self.agents) == len(init_pos)
        for agent, pos in zip(self.agents, init_pos):
            self.place_agent(agent, pos)

    def get_state(self) -> State:
        """get state with full information about all objects in the env"""
        state = self.grid.encode()

        # scale to range of [0, 1]
        state = np.divide(state, self.state_scaling)

        return state

    def _get_obs(self) -> Observation:
        obs: NDArray = np.zeros((self.num_agents, 1))
        for agent in self.agents:
            # each agent observes its x position
            obs[agent.index] = np.array(agent.pos[0])

        # if self.obs_type == "array_scaled":
        #     obs = obs / self.obs_scaling

        return obs

    def _scale_agent_obs(self, agent_obs: NDArray) -> NDArray:
        return agent_obs / self.obs_scaling

    def get_env_info(self) -> EnvInfo:
        # obs_shape should only be the shape of a single agent
        # self.observation_space.shape = (n_agents, dim_1_size, dim_2_size, ...)
        obs_shape = int(np.prod(self.observation_space.shape[1:]))

        env_info: EnvInfo = {
            # length of the state tensor when flattened to a 1D vector
            "state_shape": self.width * self.height * self.world.encode_dim,
            "obs_shape": obs_shape,
            "n_actions": len(self.actions),
            "n_agents": self.num_agents,
        }

        return env_info

    def get_avail_actions(self, verbose=False) -> list[list[bool]]:
        """gets available actions for each agent

        Returns
        -------
        avail_actions: list[list[bool]]
            available actions for each agent
        """
        avail_actions = []
        for agent in self.agents:
            avail_actions.append(self._get_avail_actions_agent(agent))

        if verbose:
            self.print_avail_actions_str(avail_actions)

        return avail_actions

    def print_avail_actions_str(self, avail_actions):
        if self.actions == HallwayActions:
            actions = [a.name for a in HallwayActions]

            print("Available actions")
            for agent_idx, avail_actions_agent in enumerate(avail_actions):
                action_str = ""
                for i, avail in enumerate(avail_actions_agent):
                    if avail:
                        action_str += f"{actions[i]},  "

                print(f"Agent: {agent_idx} --- {action_str}")

    def _get_avail_actions_agent(self, agent: Agent) -> list[bool]:
        if self.actions == HallwayActions:
            avail_actions_tmp: dict[int, bool] = {}

            # populate with filler data
            for action in HallwayActions:
                avail_actions_tmp[action.value] = True

            # set the values in avail_actions
            # you should be able to set the desired order of the neighbor positions here
            neighbor_positions: dict[str, NDArray[np.int_]] = (
                agent.get_left_right_neighbor_pos()
            )

            for direction, pos in neighbor_positions.items():
                neighbor_cell = self.grid.get(*pos)
                if (neighbor_cell is None) or (neighbor_cell.can_overlap()):
                    continue
                else:
                    avail_actions_tmp[HallwayActions[direction].value] = False

            # turn avail_actions into a list with the ordering of the actions same as in HallwayActions
            avail_actions: list[bool] = list(avail_actions_tmp.values())

        else:
            raise NotImplementedError(
                "This action space is not implemented in _get_avail_actions"
            )

        return avail_actions

    # general env step logic
    def step(
        self,
        actions: NDArray[np.int_],
    ) -> tuple[Observation, float, bool, bool, StepInfo]:

        self.step_count += 1

        # transition from s_t to s_{t+1}
        curr_state, next_state = self._move_agents(actions=actions)

        # get reward for transition (s_t, a_t, s_{t+1})
        reward: float = self._reward(
            curr_state=curr_state, actions=actions, next_state=next_state
        )

        # get observation from being in s_{t+1}
        obs: Observation = self._get_obs()

        terminated, all_at_goal = self._terminated(next_state=next_state)

        # truncated is handled by a Gymnasium wrapper
        truncated: bool = False

        info: StepInfo = self._get_step_info(all_at_goal)

        return obs, reward, terminated, truncated, info

    # agent movement
    def _move_agents(
        self, actions: list[int]
    ) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
        """
        Move agents based on the chosen action and environment randomness

        Parameters
        ----------
        actions : list[int]
            Actions to take.

        Returns
        -------
        curr_state : NDArray[np.int_]
            (x, y) positions of the agents at time t (before the state transition).
        next_state : NDArray[np.int_]
            (x, y) positions of the agents at time t+1 (after the state transition).
        """
        # Randomly generate the order of the agents
        agent_indices: list[int] = list(range(self.num_agents))
        self.np_random.shuffle(agent_indices)

        # placeholder for the (x, y) positions of all the agents
        ## assumes no other state information matters for the reward calculation
        curr_state: NDArray[np.int_] = np.zeros((self.num_agents, 2))
        next_state: NDArray[np.int_] = np.zeros((self.num_agents, 2))

        for i in agent_indices:
            curr_state[i, :] = self.agents[i].pos
            self._move_agent(action=actions[i], agent=self.agents[i])

            # set next_state_list[:, i] to be the next (x, y) position of agent i
            next_state[i, :] = self.agents[i].pos

        return curr_state, next_state

    def _move_agent(self, action: int, agent: Agent) -> None:
        """
        Move agents based on the action.

        Parameters
        ----------
        action : int
            Action chosen by the agent.
        agent : Agent
            Agent to move.
        """
        ################
        # pick out the next position the agent would go to based on its action if there were no randomness in the env's transition function
        ## this logic would not be necessary if there were action masking based on avail_actions that would prevent an agent from selecting an action like "walk into the wall", but I guess that isn't standard in gymnasium envs
        chosen_next_state: Position
        assert agent.pos is not None
        chosen_next_state = agent.pos + agent.dir_to_vec[action]

        # define a flag to tell if the agent will move or not
        chosen_next_state_in_available_pos: bool = False
        available_pos: list[Position] = self._get_available_pos(agent=agent)

        for pos in available_pos:
            if np.array_equal(pos, chosen_next_state):
                chosen_next_state_in_available_pos = True
                break
            else:
                pass

        ################
        # define agent movement logic
        # agent moves to a new state
        if chosen_next_state_in_available_pos:
            action_probs: list[float] = []

            # assign probabilities to each possible action
            # assigns equal probability to all other available next positions
            # EX: if the agent chose "up", and P(s, "up", s_up) = 0.97, then
            # P(s, "right", s_right) = P(s, "down", s_down) = P(s, "left", s_left) =
            # (1 - 0.97) / 3 = 0.1
            for pos in available_pos:
                action_probs.append(
                    self.p_intended_movement
                    if np.array_equal(pos, chosen_next_state)
                    else (1 - self.p_intended_movement) / (len(available_pos) - 1)
                )

            # normalize the action probabilities
            action_probs = np.array(action_probs) / np.sum(action_probs)
            avail_pos_indices: list[int] = list(range(len(available_pos)))

            # chose the next position based on the randomly chosen action
            next_state_idx = self.np_random.choice(avail_pos_indices, p=action_probs)
            next_state = available_pos[next_state_idx]

            # make the agent move
            next_cell: WorldObjT | None = self.grid.get(*next_state)
            if next_cell is None:
                agent.move(next_state, self.grid, self.init_grid, bg_color=None)
            elif next_cell.can_overlap():
                agent.move(
                    next_state, self.grid, self.init_grid, bg_color=next_cell.bg_color
                )
            else:
                raise ValueError(
                    f"Invalid action f{action} and position f{next_state} for agent {agent.index}. \
                        Available positions: {available_pos}"
                )

        # agent stays in its current state
        else:
            pass

    def _get_available_pos(self, agent: Agent) -> list[Position]:
        possible_pos: list[Position] = []

        for direction in agent.dir_to_vec:
            next_state: Position = agent.pos + direction

            if (
                next_state[0] < 0
                or next_state[1] < 0
                or next_state[0] >= self.width
                or next_state[1] >= self.height
            ):
                pass
            else:
                next_cell: WorldObjT | None = self.grid.get(*next_state)

                if self.grid.get(*next_state) is None:
                    possible_pos.append(next_state)
                elif next_cell.can_overlap():
                    possible_pos.append(next_state)
                else:
                    pass

        return possible_pos

    # reward function
    def _reward(
        self,
        curr_state: NDArray[np.int_],
        actions: NDArray[np.int_],
        next_state: NDArray[np.int_],
    ) -> float:
        """
        Parameters
        ----------
        curr_state : NDArray[np.int_]
            current joint state
        actions : NDArray[np.int_]
            current agent actions
        next_state : NDArray[np.int_]
            next joint state
        Returns
        -------
        float
            reward
        """
        reward: float = 0.0

        # Give a reward if all agents arrive at the goal at the same time.
        # You just need to check if all agents are in the goal state b/c _terminated()
        # will end an ep if any agent reaches the goal.
        all_at_goal_flag = self._check_all_agents_reached_goal(next_state=next_state)

        if all_at_goal_flag:
            reward += 10.0

        return reward

    # termination function
    def _terminated(self, next_state: NDArray[np.int_]) -> tuple[bool, bool]:
        """check terminated conditions

        Parameters
        ----------
        next_state : NDArray[np.int_]
            next state

        Returns
        -------
        tuple[bool, bool]
            terminated status, flag that's true when all agents are at a goal state
        """
        # I define two different variables here b/c we may consider other conditions
        # in defining "terminated" in the future or in other envs
        terminated = self._check_any_agent_reached_goal(next_state=next_state)

        all_at_goal_flag = self._check_all_agents_reached_goal(next_state=next_state)

        return terminated, all_at_goal_flag

    def _check_any_agent_reached_goal(self, next_state: NDArray[np.int_]) -> bool:
        """check if each agent is in a valid goal state"""

        reached_goal: NDArray = np.zeros(len(self.agents))

        for i, agent in enumerate(self.agents):
            # ensure goal_state_set for this agent is a 2D array so the logic below works as expected
            goal_state_set = self.agent_goal_state_sets[agent.index]
            if len(goal_state_set.shape) == 1:
                goal_state_set = np.expand_dims(goal_state_set, 0)

            reached_goal[i] = np.any(
                np.all(
                    next_state[agent.index, :] == goal_state_set,
                    axis=1,
                )
            )

        # Convert from np bool to python bool b/c
        # np bools cannot be interpreted as integers.
        # This is an issue that comes up later in the training pipeline.
        any_at_goal_flag = bool(np.any(reached_goal))

        return any_at_goal_flag


    def _check_all_agents_reached_goal(self, next_state: NDArray[np.int_]) -> bool:
        """check if each agent is in a valid goal state"""

        reached_goal: NDArray = np.zeros(len(self.agents))

        for i, agent in enumerate(self.agents):
            # ensure goal_state_set for this agent is a 2D array so the logic below works as expected
            goal_state_set = self.agent_goal_state_sets[agent.index]
            if len(goal_state_set.shape) == 1:
                goal_state_set = np.expand_dims(goal_state_set, 0)

            reached_goal[i] = np.any(
                np.all(
                    next_state[agent.index, :] == goal_state_set,
                    axis=1,
                )
            )

        # Convert from np bool to python bool b/c
        # np bools cannot be interpreted as integers.
        # This is an issue that comes up later in the training pipeline.
        all_at_goal_flag = bool(np.all(reached_goal))

        return all_at_goal_flag

    # step info
    def _get_step_info(self, all_at_goal: bool = False) -> StepInfo:
        """other env info to be returned by the env's step method"""
        step_info: StepInfo = {
            "success": all_at_goal,
            "position": self.get_agent_positions(),
        }

        return step_info

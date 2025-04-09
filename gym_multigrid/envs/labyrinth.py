from typing import Any, Literal, Optional, TypedDict
from dataclasses import asdict

import numpy as np
from numpy.typing import NDArray
from gymnasium import spaces

from ..core.agent import NavigationActions, ActionsT, Agent, NAV_DIR_TO_VEC
from ..core.grid import Grid
from ..core.object import AgentGoal, Wall, WorldObjT, Zone
from ..core.object import SimpleDoor as Door
from ..core.world import WorldT, LabyrinthWorld
from ..multigrid import MultiGridEnv
from ..typing import Position
from ..utils.subtasks import (
    EnvObjectGroup,
    SubtaskData,
    StateData,
    HLMDPData,
    ObjectGroup,
    ObjGroupT,
)


import pdb



##################
# updated interface
##################
state_data_list = [
    StateData(
        idx=0,
        outgoing_init_state_dist=[1.0, [[1, 3], [1, 6]]],
    ),
    StateData(
        idx=1,
        outgoing_init_state_dist=[1.0, [[3, 3], [3, 6]]],
    )
]


subtask_data_list = [
    SubtaskData(
        edge=(0, 1),
        idx=0,
        final_state=[[3, 3], [3, 6]],
        termination_condition="reach_assigned_final_state",
    ),
    SubtaskData(
        edge=(1, 2),
        idx=1,
        final_state=[[5, 3], [5, 6]],
        termination_condition="reach_assigned_final_state",
    ),
]


hlmdp_data = HLMDPData(state_data_list, subtask_data_list)


class EnvInfo(TypedDict):
    """outputs info about the environment used for PyMARL training
    """
    state_shape: int
    obs_shape: int
    n_actions: int
    n_agents: int


##################
# the env class
##################
class LabyrinthEnv(MultiGridEnv):
    #############
    # setup and env properties
    #############
    def __init__(
        self,
        num_agents: int = 3,
        seed: int = 1,
        p_intended_movement: float = 0.95,
        actions_set: type[ActionsT] = NavigationActions,
        subtask_idx: int = 0,
        world: WorldT = LabyrinthWorld,
        hlmdp_data: HLMDPData = hlmdp_data,
        observation_option: Literal[
            "final_goal", "intermediate_goal"
        ] = "intermediate_goal",
        obs_type: Literal["dict", "array"] = "array",
        width: int = 10,
        height: int = 10,
        agent_dir_to_vec: list[NDArray[np.int_]] = NAV_DIR_TO_VEC,
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
    ) -> None:
        """
        Constructor for the LabyrinthEnv class.

        Parameters
        ----------
        width : int = 10
            Width of the grid.
        height : int = 9
            Height of the grid.
        num_agents : list[int] = [0, 1, 2]
            Indices of agents in the environment.
        p_intended_movement : float = 0.95
            Probability of the intended movement.
            Should be in the range [0, 1].
        subtask_idx: int = 0
            The current subtask index.
        hlmdp_data: HLMDPData = hlmdp_data
            Defines the structure of the leader's high-level MDP
        observation_option : Literal["final_goal", "intermediate_goal"] = "final_goal"
            Observation option.
            - "final_goal": The observation is the positions of the final goal and agents.
            - "intermediate_goal": The observation is the positions of all the goals and agents.
        actions_set : type[ActionsT] = NavigationActions
            Set of actions for the agents.
            By default, there are five actions: "stay", "up", "right", "down", and "left".
        agent_dir_to_vec : list[NDArray[np.int_]] = NAV_DIR_TO_VEC
            Direction vectors for the agents.
            The length of the list should be equal to the number of actions in the actions set.
        world : WorldT = LabyrinthWorld
            World for the environment.
        render_mode : Literal["human", "rgb_array"] = "rgb_array"
            Render mode for the environment.
        """
        self.num_agents: int = num_agents
        self.p_intended_movement: float = p_intended_movement
        self.subtask_idx = subtask_idx
        self.hlmdp_data = hlmdp_data

        # observation config
        self.observation_option: Literal["final_goal", "intermediate_goal"] = (
            observation_option
        )
        self.obs_type = obs_type

        # initialize RNG
        self.rng = np.random.default_rng(seed)

        # agent config
        agent_view_size: int = None
        agents: list[Agent] = [
            Agent(world, i, agent_view_size, actions_set, agent_dir_to_vec)
            for i in range(num_agents)
        ]

        # goal config
        self.final_goal: tuple[tuple[int, int], ...] = ()
        self.agent_goals: dict[int, list[tuple[int, int]]] = {}

        # self.subtask_dict: dict[int, Subtask] = {
        #     i: Subtask(**subtask_config[i]) for i in range(len(subtask_config))
        # }

        # self.terminal_subtasks: list[Subtask] = [
        #     subtask
        #     for subtask in self.subtask_dict.values()
        #     if subtask.next_subtask == "terminal"
        # ]

        for subtask in self.subtask_dict.values():
            for agent_index, pos in subtask.assigned_agent_goal.items():
                if agent_index in self.agent_goals:
                    self.agent_goals[agent_index].append(pos)
                else:
                    self.agent_goals[agent_index] = [pos]

            if subtask.next_subtask == "terminal":
                self.final_goal = tuple(subtask.assigned_agent_goal.values())
                self.final_goal_group_index = subtask.goal_group_index


        uncached_object_types: list[str] = ["agent"]

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

        # define available objects in this environment
        self.object_options: dict[str, WorldObjT] = {
                "goal": AgentGoal,
                "door": Door,
                "zone": Zone,
                "wall": Wall,
        }


    def _set_observation_space(self) -> spaces.Box:
        max_x: int = self.width - 1
        max_y: int = self.height - 1

        # goal_indices was originally called "nums_goal", which is a confusing name for a variable when you also have "num_goals" (with no "s" after "num") as a different variable
        if self.observation_option == "final_goal":
            goal_indices: list[int] = [1 for _ in range(self.num_agents)]
        elif self.observation_option == "intermediate_goal":
            goal_indices: list[int] = [
                len(agent_goals) for agent_goals in self.agent_goals.values()
            ]
        else:
            raise ValueError(f"Invalid observation option: {self.observation_option}")

        if self.obs_type == "dict":
            observation_space = spaces.Dict(
                {
                    str(i): spaces.Box(
                        low=np.zeros(2 * (num_goals + 1)),
                        high=np.array([max_x, max_y] * (num_goals + 1)),
                        dtype=np.int_,
                    )
                    for i, num_goals in enumerate(goal_indices)
                }
            )

        else:
            obs = self.get_obs()

            # TODO come back and fix this to be the actual shape of the obs
            observation_space = spaces.Box(
                low=0,
                high=1,
                shape=[3],
            )

        return observation_space

    # def _find_first_subtasks(self) -> list[int]:
    #     # 1. Construct the graph of the goal groups
    #     # Each tuple contains (goal_group_index, next_goal)
    #     nodes: list[tuple[int, int | None]] = []
    #     final_nodes: list[tuple[int, int | None]] = []
    #     for subtask_id, subtask in self.subtask_dict.items():
    #         if subtask.next_subtask == "terminal":
    #             final_nodes.append((subtask.goal_group_index, None))
    #         else:
    #             nodes.append((subtask_id, subtask.next_subtask))

    #     # 2. Find the first goal groups from the final goal groups
    #     while True:
    #         next_nodes: list[tuple[int, int | None]] = []
    #         for node in nodes:
    #             for final_node in final_nodes:
    #                 if node[1] == final_node[0]:
    #                     next_nodes.append(node)
    #                 else:
    #                     pass

    #         # Remove duplicated nodes
    #         next_nodes = list(set(next_nodes))

    #         if len(next_nodes) == 0:
    #             break
    #         else:
    #             final_nodes = next_nodes

    #     return [node[0] for node in final_nodes]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[NDArray[np.int_], dict[str, Any]]:

        super().reset(seed=seed, options=options)

        # # define the current subtask
        # self.current_subtasks: list[Subtask] = [
        #     self.subtask_dict[subtask_id] for subtask_id in self._find_first_subtasks()
        # ]

        # # define the reward function for the current subtask
        # self.rewarded_subtasks: list[Subtask]
        # if self.reward_config["reward_option"] == "final_goal":
        #     self.rewarded_subtasks = [
        #         subtask
        #         for subtask in self.current_subtasks
        #         if subtask.next_subtask == "terminal"
        #     ]

        # elif self.reward_config["reward_option"] == "intermediate_goal":
        #     self.rewarded_subtasks = self.current_subtasks
        # else:
        #     raise ValueError(
        #         f"Invalid reward option: {self.reward_config['reward_option']}"
        #     )

        obs = self.get_obs()
        info: dict[str, Any] = self._get_info()

        return obs, info

    def _gen_grid(self, width, height) -> None:
        self.grid = Grid(width, height, self.world)

        # make a list of all the objects you want to spawn in the env
        env_object_config = [
            # surrounding wall
            EnvObjectGroup(
                obj_type="wall",
                group_idx=0,
                pos=((0, 0, 7, 10),),
                color="grey",
                spawned_subtask_idxs=(0, 1),
                fill_mode="empty"
                ),
            # middle walls
            EnvObjectGroup(
                obj_type="wall",
                group_idx=1,
                pos=((3, 1),),
                color="grey",
                spawned_subtask_idxs=(0, 1),
                fill_mode="empty"
                ),
            EnvObjectGroup(
                obj_type="wall",
                group_idx=2,
                pos=((3, 8),),
                color="grey",
                spawned_subtask_idxs=(0, 1),
                fill_mode="empty"
                ),
            EnvObjectGroup(
                obj_type="wall",
                group_idx=2,
                pos=((3, 4, 1, 2),),
                color="grey",
                spawned_subtask_idxs=(0, 1),
                fill_mode="empty"
                ),
        ]

        # add the goals for the current subtask
        for subtask_idx, data in self.hlmdp_data.subtask_data.items():
            if subtask_idx == self.subtask_idx:
                env_object_config.append(
                    EnvObjectGroup(
                        obj_type="goal",
                        group_idx=subtask_idx,
                        pos=data.final_state,
                        color="green",
                        spawned_subtask_idxs=(subtask_idx),
                        fill_mode="empty"
                    )
                )

        obj_group_dict: dict[str, dict[int, ObjGroupT]] = {}

        # Place objects
        for obj_group_config in env_object_config:
            obj_type: str = obj_group_config.obj_type
            group_index: int = obj_group_config.group_idx

            if obj_type not in obj_group_dict:
                obj_group_dict[obj_type] = {}
            else:
                pass

            obj_group_dict[obj_type][group_index] = ObjectGroup(
                object_options=self.object_options, **asdict(obj_group_config)
            )
            obj_group_dict[obj_type][group_index].put_objects(self.grid, self.world)

        self.obj_group_dict = obj_group_dict

        # this has to be done before the agents are placed
        self.init_grid: Grid = self.grid.copy()

        ########################
        # all the stuff above here has to be done before the agents are placed
        # espeically the init_grid thing
        ########################
        # Place the agents
        # pick the init_pos based on the subtask (assuming it is an init pos and not a distribution to sample from)
        init_pos = self.hlmdp_data.subtask_data[self.subtask_idx].init_state_dist[1]
        assert len(self.agents) == len(init_pos)
        for agent, pos in zip(self.agents, init_pos):
            self.place_agent(agent, pos)

        # # Initialize detectors
        # self.detectors: list[Detector] = [
        #     Detector(**detector_config) for detector_config in self.detector_config
        # ]

    def get_obs(self) -> dict[str: NDArray[np.int_]]:
        if self.obs_type == "dict":
            obs: dict[str, Any] = {}

            for i, agent in enumerate(self.agents):
                agent_obs: list[tuple[int, int]] = [agent.pos]

                if self.observation_option == "final_goal":
                    agent_obs.append(self.final_goal[i])
                elif self.observation_option == "intermediate_goal":
                    agent_obs.extend(self.agent_goals[i])
                else:
                    raise ValueError(
                        f"Invalid observation option: {self.observation_option}"
                    )

                obs[str(i)] = np.array(agent_obs).flatten()

        elif self.obs_type == "array":
            # TODO you should probably make this an array
            obs = np.zeros((3, self.num_agents))

        return obs

    def get_env_info(self) -> EnvInfo:
        env_info = EnvInfo(
            state_shape = self.width * self.height * self.world.encode_dim,
            obs_shape = int(np.prod(self.observation_space.shape)),
            n_actions = len(self.actions),
            n_agents = self.num_agents
        )
        return env_info

    def get_avail_actions(self, verbose=False) -> list[list[bool]]:
        """gets available actions for each agent

        Returns
        -------
        list[list[bool]]
            available actions for each agent
        """
        avail_actions = []
        for agent in self.agents:
            avail_actions.append(self._get_avail_actions_agent(agent))

        if verbose:
            self.print_avail_actions_str(avail_actions)

        return avail_actions

    def print_avail_actions_str(self, avail_actions):
        if self.actions == NavigationActions:
            actions = [a.name for a in NavigationActions]

            print("Available actions")
            for agent_idx, avail_actions_agent in enumerate(avail_actions):
                action_str = ""
                for i, avail in enumerate(avail_actions_agent):
                    if avail:
                        action_str += f"{actions[i]},  "

                print(f"Agent: {agent_idx} --- {action_str}")

    def _get_avail_actions_agent(self, agent: Agent) -> list[bool]:
        # we only handle the case of NavigationActions
        ## other action sets need to define their own rules for action availability
        if self.actions == NavigationActions:
            avail_actions = []
            for action in NavigationActions:
                # we will handle the logic for "stay" later after the movement actions
                if action.name != "stay":
                    avail_actions.append(1)

            neighbor_positions = agent.neighbor_pos
            for i, pos in enumerate(neighbor_positions):
                neighbor_cell = self.grid.get(*pos)
                if (neighbor_cell is None) or (neighbor_cell.can_overlap()):
                    continue
                else:
                    avail_actions[i] = 0

            # add representation of "stay" action, which is always available
            avail_actions.insert(0, 1)

        return avail_actions

    #############
    # general env step logic
    #############
    def step(
        self,
        actions: NDArray[np.int_],
    ) -> tuple[NDArray[np.int_], float, bool, bool, dict[str, Any]]:

        self.step_count += 1

        # TODO dude wtf, this "actual action" thing is just wrong
        ## this only appears in the labyrinth env, and not in the CTF env
        ## Ok, so here's what this would do. Imagine deterministic state transitions and deterministic order for resolving agent actions.
        ## If agent 1 selected "up", and there was another agent in that state, it will end up staying in its current state (simulating a collision).
        ## This should NOT be treated as if agent 1 selected "stay still". Agent 1's expected return should update as if it took action "up", but it stayed in its current state. That will basically teach it to avoid attempting to go "up" when another agent it there. In the other case where we manually replace its action, we are messing with the learning process by manually changing the agent's policy.
        ### That could have serious implications for training on-policy RL methods, so this is really really important to get right. This is literally putting a huge, manual bias on the agent's policy that should not be there.
        ## The reward should NOT be based on the "actual_actions", but should be based on the action the agent selected.
        actual_actions: list[int] = self._move_agents(actions)

        obs = self.get_obs()
        reward: float = self._reward(np.array(actual_actions))
        terminated: bool = self._terminated()
        truncated: bool = False
        info: dict[str, Any] = self._get_info()

        return obs, reward, terminated, truncated, info

    #############
    # agent movement
    #############
    def _move_agents(self, actions: list[int]) -> list[int]:
        """
        Move agents based on the actions.

        Parameters
        ----------
        actions : list[int]
            Actions to take.

        Returns
        -------
        actual_actions : list[int]
            Actual actions taken.
        """
        # Randomly generate the order of the agents by indices using self.np_random.
        agent_indices: list[int] = list(range(self.num_agents))
        actual_actions: list[int] = [0 for _ in range(self.num_agents)]
        self.np_random.shuffle(agent_indices)
        for i in agent_indices:
            actual_action: int = self._move_agent(actions[i], self.agents[i])
            actual_actions[i] = actual_action

        return actual_actions

    def _move_agent(self, action: int, agent: Agent) -> int:
        """
        Move agents based on the action.

        Parameters
        ----------
        action : int
            Action to take.
        agent : Agent
            Agent to move.

        Returns
        -------
        actual_action : int
            Actual action taken.
        """

        next_pos: Position

        assert agent.pos is not None

        next_pos = agent.pos + agent.dir_to_vec[action]
        available_pos: list[Position] = self._get_available_pos(agent)

        next_pos_in_available_pos: bool = False
        for pos in available_pos:
            if np.array_equal(pos, next_pos):
                next_pos_in_available_pos = True
                break
            else:
                pass

        if next_pos_in_available_pos:
            action_probs: list[float] = []
            for pos in available_pos:
                action_probs.append(
                    self.p_intended_movement
                    if np.array_equal(pos, next_pos)
                    else (1 - self.p_intended_movement) / (len(available_pos) - 1)
                )

            # Normalize the action probabilities
            action_probs = np.array(action_probs) / np.sum(action_probs)
            avail_pos_indices: list[int] = list(range(len(available_pos)))

            next_pos_index = self.np_random.choice(avail_pos_indices, p=action_probs)
            next_pos = available_pos[next_pos_index]

            actual_action_vec: NDArray[np.int_] = next_pos - agent.pos
            actual_action: int = np.where(
                np.all(actual_action_vec == agent.dir_to_vec, axis=1)
            )[0][0]

            next_cell: WorldObjT | None = self.grid.get(*next_pos)
            if next_cell is None:
                agent.move(next_pos, self.grid, self.init_grid, bg_color=None)
            elif next_cell.can_overlap():
                agent.move(
                    next_pos, self.grid, self.init_grid, bg_color=next_cell.bg_color
                )
            else:
                raise ValueError(
                    f"Invalid action f{action} and position f{next_pos} for agent {agent.index}. Available positions: {available_pos}"
                )
        else:
            actual_action: int = self.actions.stay

        return actual_action

    def _get_available_pos(self, agent: Agent) -> list[Position]:
        possible_pos: list[Position] = []

        for direction in agent.dir_to_vec:
            next_pos: Position = agent.pos + direction

            if (
                next_pos[0] < 0
                or next_pos[1] < 0
                or next_pos[0] >= self.width
                or next_pos[1] >= self.height
            ):
                pass
            else:
                next_cell: WorldObjT | None = self.grid.get(*next_pos)

                if self.grid.get(*next_pos) is None:
                    possible_pos.append(next_pos)
                elif next_cell.can_overlap():
                    possible_pos.append(next_pos)
                else:
                    pass

        return possible_pos

    #############
    # reward function
    #############
    def _reward(self, actions: NDArray[np.int_]) -> float:
        reward: float = 0.0

        # 1. Movement penalty for each agent if an action is not "stay"
        # reward += self.reward_config["movement_reward"] * np.sum(
        #     actions != self.actions.stay
        # )

        # # 2. Reward for each agent if it is on its assigned goal
        # agent_goal_statuses: list[int] = []
        # for agent in self.agents:
        #     agent_goal_statuses.append(
        #         self._is_agent_on_assigned_goal(
        #             (agent.pos[0], agent.pos[1]), agent.index, self.current_subtasks
        #         )
        #     )

        # num_goaled_agents: int = 0
        # for agent in self.agents:
        #     for subtask in self.rewarded_subtasks:
        #         if (
        #             agent.index in subtask.assigned_agent_goal
        #             and (agent.pos[0], agent.pos[1])
        #             == subtask.assigned_agent_goal[agent.index]
        #         ):
        #             num_goaled_agents += 1
        #         else:
        #             pass

        # reward += num_goaled_agents * self.reward_config["agent_on_goal_reward"]

        # # 3. Penalty for each agent if it moves away from its assigned goal though it was on it
        # for agent, action in zip(self.agents, actions):
            # this doesn't work if you have stochastic transitions
            ## it only works if you manually update "action" based on some other logic, but that is like the "actual_actions" thing in that it is not right according to RL theory
            ## the better way to do this is just grab previous_pos before moving the agents and input it to the reward function
        #     prev_pos: Position = self._get_previous_agent_pos(action, agent)
        #     prev_agent_goal: int = self._is_agent_on_assigned_goal(
        #         (prev_pos[0], prev_pos[1]), agent.index, self.rewarded_subtasks
        #     )
        #     curr_agent_goal: int = self._is_agent_on_assigned_goal(
        #         (agent.pos[0], agent.pos[1]), agent.index, self.rewarded_subtasks
        #     )

        #     # If the reward option is "final_goal", the penalty is given only if the agent was on the final goal.
        #     if prev_agent_goal != -1 and prev_agent_goal != curr_agent_goal:
        #         reward += self.reward_config["agent_move_away_from_goal_reward"]
        #     else:
        #         pass

        # # 4. Reward for all agents if they are on their assigned goals on the same goal group and unlock the door
        # # If the door is already unlocked, the reward is not given.
        # for subtask in self.current_subtasks:
        #     all_conditions_satisfied: bool = True
        #     for trigger in subtask.triggers:
        #         if not trigger.is_condition_satisfied(self.agents, subtask):
        #             all_conditions_satisfied = False
        #             break
        #         else:
        #             pass

        #     all_agents_on_goals: bool = True
        #     for agent in self.agents:
        #         if (
        #             agent.index in subtask.assigned_agent_goal
        #             and (agent.pos[0], agent.pos[1])
        #             == subtask.assigned_agent_goal[agent.index]
        #         ):
        #             pass
        #         else:
        #             all_agents_on_goals = False
        #             break

        #     all_conditions_satisfied = all_conditions_satisfied and all_agents_on_goals

        #     if all_conditions_satisfied:
        #         for trigger in subtask.triggers:
        #             trigger.trigger_action(self.obj_group_dict, self.grid)
        #             trigger.trigger_action(self.obj_group_dict, self.init_grid)

        #         if subtask in self.rewarded_subtasks:
        #             reward += self.reward_config["all_agents_on_goal_reward"]
        #         else:
        #             pass

        #         if subtask.next_subtask != "terminal":
        #             self.current_subtasks = [self.subtask_dict[subtask.next_subtask]]

        #         if reward_config["reward_option"] == "intermediate_goal":
        #             self.rewarded_subtasks = self.current_subtasks
        #         else:
        #             pass

        #         break
        #     else:
        #         pass

        return reward

    def _get_previous_agent_pos(self, action: int, agent: Agent) -> Position:
        previous_pos: Position

        assert agent.pos is not None

        previous_pos = agent.pos - agent.dir_to_vec[action]

        return previous_pos

    def _is_agent_on_assigned_goal(
        self, pos: tuple[int, int], agent_index: int
    ) -> int:
        return -1

    # def _is_agent_on_assigned_goal(
    #     self, pos: tuple[int, int], agent_index: int, considered_subtasks: list[Subtask]
    # ) -> int:

        # for subtask in considered_subtasks:
        #     if pos == subtask.assigned_agent_goal[agent_index]:
        #         return subtask.goal_group_index
        #     else:
        #         pass

        return -1

    #############
    # termination function
    #############
    def _terminated(self) -> bool:
        """
        Returns
        -------
        bool
            terminated tells whether the env is terminated or not
        """
        terminated = self._agents_reached_terminal_goal()
        # terminated = self._agents_detected() or self._agents_reached_terminal_goal()

        return terminated

    def _agents_detected(self) -> bool:
        detected: bool = False
        for detector in self.detectors:
            if detector.detect_agents(self.agents, self.obj_group_dict, self.np_random):
                detected = True
            else:
                pass

        return detected

    def _agents_reached_terminal_goal(self) -> bool:
        for agent in self.agents:
            if agent.pos is None:
                return False
            elif not self._is_agent_on_terminal_goal(agent.pos):
                return False
            else:
                pass

        return True

    def _is_agent_on_terminal_goal(self, pos: Position) -> bool:
        for goal_pos in self.final_goal:
            if pos[0] == goal_pos[0] and pos[1] == goal_pos[1]:
                return True

        return False


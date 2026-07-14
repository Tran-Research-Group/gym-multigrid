from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from gym_multigrid.core.agent import Agent
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.world import DefaultWorld
from gym_multigrid.multigrid import MultiGridEnv


class CommsSequenceEnv(MultiGridEnv):
    """A simple no-movement coordination environment over time.

    Each agent chooses one action per step from a discrete action set. The
    environment samples a hidden target sequence of length ``sequence_length``.
    One designated agent starts with that target sequence in its observation,
    while the others begin with an empty observation history. Agents receive
    a reward when they choose the next action in the sequence, and the episode
    ends when all agents complete their assigned sequence.

    The observation for each agent is a fixed-size one-hot encoding of the
    last ``sequence_length`` actions taken by that agent. The global state is a
    fixed-size one-hot encoding of the last ``sequence_length`` team actions.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        n_agents: int = 4,
        n_actions: int = 5,
        sequence_length: int = 3,
        max_steps: int = 10,
        success_reward: float = 1.0,
        failure_penalty: float = 0.0,
        informed_agent_index: int = 0,
        render_mode: str = "rgb_array",
    ) -> None:
        if n_agents <= 0:
            raise ValueError("n_agents must be positive")
        if n_actions <= 1:
            raise ValueError("n_actions must be at least 2")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

        self.n_agents = n_agents
        self.n_actions = n_actions
        self.sequence_length = sequence_length
        self.max_steps = max_steps
        self.success_reward = success_reward
        self.failure_penalty = failure_penalty
        self.informed_agent_index = informed_agent_index % n_agents
        self.render_mode = render_mode

        self.world = DefaultWorld
        agents = [Agent(self.world, index=i) for i in range(n_agents)]

        super().__init__(
            agents=agents,
            width=1,
            height=1,
            world=self.world,
            render_mode=render_mode,
            partial_obs=False,
            max_steps=max_steps,
        )

        self.action_space = spaces.Discrete(n_actions)
        self.ac_dim = n_actions
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.n_agents, self.sequence_length, self.n_actions),
            dtype=np.float32,
        )

        self.target_sequence: NDArray[np.int64] | None = None
        self.agent_progress: NDArray[np.int64] = np.zeros(n_agents, dtype=np.int64)
        self.agent_completed: NDArray[np.bool_] = np.zeros(n_agents, dtype=np.bool_)
        self._action_histories: list[list[int]] = [[] for _ in range(n_agents)]
        self._obs_histories: list[list[int]] = [[] for _ in range(n_agents)]
        self._team_history: list[int] = []
        self.step_count = 0

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height, self.world)

    def _reset_agents(self) -> None:
        for agent in self.agents:
            agent.reset()
            self.place_agent(agent, pos=(0, 0), reset_agent_status=True)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)

        self.target_sequence = np.random.randint(
            0, self.n_actions, size=self.sequence_length
        )
        self.agent_progress = np.zeros(self.n_agents, dtype=np.int64)
        self.agent_completed = np.zeros(self.n_agents, dtype=np.bool_)
        self._action_histories = [[] for _ in range(self.n_agents)]
        self._obs_histories = [[] for _ in range(self.n_agents)]
        self._team_history = []
        self.step_count = 0

        for agent_idx in range(self.n_agents):
            self._obs_histories[agent_idx] = []
            if agent_idx == self.informed_agent_index:
                self._obs_histories[agent_idx] = self.target_sequence.astype(
                    int
                ).tolist()

        obs = self._get_obs()
        info = self._get_info(terminated=False, truncated=False)
        return obs, info

    def step(
        self, actions: NDArray[np.int64] | list[int] | int
    ) -> tuple[NDArray[np.float32], NDArray[np.float64], bool, bool, dict[str, Any]]:
        if isinstance(actions, (int, np.integer)):
            action_array = np.full(self.n_agents, int(actions), dtype=np.int64)
        else:
            action_array = np.asarray(actions, dtype=np.int64)
            if action_array.shape != (self.n_agents,):
                raise ValueError(
                    f"Expected actions of shape {(self.n_agents,)}, got {action_array.shape}"
                )

        rewards = np.zeros(self.n_agents, dtype=np.float64)
        self.step_count += 1

        for agent_idx in range(self.n_agents):
            action = int(action_array[agent_idx])
            if action < 0 or action >= self.n_actions:
                raise ValueError(f"Invalid action {action} for agent {agent_idx}")

            if self.agent_completed[agent_idx]:
                self._action_histories[agent_idx].append(action)
                self._obs_histories[agent_idx].append(action)
                self._team_history.append(action)
                continue

            self._action_histories[agent_idx].append(action)
            self._obs_histories[agent_idx].append(action)
            self._team_history.append(action)

            expected_action = int(self.target_sequence[self.agent_progress[agent_idx]])
            if action == expected_action:
                rewards[agent_idx] += self.success_reward
                self.agent_progress[agent_idx] += 1
                if self.agent_progress[agent_idx] >= self.sequence_length:
                    self.agent_completed[agent_idx] = True
            else:
                rewards[agent_idx] += self.failure_penalty

        terminated = bool(np.all(self.agent_completed))
        truncated = bool(self.step_count >= self.max_steps)
        obs = self._get_obs()
        info = self._get_info(terminated=terminated, truncated=truncated)
        return obs, rewards, terminated, truncated, info

    def _get_obs(self) -> NDArray[np.float32]:
        obs = np.zeros(
            (self.n_agents, self.sequence_length, self.n_actions),
            dtype=np.float32,
        )
        for agent_idx in range(self.n_agents):
            history = self._obs_histories[agent_idx]
            for offset, action in enumerate(history[-self.sequence_length :]):
                obs[agent_idx, offset, action] = 1.0
        return obs

    def _get_state(self) -> NDArray[np.float32]:
        state = np.zeros((self.sequence_length, self.n_actions), dtype=np.float32)
        for offset, action in enumerate(self._team_history[-self.sequence_length :]):
            state[offset, action] = 1.0
        return state

    def _get_info(self, terminated: bool, truncated: bool) -> dict[str, Any]:
        return {
            "episode_length": self.step_count,
            "completed_agents": int(np.sum(self.agent_completed)),
            "agent_progress": self.agent_progress.tolist(),
            "agent_completed": self.agent_completed.astype(bool).tolist(),
            "target_sequence": self.target_sequence.tolist()
            if self.target_sequence is not None
            else None,
            "informed_agent": self.informed_agent_index,
            "terminated": terminated,
            "truncated": truncated,
        }

    def _get_state_size(self) -> int:
        return self.sequence_length * self.n_actions

    def _get_obs_size(self) -> int:
        return self.n_agents * self.sequence_length * self.n_actions

    def _set_action_space(self) -> tuple[spaces.Space, int]:
        return spaces.Discrete(self.n_actions), self.n_actions

    def _set_observation_space(self) -> spaces.Space:
        return spaces.Box(
            low=0,
            high=1,
            shape=(self.n_agents, self.sequence_length, self.n_actions),
            dtype=np.float32,
        )

    def render(self):
        return None

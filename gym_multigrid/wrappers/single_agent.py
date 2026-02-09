"""
Wrapper for single-ego-agent environments to work with Stable Baselines3.

Handles two SB3 compatibility issues:
1. Action format: SB3 outputs a single int, but env.step() expects a list.
2. Observation dtype: SB3 applies VecTransposeImage to 3D uint8 observations
   (assumes they are images). Casting to float32 prevents this.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SingleAgentWrapper(gym.Wrapper):
    """
    Wraps a multi-agent environment with exactly one ego agent for SB3 compatibility.

    Parameters
    ----------
    env : gym.Env
        Environment with exactly one ego agent.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        if not hasattr(env.unwrapped, "ego_agent_indices"):
            raise ValueError(
                "SingleAgentWrapper requires an environment with "
                "'ego_agent_indices' attribute."
            )

        if len(env.unwrapped.ego_agent_indices) != 1:
            raise ValueError(
                f"SingleAgentWrapper requires exactly 1 ego agent, "
                f"but found {len(env.unwrapped.ego_agent_indices)}."
            )

        # Override observation space dtype to float32.
        # Prevents SB3 from applying VecTransposeImage which
        # triggers on 3D uint8 Box and scrambles the observation.
        old_space = env.observation_space
        self.observation_space = spaces.Box(
            low=0.0,
            high=255.0,
            shape=old_space.shape,
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs.astype(np.float32), info

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = [int(action.item())]
        elif isinstance(action, (int, np.integer)):
            action = [int(action)]
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs.astype(np.float32), reward, terminated, truncated, info

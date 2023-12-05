

from __future__ import annotations

import gymnasium as gym
from gymnasium import Env


from gym_multigrid.core.agent import Actions
from gym_multigrid.multigrid import MultiGridEnv

import enum

class ManualControl:
    def __init__(
        self,
        env: Env,
        seed=None,
    ) -> None:
        self.env = env
        self.seed = seed
        self.closed = False

    def start(self):
        self.reset(self.seed)
        while not self.closed:
            key = input()
            self.key_handler(key)

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.env.render()

    def step(self, actions: Actions):
        _, reward, terminated, truncated, _ = self.env.step(actions)
        if terminated:
            print("Terminated")
            self.reset(self.seed)
        elif truncated:
            print("truncated")
            self.reset(self.seed)
        else:
            self.env.render()


    def key_handler(self, event):

        key: str = event
        print("pressed", key)

        if key == "escape":
            self.env.close()
            return
        if key == "backspace":
            self.reset()
            return
    
        key_to_action = {
            "left": Actions.left,
            "right": Actions.right,
            "up": Actions.forward,
            "space": Actions.toggle,
            "pageup": Actions.pickup,
            "pagedown": Actions.drop,
            "tab": Actions.pickup,
            "left shift": Actions.drop,
            "enter": Actions.done,
        }
        if key in key_to_action.keys():
            action = [key_to_action[key]]
            
            self.step(action)
        
        else:
            print(key)



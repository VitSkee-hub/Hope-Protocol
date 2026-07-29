# envs/hope_reward_wrapper.py

import gymnasium as gym
from typing import Tuple, Any

from hope.hope_protocol import HopeProtocol


class HopeRewardWrapper(gym.Wrapper):
    """
    A Gymnasium wrapper that applies the Hope Protocol
    to rewards returned by the environment.
    """

    def __init__(self, env: gym.Env, hope_protocol: HopeProtocol):
        super().__init__(env)
        self.hope_protocol = hope_protocol
        self.step_count = 0

    def step(self, action) -> Tuple[Any, float, bool, bool, dict]:
        observation, reward, terminated, truncated, info = self.env.step(action)

        transformed_reward = self.hope_protocol.transform(
            reward,
            step=self.step_count,
        )

        # Optional: log both rewards for analysis
        info = dict(info)
        info["raw_reward"] = reward
        info["transformed_reward"] = transformed_reward

        self.step_count += 1

        return observation, transformed_reward, terminated, truncated, info

def reset(self, **kwargs):
    self.step_count = 0
    self.hope_protocol.reset()
    return self.env.reset(**kwargs)


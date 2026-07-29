import gymnasium as gym

from hope.hope_protocol import HopeConfig, HopeProtocol
from envs.hope_reward_wrapper import HopeRewardWrapper


env = gym.make("FrozenLake-v1", is_slippery=True)


hope = HopeProtocol(
    HopeConfig(
        p_hope=0.5,          # high so we SEE it
        min_multiplier=2.0,
        max_multiplier=3.0,
        mode="positive_only",
    )
)

wrapped_env = HopeRewardWrapper(env, hope)

obs, info = wrapped_env.reset()

print("Reset:", obs)

for step in range(10):
    action = wrapped_env.action_space.sample()
    obs, reward, terminated, truncated, info = wrapped_env.step(action)

    print(
        f"step={step:02d} | "
        f"raw={info['raw_reward']:.2f} | "
        f"transformed={reward:.2f}"
    )

    if terminated or truncated:
        print("Episode ended.")
        break

import gymnasium as gym
import numpy as np

from agents.q_learning import QLearningAgent
from hope.hope_protocol import HopeConfig, HopeProtocol
from envs.hope_reward_wrapper import HopeRewardWrapper


N_EPISODES = 1000
MAX_STEPS = 100

# --- Environment ---
base_env = gym.make("FrozenLake-v1", is_slippery=True)

hope = HopeProtocol(
    HopeConfig(
        p_hope_base=0.01,        # very low by default
        p_hope_boost=0.2,        # kicks in after stagnation
        stagnation_threshold=500,
        min_multiplier=2.0,
        max_multiplier=3.0,
        mode="positive_only",
    )
)


env = HopeRewardWrapper(base_env, hope)

n_states = env.observation_space.n
n_actions = env.action_space.n

# --- Agent (IDENTICAL to baseline) ---
agent = QLearningAgent(
    n_states=n_states,
    n_actions=n_actions,
    alpha=0.1,
    gamma=0.99,
    epsilon=0.3,
)

episode_rewards = []

for episode in range(N_EPISODES):
    state, _ = env.reset()
    total_reward = 0.0

    for step in range(MAX_STEPS):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        agent.update(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        if done:
            break

    episode_rewards.append(total_reward)

    if (episode + 1) % 100 == 0:
        print(
            f"[HOPE] Episode {episode + 1:4d} | "
            f"avg reward (last 100): {np.mean(episode_rewards[-100:]):.3f}"
        )

env.close()

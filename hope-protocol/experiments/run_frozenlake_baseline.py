import gymnasium as gym
import numpy as np

from agents.q_learning import QLearningAgent


N_EPISODES = 5000
MAX_STEPS = 100


env = gym.make("FrozenLake-v1", is_slippery=True)
n_states = env.observation_space.n
n_actions = env.action_space.n

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
        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated
        agent.update(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        if done:
            break

    episode_rewards.append(total_reward)

    if (episode + 1) % 100 == 0:
        print(
            f"Episode {episode + 1:4d} | "
            f"avg reward (last 100): {np.mean(episode_rewards[-100:]):.3f}"
        )

env.close()

import csv
import os
import time
import random
import numpy as np
import gymnasium as gym

from agents.q_learning import QLearningAgent
from hope.hope_protocol import HopeConfig, HopeProtocol
from envs.hope_reward_wrapper import HopeRewardWrapper


def run(seed: int, use_hope: bool, epsilon: float, episodes: int = 5000, max_steps: int = 100):
    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)

    base_env = gym.make("FrozenLake-v1", is_slippery=True)
    base_env.reset(seed=seed)
    base_env.action_space.seed(seed)

    if use_hope:
        hope = HopeProtocol(
            HopeConfig(
                p_hope_base=0.01,
                p_hope_boost=0.2,
                stagnation_threshold=500,
                min_multiplier=2.0,
                max_multiplier=3.0,
                mode="positive_only",
            )
        )
        env = HopeRewardWrapper(base_env, hope)
    else:
        env = base_env

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    agent = QLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        alpha=0.1,
        gamma=0.99,
        epsilon=epsilon,
    )

    # Per-episode logs
    success_raw = np.zeros(episodes, dtype=np.int32)
    reward_transformed_sum = np.zeros(episodes, dtype=np.float32)

    ttfs = None
    longest_success_streak = 0
    longest_fail_streak = 0
    cur_success_streak = 0
    cur_fail_streak = 0

    for ep in range(episodes):
        state, info = env.reset()
        ep_transformed_total = 0.0
        ep_success = 0

        for _ in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            agent.update(state, action, reward, next_state, done)

            state = next_state
            ep_transformed_total += float(reward)

            # True success is based on RAW reward (goal reached)
            raw_r = float(info.get("raw_reward", 0.0)) if use_hope else float(reward)
            if raw_r > 0:
                ep_success = 1

            if done:
                break

        success_raw[ep] = ep_success
        reward_transformed_sum[ep] = ep_transformed_total

        if ep_success == 1:
            if ttfs is None:
                ttfs = ep + 1  # 1-indexed for human readability
            cur_success_streak += 1
            cur_fail_streak = 0
            longest_success_streak = max(longest_success_streak, cur_success_streak)
        else:
            cur_fail_streak += 1
            cur_success_streak = 0
            longest_fail_streak = max(longest_fail_streak, cur_fail_streak)

    env.close()

    return {
        "seed": seed,
        "use_hope": use_hope,
        "epsilon": epsilon,
        "episodes": episodes,
        "ttfs": ttfs if ttfs is not None else -1,
        "final_success_rate_500": float(np.mean(success_raw[-500:])),
        "final_success_rate_1000": float(np.mean(success_raw[-1000:])),
        "longest_success_streak": int(longest_success_streak),
        "longest_fail_streak": int(longest_fail_streak),
        "success_raw": success_raw,
        "reward_transformed_sum": reward_transformed_sum,
    }


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.astype(np.float32)
    c = np.cumsum(np.insert(x.astype(np.float32), 0, 0.0))
    return (c[window:] - c[:-window]) / window


def plateau_episode(success_raw: np.ndarray, window: int = 500, slope_eps: float = 1e-4, patience: int = 5) -> int:
    """
    Detect plateau as: rolling success rate slope is ~0 for `patience` consecutive windows.
    Returns episode index where plateau starts, or -1.
    """
    rm = rolling_mean(success_raw, window)
    if len(rm) < (patience + 2):
        return -1

    # Slope between consecutive rolling means
    slopes = np.diff(rm)

    # We consider plateau if abs(slope) stays below slope_eps for a while
    count = 0
    for i, s in enumerate(slopes):
        if abs(float(s)) < slope_eps:
            count += 1
            if count >= patience:
                # Map back to original episode index (approx)
                return int(i) + window
        else:
            count = 0
    return -1


def main():
    # Config
    eps = 0.3
    episodes = 5000
    seeds = list(range(10))  # bump to 30 later

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    summary_path = os.path.join(out_dir, f"summary_{stamp}.csv")

    rows = []

    for use_hope in [False, True]:
        for seed in seeds:
            result = run(seed=seed, use_hope=use_hope, epsilon=eps, episodes=episodes)
            plateau = plateau_episode(result["success_raw"], window=500, slope_eps=1e-4, patience=8)

            row = {
                "seed": result["seed"],
                "use_hope": result["use_hope"],
                "epsilon": result["epsilon"],
                "episodes": result["episodes"],
                "ttfs": result["ttfs"],
                "final_success_rate_500": result["final_success_rate_500"],
                "final_success_rate_1000": result["final_success_rate_1000"],
                "longest_success_streak": result["longest_success_streak"],
                "longest_fail_streak": result["longest_fail_streak"],
                "plateau_episode": plateau,
            }
            rows.append(row)

            label = "HOPE" if use_hope else "BASE"
            print(
                f"{label} seed={seed:02d} | ttfs={row['ttfs']:4d} | "
                f"final500={row['final_success_rate_500']:.3f} | "
                f"plateau={row['plateau_episode']}"
            )

    # Write summary CSV
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote: {summary_path}")


if __name__ == "__main__":
    main()

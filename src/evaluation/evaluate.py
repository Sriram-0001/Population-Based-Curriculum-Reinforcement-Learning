import gymnasium as gym
import numpy as np


def evaluate_detailed(model, episodes=50, seed=42):
    env = gym.make("MountainCar-v0")

    rewards = []
    lengths = []
    successes = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0
        steps = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        rewards.append(total_reward)
        lengths.append(steps)
        successes.append(obs[0] >= 0.5)

    return {
        "avg_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "avg_length": np.mean(lengths),
        "success_rate": np.mean(successes),
        "rewards": rewards,
        "lengths": lengths,
        "successes": successes
    }

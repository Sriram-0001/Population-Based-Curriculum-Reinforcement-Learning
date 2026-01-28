import gymnasium as gym
import numpy as np


def evaluate(model, episodes=20):
    env = gym.make("MountainCar-v0")
    rewards, success = [], []

    for _ in range(episodes):
        obs, _ = env.reset()
        total = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated

        rewards.append(total)
        success.append(obs[0] >= 0.5)

    return np.mean(rewards), np.mean(success)

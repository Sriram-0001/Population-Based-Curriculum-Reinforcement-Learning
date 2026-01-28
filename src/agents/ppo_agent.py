from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from src.envs.shaped_mountaincar import ShapedMountainCar


def make_env(seed, shaping_coef):
    def _init():
        env = gym.make("MountainCar-v0")
        env = ShapedMountainCar(env, shaping_coef)
        env.reset(seed=seed)
        return Monitor(env)
    return _init


def build_env(shaping_coef, n_envs=8):
    return DummyVecEnv([make_env(i, shaping_coef) for i in range(n_envs)])


def build_agent(shaping_coef):
    env = build_env(shaping_coef)
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
    )

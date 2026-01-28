import gymnasium as gym

class ShapedMountainCar(gym.Wrapper):
    def __init__(self, env, shaping_coef):
        super().__init__(env)
        self.shaping_coef = shaping_coef
        self.prev_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_pos = obs[0]
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        progress = obs[0] - self.prev_pos
        reward += self.shaping_coef * progress
        self.prev_pos = obs[0]
        return obs, reward, terminated, truncated, info

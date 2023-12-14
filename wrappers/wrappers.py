import numpy as np
import torch
import torchvision
import gym
from gym.spaces import Box

class SkipFrame(gym.Wrapper):
    def __init__(self, 
                 env, 
                 skip_frame):
        super().__init__(env)
        self._skip_frame = skip_frame

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip_frame):
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        transform = torchvision.transforms.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(self.shape, antialias=True), torchvision.transforms.Normalize(0, 255)]
        )
        observation = transform(observation).squeeze(0)
        return observation


class ResetWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        if isinstance(obs, np.ndarray) and len(obs.shape) > 1:
            return [(single_obs, {}) for single_obs in obs], info
        return obs, info

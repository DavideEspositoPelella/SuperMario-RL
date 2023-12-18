# This code takes inspiration from the wrappers in pytorch documentation: https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
import numpy as np
import torch
import torchvision
import gym
from gym.spaces import Box
from typing import Tuple, Union

class SkipFrame(gym.Wrapper):
    def __init__(self, 
                 env: gym.Env, 
                 skip_frame: int=4) -> None:
        """
        Initializes the SkipFrame wrapper.

        Args:
            - env (gym.Env): The environment.
            - skip_frame (int): Number of frames to skip. Default to 4.
        """
        super().__init__(env)
        self.skip_frame = skip_frame

    def step(self, 
             action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Skip several frames and repeat the given action, summing up the rewards.

        Args:
            - action (int): The action to repeat.
        
        Returns:
            - obs (np.ndarray): The observation.
            - total_reward (float): The sum of rewards.
            - terminated (bool): Whether the episode is over.
            - truncated (bool): Whether the episode is truncated.
            - info (dict): The info of the current step.
        """
        total_reward = 0.0
        for _ in range(self.skip_frame):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated:
                break
        return obs, total_reward, terminated, truncated, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, 
                 env: gym.Env) -> None:
        """
        Initializes the GrayScaleObservation wrapper.

        Args:
            - env (gym.Env): The environment.
        """
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, 
                    observation: np.ndarray) -> torch.Tensor:
        """
        Convert the observation to grayscale.

        Args:
            - observation (np.ndarray): The observation.
        
        Returns:
            - observation (torch.Tensor): The grayscale observation.
        """
        # transpose to have as output shape (C, H, W)
        observation = np.transpose(observation, (2, 0, 1))
        # convert to tensor and grayscale
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        transform = torchvision.transforms.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, 
                 env: gym.Env, 
                 shape: Union[int, Tuple[int, int]]) -> None:
        """
        Initializes the ResizeObservation wrapper.

        Args:
            - env (gym.Env): The environment.
            - shape (Union[int, Tuple[int, int]]): The shape of the resized observation.
        """
        
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, 
                    observation: torch.Tensor) -> torch.Tensor:
        """
        Resize the observation and normalize it.

        Args:
            - observation (torch.Tensor): The observation.
        
        Returns:
            - observation (torch.Tensor): The resized observation.
        """
        # resize the observation and normalize
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(self.shape, antialias=True), 
             torchvision.transforms.Normalize(0, 255)])
        # remove the first dimension
        observation = transform(observation).squeeze(0)
        return observation

class ResetWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        """Reset the environment."""
        obs, info = super().reset(**kwargs)
        if isinstance(obs, np.ndarray) and len(obs.shape) > 1:
            return [(single_obs, {}) for single_obs in obs], info
        return obs, info

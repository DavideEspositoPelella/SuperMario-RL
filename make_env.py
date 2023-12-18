  
import gym
from gym.wrappers import FrameStack 
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from wrappers.wrappers import SkipFrame, GrayScaleObservation, ResizeObservation
from nes_py.wrappers import JoypadSpace

import gym_super_mario_bros




def make_env(skip_frame: int=2, 
             stack: int=4, 
             resize_shape: int=42) -> gym.Env:
    """
    Creates the environment.

    Args:
        - skip_frame (int): Number of frames to skip. Default to 2.
        - stack (int): Number of frames to stack. Default to 4.
        - resize_shape (int): Size of the resized frame. Default to 42.

    Returns:
        - env (gym.Env): The environment.
    """
    env_ID = "SuperMarioBros-1-1-v0"
    if gym.__version__ < '0.26':
        env = gym_super_mario_bros.make(env_ID, new_step_api=True)
    else:
        env = gym_super_mario_bros.make(env_ID, render_mode='human', apply_api_compatibility=True)
    # wrap to skip frames, grayscale, stack and resize the observations 
    env = SkipFrame(env, skip_frame=skip_frame)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=resize_shape)
    # wrap to use the simple movement actions
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    if gym.__version__ < '0.26':
        env = FrameStack(env, 
                         num_stack=stack, 
                         new_step_api=True)
    else:
        env = FrameStack(env, 
                         num_stack=stack)    
    return env
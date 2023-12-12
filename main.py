import os

from args import get_args
from agents.ddqn_per import DDQN


import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import FrameStack
import numpy as np

from tensorboardX import SummaryWriter


args = get_args()

try:
    os.makedirs(args.log_dir)
except OSError:
    pass

#TODO the evaluation function must be updated to work with the new environment
def evaluate(algorithm: str='ddqn',
             episodes: int=20000) -> None:

    
    """
    Evaluate the agent with the specified algorithm for the specified number of episodes.

    Args:
        - algorithm (str): The algorithm to use to train the agent. Available options are 'ddqn', 'ddqn_per', 'a3c', 'dueling_ddqn', and 'sarsa'. Defaults to 'ddqn'.
        - episodes (int): The number of episodes to train for. Defaults to 20000.
    """
    env_ID = "SuperMarioBros-1-1-v0"
    if gym.__version__ < '0.26':
        env = gym_super_mario_bros.make(env_ID,
                                        new_step_api=True)

    else:
        env = gym_super_mario_bros.make(env_ID, 
                                        render_mode='human', 
                                        apply_api_compatibility=True)
    if algorithm == 'ddqn':
        agent = DDQN(episodes=episodes, 
                     prioritized=False)
    elif algorithm == 'ddqn_per':
        agent = DDQN(episodes=episodes, 
                     prioritized=True)
    else:
        raise ValueError("Invalid algorithm selected")
    
    # agent.load()
    # rewards = []
    # for episode in range(episodes):
    #     total_reward = 0
    #     done = False
    #     state, _ = env.reset()
    #     while not done:
    #         action = agent.act(state)
    #         state, reward, terminated, truncated, info = env.step(action)
    #         done = terminated or truncated
    #         total_reward += reward

    #     rewards.append(total_reward)

    # print('Mean Reward:', np.mean(rewards))

def train(algorithm: str='ddqn', 
          episodes: int=20000) -> None:
    """
    Train the agent with the specified algorithm for the specified number of episodes.

    Args:
        - algorithm (str): The algorithm to use to train the agent. Available options are 'ddqn', 'ddqn_per', 'a3c', 'dueling_ddqn', and 'sarsa'. Defaults to 'ddqn'.
        - episodes (int): The number of episodes to train for. Defaults to 20000.
    """
    if algorithm == 'ddqn':
        agent = DDQN(episodes=episodes, 
                     prioritized=False)
    elif algorithm == 'ddqn_per':
        agent = DDQN(episodes=episodes, 
                     prioritized=True)
    else:
        raise ValueError("Invalid algorithm selected")
    agent.train()
    agent.save()

def main():
    tb_writer = SummaryWriter(log_dir=args.save_dir)
    if args.train:
        train(algorithm=args.algorithm,
              episodes=args.episodes)
    if args.evaluate:
        evaluate(algorithm=args.algorithm, 
                 episodes=args.episodes)


if __name__ == '__main__':
    main()

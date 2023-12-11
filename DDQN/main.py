
# Import necessary libraries
import argparse
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import FrameStack
import numpy as np
from DDQN import DDQN
from DDQN_PER import DDQNPrioritized

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
        agent = DDQN(episodes=episodes)
    elif algorithm == 'ddqn_per':
        agent = DDQNPrioritized(episodes=episodes)
    # elif algorithm == 'a3c':
    #     agent = A3C(...)
    # elif algorithm == 'dueling_ddqn':
    #     agent = DuelingDDQN(...)
    # elif algorithm == 'sarsa':
    #     agent = SARSA(...)
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
        agent = DDQN(episodes=episodes)
    elif algorithm == 'ddqn_per':
        agent = DDQNPrioritized(episodes=episodes)
    # elif algorithm == 'a3c':
    #     agent = A3C(episodes=episodes)
    # elif algorithm == 'dueling_ddqn':
    #     agent = DuelingDDQN(episodes=episodes)
    # elif algorithm == 'sarsa':
    #     agent = SARSA(episodes=episodes)
    else:
        raise ValueError("Invalid algorithm selected")
    agent.train()
    agent.save()

def main():
    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('--episodes', 
                        type=int, 
                        default=20000, 
                        help='Number of episodes to train')
    parser.add_argument('--algorithm', 
                        type=str, 
                        default='ddqn', 
                        choices=['ddqn', 'ddqn_per', 'dueling_ddqn' 'a3c', 'sarsa'], 
                        help='The algorithm to use')
    args = parser.parse_args()
    if args.train:
        train(algorithm=args.algorithm,
              episodes=args.episodes)
    if args.evaluate:
        evaluate(algorithm=args.algorithm, 
                 episodes=args.episodes)


if __name__ == '__main__':
    main()

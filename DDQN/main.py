
# Import necessary libraries
import argparse
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import FrameStack
import numpy as np
from DDQN import DDQN

# Import the Policy from DDQN.py (assuming it's in the same directory)
from DDQN import MarioNet, SkipFrame, GrayScaleObservation, ResizeObservation

# Evaluate function
def evaluate(env=None, n_episodes=5, render=False):
    # Initialize the environment and model
    env_ID = "SuperMarioBros-1-1-v0"
    env_ID = "SuperMarioBros-1-1-v0"
    if gym.__version__ < '0.26':
        env = gym_super_mario_bros.make(env_ID,
                                        new_step_api=True)

    else:
        env = gym_super_mario_bros.make(env_ID, 
                                        render_mode='human', 
                                        apply_api_compatibility=True)
    model = DDQN(...)  # Initialize MarioNet with appropriate parameters
    model.load()  # Load model weights

    if render:
        env = gym_super_mario_bros.make(env_ID, new_step_api=True, render_mode="human")

    rewards = []
    for episode in range(n_episodes):
        total_reward = 0
        done = False
        state, _ = env.reset()
        while not done:
            action = model.act(state)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        rewards.append(total_reward)

    print('Mean Reward:', np.mean(rewards))

# Train function
def train(render=False):
    # Implementation for training the model
    agent = DDQN()
    agent.train()
    agent.save()

# Main function
def main():
    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-e', '--evaluate', action='store_true')
    args = parser.parse_args()

    if args.train:
        train(render=args.render)

    if args.evaluate:
        evaluate(render=args.render)

if __name__ == '__main__':
    main()

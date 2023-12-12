import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('--algorithm', 
                        type=str, 
                        default='ddqn', 
                        choices=['ddqn', 'ddqn_per', 'a3c', 'sarsa'], 
                        help='The algorithm to use')
    parser.add_argument('--episodes', 
                        type=int, 
                        default=20000, 
                        help='Number of episodes to train')
    parser.add_argument('--log-interval', 
                        type=int, 
                        default=10,
                        help='Log interval. Default to 10')
    parser.add_argument('--save-interval', 
                        type=int, 
                        default=100,
                        help='Save interval. Default to 100')
    parser.add_argument('--log-dir', 
                        default='./logs/',
                        help='where to save agent logs. Default to ./logs')
    parser.add_argument('--save-dir', 
                        default='./trained_models/',
                        help='Where to save agent logs, Default to ./trained_models/')
    parser.add_argument('--model', 
                        type=str, 
                        default='mario_net_0.chkpt', 
                        help='The model to continue training from')

    
    args = parser.parse_args()

    return args
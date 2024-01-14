import argparse

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('-t', 
                        '--train', 
                        action='store_true')
    parser.add_argument('-e', 
                        '--evaluate', 
                        action='store_true')
    parser.add_argument('--algorithm', 
                        type=str, 
                        default='a2c', 
                        choices=['ddqn', 'ddqn_per', 'a2c'], 
                        help='The algorithm to use.')
    parser.add_argument('--episodes', 
                        type=int, 
                        default=20000, 
                        help='Number of episodes to train.')
    parser.add_argument('--icm', 
                        action='store_true',
                        help='Include ICM (Intrinsic Curiosity Module) in training/evaluation.')
    parser.add_argument('--adaptive', 
                        action='store_true',
                        help='Use adaptive exploration.')
    parser.add_argument('--ou-noise',
                        action='store_true',
                        help='Use Ornstein-Uhlenbeck noise.')
    parser.add_argument('--tb', 
                        action='store_true', 
                        help='Enable TensorBoard logging.')
    parser.add_argument('--log-freq', 
                        type=int, 
                        default=100,
                        help='Log frequency. Default to 100')
    parser.add_argument('--save-freq', 
                        type=int, 
                        default=100,
                        help='Save frequency. Default to 100')
    parser.add_argument('--log-dir', 
                        default='./logs/',
                        help='Where to save agent logs. Default to ./logs/')
    parser.add_argument('--save-dir', 
                        default='./checkpoints/',
                        help='Where to save agent checkpoints, Default to ./checkpoints/')
    parser.add_argument('--model', 
                        type=str,
                        default='False', 
                        help='The model from which resume training or to evaluate.')
    
    args = parser.parse_args()

    return args

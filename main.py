import json
from pathlib import Path
import warnings
from args import get_args
import torch
from torch.utils.tensorboard import SummaryWriter

import gym

from agents.ddqn_agent import DDQNAgent
from agents.a2c_agent import A2CAgent
from util.util import create_dir, init_tensorboard, close_tb, set_seed
from config import Config
import make_env

warnings.filterwarnings("ignore", category=UserWarning, module='gym.envs.registration')

args = get_args()

def train(env:gym.Env,
          config: Config,
          algorithm: str,
          icm: bool,
          tb_writer: SummaryWriter,
          log_dir: Path, 
          save_dir: Path) -> None:
    """
    Train the agent with the specified algorithm for the specified number of episodes.

    Args:
        - env (gym.Env): The environment.
        - config (Config): The configuration object.
        - algorithm (str): The algorithm to use.
        - icm (bool): Whether to use ICM.
        - tb_writer (SummaryWriter): The TensorBoard SummaryWriter object.
        - log_dir (Path): The directory where to save TensorBoard logs.
        - save_dir (Path): The directory where to save trained models.
    """
    # define the agent
    if algorithm == 'ddqn':
        agent = DDQNAgent(env=env,
                          config=config,
                          prioritized=False, 
                          icm=icm, 
                          tb_writer=tb_writer,
                          log_dir=log_dir, 
                          save_dir=save_dir)
    elif algorithm == 'ddqn_per':
        agent = DDQNAgent(env=env, 
                          config=config, 
                          prioritized=True,
                          icm=icm,
                          tb_writer=tb_writer,
                          log_dir=log_dir,
                          save_dir=save_dir)
    elif algorithm == 'a2c':
        agent = A2CAgent(env=env, 
                         config=config,
                         icm=icm,
                         tb_writer=tb_writer,
                         log_dir=log_dir,
                         save_dir=save_dir)
    else:
        raise ValueError("Invalid algorithm selected!")
    # load model if specified and train
    if args.model != 'False':
        agent.load(args.model)
    agent.train()
    agent.save()


def evaluate(env: gym.Env,
             config: Config,
             algorithm: str,
             icm: bool,
             tb_writer: SummaryWriter,
             log_dir: Path, 
             save_dir: Path) -> None:
    """
    Evaluate the agent with the specified algorithm for the specified number of episodes.

    Args:
        - env (gym.Env): The environment.
        - config (Config): The configuration object.
        - algorithm (str): The algorithm to use.
        - icm (bool): Whether to use ICM.
        - tb_writer (SummaryWriter): The TensorBoard SummaryWriter object.
        - log_dir (Path): The directory where to save TensorBoard logs.
        - save_dir (Path): The directory where to save trained models.
    """
    # define the agent
    with torch.no_grad():
        if algorithm == 'ddqn':
            agent = DDQNAgent(env=env,
                              config=config, 
                              prioritized=False,
                              icm=icm, 
                              tb_writer=tb_writer,
                              log_dir=log_dir,
                              save_dir=save_dir)
        elif algorithm == 'ddqn_per':
            agent = DDQNAgent(env=env,
                              config=config, 
                              prioritized=True,
                              icm=icm, 
                              tb_writer=tb_writer,
                              log_dir=log_dir,
                              save_dir=save_dir)
        elif algorithm == 'a2c':
            agent = A2CAgent(env=env, 
                             config=config, 
                             icm=icm,
                             tb_writer=tb_writer,
                             log_dir=log_dir,
                             save_dir=save_dir)
        else:
            raise ValueError("Invalid algorithm selected!")
        # load model if specified and evaluate
        if args.model != 'False':
            agent.load(args.model)
            agent.evaluate(env)
        else:
            print("Error: No model to evaluate provided \n")
    

def main():
    # set seed
    seed = 2023
    set_seed(seed)
    # create save directory and log directory
    save_dir = create_dir(args.save_dir, args.algorithm, args.icm)
    log_dir = create_dir(args.log_dir, args.algorithm, args.icm)
    # initialize tensorboard
    if args.tb:
        tb_writer, tb_process, log_dir = init_tensorboard(log_dir)
    else:
        tb_writer, log_dir = None, None

    if args.algorithm == 'ddqn' or args.algorithm == 'ddqn_per':
        config = Config(skip_frame = 2, stack = 4, resize_shape = 42,
                        exploration_rate=0.5, exploration_rate_decay=0.999, exploration_rate_min=0.1,
                        memory_size=10000, burn_in=2000, alpha=0.7, beta=0.5, epsilon_buffer=0.01,
                        gamma=0.99, batch_size=64, lr=0.00001,
                        update_freq=3, sync_freq=100, episodes=args.episodes,
                        feature_size=288, eta=1.0, beta_icm=0.2, lambda_icm=0.1,
                        log_freq=args.log_freq, save_freq=args.save_freq)
    elif args.algorithm == 'a2c':
        config = Config(skip_frame = 4, stack = 4, resize_shape = 42, n_steps=5, lr=0.0007,
                        gamma=0.99, actor_lr=0.0001, critic_lr=0.0005, ent_coef=0.01, 
                        episodes=args.episodes, eta=0.1, beta_icm=0.2, lambda_icm=0.1,
                        log_freq=args.log_freq, save_freq=args.save_freq)

    
    # create the environment     
    env = make_env.make_env(skip_frame=config.skip_frame, 
                            stack=config.stack, 
                            resize_shape=config.resize_shape)
    
    # add config to tensorboard
    if args.tb:
        config_json = json.dumps(vars(config), indent=4)
        tb_writer.add_text('config', config_json)

    # train/evaluate
    if args.train:
        train(env, config, args.algorithm, args.icm, 
              tb_writer, log_dir, save_dir)
    if args.evaluate:
        evaluate(env, config, args.algorithm, args.icm, 
                 tb_writer, log_dir, save_dir)
    # close tensorboard
    if args.tb:
        close_tb(tb_writer, tb_process)

if __name__ == '__main__':
    main()

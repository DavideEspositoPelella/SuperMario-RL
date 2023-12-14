from pathlib import Path
import warnings
from args import get_args
import torch
from torch.utils.tensorboard import SummaryWriter

from agents.ddqn_agent import DDQNAgent
from utils.utils import create_dir, init_tensorboard, close_tb

warnings.filterwarnings("ignore", category=UserWarning, module='gym.envs.registration')



args = get_args()


def train(algorithm: str,
          episodes: int,
          icm: bool,
          tb_writer: SummaryWriter,
          log_dir: Path, 
          save_dir: Path,
          log_freq: int, 
          save_freq: int) -> None:
    """
    Train the agent with the specified algorithm for the specified number of episodes.

    Args:
        - algorithm (str): The algorithm to use.
        - episodes (int): The number of episodes to train.
        - icm (bool): Whether to use ICM.
        - tb_writer (SummaryWriter): The TensorBoard SummaryWriter object.
        - log_dir (Path): The directory where to save TensorBoard logs.
        - save_dir (Path): The directory where to save trained models.
        - log_freq (int): Log frequency.
        - save_freq (int): Model save frequency.
    """
    if algorithm == 'ddqn':
        agent = DDQNAgent(episodes=episodes,
                          prioritized=False, 
                          icm=icm, 
                          tb_writer=tb_writer,
                          log_dir=log_dir, 
                          save_dir=save_dir, 
                          log_freq=log_freq,
                          save_freq=save_freq)
        
    elif algorithm == 'ddqn_per':
        agent = DDQNAgent(episodes=episodes, 
                          prioritized=True,
                          icm=icm,
                          tb_writer=tb_writer,
                          log_dir=log_dir,
                          save_dir=save_dir, 
                          log_freq=log_freq,
                          save_freq=save_freq)
    else:
        raise ValueError("Invalid algorithm selected")
    
    if args.model != 'False':
        agent.load(args.model)
    agent.train()
    agent.save()


def evaluate(algorithm: str,
             episodes: int,
             icm: bool,
             tb_writer: SummaryWriter,
             log_dir: Path, 
             save_dir: Path, 
             log_freq, 
             save_freq) -> None:
    """
    Evaluate the agent with the specified algorithm for the specified number of episodes.

    Args:
        - algorithm (str): The algorithm to use.
        - episodes (int): The number of episodes to train.
        - icm (bool): Whether to use ICM.
        - tb_writer (SummaryWriter): The TensorBoard SummaryWriter object.
        - log_dir (Path): The directory where to save TensorBoard logs.
        - save_dir (Path): The directory where to save trained models.
        - log_freq (int): Log frequency.
        - save_freq (int): Model save frequency.
    """
    with torch.no_grad():
        if algorithm == 'ddqn':
            agent = DDQNAgent(episodes=episodes, 
                              prioritized=False,
                              icm=icm, 
                              tb_writer=tb_writer,
                              log_dir=log_dir,
                              save_dir=save_dir, 
                              log_freq=log_freq,
                              save_freq=save_freq)
        elif algorithm == 'ddqn_per':
            agent = DDQNAgent(episodes=episodes, 
                              prioritized=True,
                              icm=icm, 
                              tb_writer=tb_writer,
                              log_dir=log_dir,
                              save_dir=save_dir, 
                              log_freq=log_freq,
                              save_freq=save_freq)
        else:
            raise ValueError("Invalid algorithm selected")
        agent.load(args.model)
        agent.evaluate(episodes)
    

def main():
    # create save directory
    save_dir = create_dir(args.save_dir, args.algorithm, args.icm)
    log_dir = create_dir(args.log_dir, args.algorithm, args.icm)
    # initialize tensorboard
    if args.tb:
        tb_writer, tb_process, log_dir = init_tensorboard(log_dir)
    else:
        tb_writer, log_dir = None, None
    # train/evaluate
    if args.train:
        train(args.algorithm, args.episodes, args.icm, tb_writer, log_dir, save_dir, args.log_freq, args.save_freq)
    if args.evaluate:
        evaluate(args.algorithm, args.episodes, args.icm, tb_writer, log_dir, save_dir, args.log_freq, args.save_freq)
    # close tensorboard
    if args.tb:
        close_tb(tb_writer, tb_process)

if __name__ == '__main__':
    main()

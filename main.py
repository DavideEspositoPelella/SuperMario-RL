import subprocess
import webbrowser
from pathlib import Path
from datetime import datetime

from args import get_args


from agents.ddqn_per import DDQN
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import torch

from torch.utils.tensorboard import SummaryWriter


args = get_args()


def create_dir(base_dir: str,
               algorithm: str, 
               icm: bool) -> Path:
    """
    Constructs the directory path based on the algorithm and icm flag.
    
    Args:
        - base_dir (str): The base directory.
        - algorithm (str): The algorithm used for training.
        - icm (bool): Whether to use ICM.
    
    Returns:
        - directory (Path): The constructed directory path.
    """
    directory = base_dir
    if algorithm == 'ddqn_per':
        directory += "ddqn_per"
    elif algorithm == 'ddqn':
        directory += "ddqn"
    if icm:
        directory += "_icm"
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def init_tensorboard(log_dir: str):
    """
    Initialize TensorBoard.

    Args:
        - log_dir (str): The directory where to save TensorBoard logs.

    Returns:
        - tb_writer (SummaryWriter): The TensorBoard SummaryWriter object.
        - tb_process (subprocess.Popen): The TensorBoard process.
        - log_dir (Path): The directory where to save TensorBoard logs.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    current_log_dir = log_dir / timestamp
    print(f"\nTensorBoard log directory: {str(log_dir)}\n")
    tb_writer = SummaryWriter(log_dir=current_log_dir)
    tb_command = ['tensorboard', '--logdir', log_dir, '--bind_all', '--load_fast=false']
    tb_process = subprocess.Popen(tb_command)
    webbrowser.open("http://localhost:6016")
    
    return tb_writer, tb_process, current_log_dir


def close_tb(tb_writer: SummaryWriter, 
             tb_process: subprocess.Popen) -> None:
    """
    Close TensorBoard process and SummaryWriter.

    Args:
        - tb_writer (SummaryWriter): The TensorBoard SummaryWriter object.
        - tb_process (subprocess.Popen): The TensorBoard process.
    """
    tb_writer.close()
    tb_process.terminate()
    tb_process.wait()


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
        agent = DDQN(episodes=episodes,
                     prioritized=False, 
                     icm=icm, 
                     tb_writer=tb_writer,
                     log_dir=log_dir, 
                     save_dir=save_dir, 
                     log_freq=log_freq,
                     save_freq=save_freq)
        
    elif algorithm == 'ddqn_per':
        agent = DDQN(episodes=episodes, 
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
            agent = DDQN(episodes=episodes, 
                         prioritized=False,
                         icm=icm, 
                         tb_writer=tb_writer,
                         log_dir=log_dir,
                         save_dir=save_dir, 
                         log_freq=log_freq,
                         save_freq=save_freq)
        elif algorithm == 'ddqn_per':
            agent = DDQN(episodes=episodes, 
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

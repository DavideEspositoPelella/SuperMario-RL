import subprocess
import webbrowser
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


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
    print(f"\nTensorBoard log directory: {str(current_log_dir)}\n")
    tb_writer = SummaryWriter(log_dir=current_log_dir)
    tb_command = ['tensorboard', '--logdir', log_dir, '--bind_all', '--load_fast=false']
    tb_process = subprocess.Popen(tb_command)
    webbrowser.open("http://localhost:6006")
    
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
from tqdm import tqdm
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from tensordict import TensorDict
from torchrl.data import TensorDictPrioritizedReplayBuffer, LazyTensorStorage, TensorDictReplayBuffer

import gym

from models.icm import ICMModel
from models.ddqn import DDQNetwork
from config import Config

import os
os.environ["OMP_NUM_THREADS"] = "1"


class A3CNet(nn.Module):
    def __init__(self, 
                 num_channels: int=4,
                 output_dim: int=5) -> None:
        """         
        Initializes the network for the agent.

        Args:
            - num_channels (int): Number of channels in input. Default to 4.
            - output_dim (int): Output dimension. Default to 5. 
        """
        super(Net, self).__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=num_channels,
                               out_channels=32,
                               kernel_size=8, 
                               stride=4)
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=4,
                               stride=2)
        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=3,
                               stride=1)
        self.flatten = nn.Flatten()
        # linear layers, the input size depends on the output of the convolutional layers
        self.fc1 = nn.Linear(64, 256)
        self.actor = nn.Linear(256, output_dim)
        self.critic = nn.Linear(256, 1)
    
    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        """ 
        Implements the forward pass.
        
        Args:
            - x (torch.Tensor): Input tensor.
        """
        # convolutional layers
        x = F.relu(self.conv1(x)) 
        x = F.relu(self.conv2(x)) 
        x = F.relu(self.conv3(x))
        # global max pooling 
        x = self.flatten(x)
        # linear layers
        x = F.relu(self.fc1(x)) 
        logits = self.actor(x)
        values = self.critic(x)
        return logits, values
    

class A3CAgent(nn.Module):
    def __init__(self, 
                 env: gym.Env,
                 config: Config,
                 prioritized: bool=False,
                 icm: bool=False,
                 tb_writer: SummaryWriter=None, 
                 log_dir: Path=Path('./logs/'),
                 save_dir: Path=Path('./checkpoints/')) -> None:
        """
        Initializes the A3C agent.

        Args:
            - env: The environment.
            - config (Config): The configuration object. 
            - prioritized (bool): Whether to use prioritized replay buffer. Default to False.
            - icm (bool): Whether to use ICM. Default to False.
            - tb_writer (SummaryWriter): The TensorBoard SummaryWriter object. Default to None.
            - log_dir (Path): The directory where to save TensorBoard logs. Default to None.
            - save_dir (Path): The directory where to save trained models. Default to None.
        """

        super(A3CAgent, self).__init__()
                # initialize the configuration settings
        self.configure_agent(env=env, config=config, prioritized=prioritized, icm=icm)
        self.define_logs_metric(tb_writer=tb_writer, log_dir=log_dir, save_dir=save_dir)
        
    def configure_agent(self,
                        env: gym.Env,
                        config: Config,
                        prioritized: bool=False,
                        icm: bool=False) -> None:
        """
        Initializes the configuration settings.

        Args:
            - env (gym.Env): The environment.
            - config (Config): The configuration object.
            - prioritized (bool): Whether to use prioritized replay buffer. Default to False.
            - icm (bool): Whether to use ICM. Default to False.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.env = env
        self.state_dim = (4, 42, 42)
        self.num_actions=self.env.action_space.n
        self.prioritized = prioritized
        self.icm = icm
        self.define_network_components()

    def define_logs_metric(self, 
                           tb_writer: SummaryWriter=None, 
                           log_dir: Path=None,
                           save_dir: Path=None):
            """#TODO: add docstring"""
            self.tb_writer = tb_writer
            self.log_dir = log_dir
            self.save_dir = save_dir
            self.loss_log_freq = self.config.log_freq
            self.save_freq = self.config.save_freq
            self.curr_step_global = 0
            self.curr_step_local = 0

    def define_network_components(self):
        """#TODO: add docstring"""
        self.net = A3CNet(input_dim=self.state_dim, 
                          output_dim=self.num_actions).float().to(self.device)

        self.loss_fn = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr)        

        if self.icm:
            self.icm_model = ICMModel(input_dim=self.state_dim, 
                                      num_actions=self.num_actions, 
                                      feature_size=self.config.feature_size, 
                                      device=self.device).float().to(self.device)
            
            self.optimizer = torch.optim.Adam(list(self.net.parameters()) + list(self.icm_model.parameters()),lr=self.config.lr)
            self.emb_loss_fn = torch.nn.MSELoss()
            self.inverse_loss_fn = torch.nn.CrossEntropyLoss()
            self.forward_loss_fn = torch.nn.MSELoss()


    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state (LazyFrame): A single observation of the current state of dimension (state_dim)
        Outputs:
        action_idx (int): An integer representing which action the Agent will perform
        """
        if np.random.rand() < self.config.exploration_rate:
            action_idx = np.random.randint(self.num_actions)
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        self.config.exploration_rate *= self.config.exploration_rate_decay
        self.config.exploration_rate = max(self.config.exploration_rate_min, self.config.exploration_rate)

        self.curr_step_global += 1
        self.curr_step_local += 1
        return action_idx

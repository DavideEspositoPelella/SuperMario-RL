from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Tuple 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from tensordict import TensorDict
from torchrl.data import TensorDictPrioritizedReplayBuffer, LazyTensorStorage, TensorDictReplayBuffer

import gym
from gym.wrappers import LazyFrames

from models.icm import ICMModel
from models.ddqn import DDQNetwork
from config import Config



class DDQNAgent(nn.Module):
    def __init__(self, 
                 env: gym.Env,
                 config: Config,
                 prioritized: bool=False,
                 icm: bool=False,
                 tb_writer: SummaryWriter=None, 
                 log_dir: Path=Path('./logs/'),
                 save_dir: Path=Path('./checkpoints/')) -> None:
        """
        Initializes the DDQN agent.

        Args:
            - env: The environment.
            - config (Config): The configuration object. 
            - prioritized (bool): Whether to use prioritized replay buffer. Default to False.
            - icm (bool): Whether to use ICM. Default to False.
            - tb_writer (SummaryWriter): The TensorBoard SummaryWriter object. Default to None.
            - log_dir (Path): The directory where to save TensorBoard logs. Default to None.
            - save_dir (Path): The directory where to save trained models. Default to None.
        """

        super(DDQNAgent, self).__init__()
        # configure the agent with the given parameters
        self.configure_agent(env=env, config=config, prioritized=prioritized, icm=icm)
        # define tensorboard logging and checkpoint saving
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
        self.state_dim = (self.config.stack, self.config.resize_shape, self.config.resize_shape)
        self.num_actions = self.env.action_space.n
        self.prioritized = prioritized
        self.icm = icm
        self.define_network_components()
    
    def define_network_components(self) -> None:
        """Initialize the networks, the replay buffer, the loss and the optimizer according to the configuration settings."""
        # ddqn network
        self.net = DDQNetwork(input_dim=self.state_dim, 
                              output_dim=self.num_actions).float().to(self.device)
        # replay buffer
        if self.prioritized:
            self.buffer = TensorDictPrioritizedReplayBuffer(storage=LazyTensorStorage(max_size = self.config.memory_size),
                                                            alpha=self.config.alpha, 
                                                            beta=self.config.beta,
                                                            priority_key='td_error',
                                                            eps=self.config.epsilon_buffer,
                                                            batch_size=self.config.batch_size)
            # a customized loss function is computed directly in training for ddqn_per
        else:
            self.buffer = TensorDictReplayBuffer(storage=LazyTensorStorage(max_size = self.config.memory_size))
            # loss function for ddqn
            self.loss_fn = torch.nn.SmoothL1Loss()
        # the optimizer for ddqn and ddqn_per
        self.optimizer = torch.optim.Adam(self.net.online_net.parameters(), lr=self.config.lr)        
        # curiosity module
        if self.icm:
            self.icm_model = ICMModel(input_dim=self.state_dim, 
                                    num_actions=self.num_actions, 
                                    feature_size=self.config.feature_size, 
                                    device=self.device).float().to(self.device)
            # the optimizer for ddqn with curiosity
            self.optimizer = torch.optim.Adam(list(self.net.online_net.parameters()) + list(self.icm_model.parameters()),lr=self.config.lr)
            # losses to integrate curiosity
            self.emb_loss_fn = torch.nn.MSELoss()
            self.inverse_loss_fn = torch.nn.CrossEntropyLoss()
            self.forward_loss_fn = torch.nn.MSELoss()

    def define_logs_metric(self, 
                           tb_writer: SummaryWriter=None, 
                           log_dir: Path=None,
                           save_dir: Path=None) -> None:
        """ 
        Initializes the tensorboard logging, checkpoint saving and the current step.
        
        Args:
            - tb_writer (SummaryWriter): The TensorBoard SummaryWriter object.
            - log_dir (Path): The directory for TensorBoard logs. 
            - save_dir (Path): The directory for model checkpoints.
        """
        self.tb_writer = tb_writer
        self.log_dir = log_dir
        self.save_dir = save_dir
        self.loss_log_freq = self.config.log_freq
        self.save_freq = self.config.save_freq
        self.curr_step_global = 0
        # defined just in case is needed for logs but not actually used
        self.curr_step_local = 0


    def act(self, 
            state: LazyFrames) -> int:
        """
        Choose an action from the state with an epsilon-greedy policy.

        Args:
            - state (LazyFrames): The current state of the environment.

        Returns:
            - action_idx (int): The index of the selected action.
        """
        if np.random.rand() < self.config.exploration_rate:
            action_idx = np.random.randint(self.num_actions)
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            # transform the state to a tensor and add a batch dimension 
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            # get the action values from the network and choose the best action
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()            
        self.curr_step_global += 1
        self.curr_step_local += 1
        return action_idx
    
        
    def store(self, 
              state: LazyFrames, 
              next_state: LazyFrames, 
              action: int, 
              reward: float, 
              done: bool) -> None:
        """
        Store the experience in the replay buffer.

        Args:
            - state (LazyFrames): The current state of the environment.
            - next_state (LazyFrames): The next state of the environment.
            - action (int): The index of the selected action.
            - reward (float): The reward received from the environment.
            - done (bool): Whether the episode is finished.
        """
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
            
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()
        # convert the state and next_state to tensors
        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])
        # use the TensorDict class to store the experience
        data = TensorDict({"state": state, 
                           "next_state": next_state, 
                           "action": action, 
                           "reward": reward, 
                           "done": done}, batch_size=[])
        self.buffer.add(data)

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Retrieve a batch of experiences from replay buffer.

        Returns:
            - state (torch.Tensor): The current state of the environment.
            - next_state (torch.Tensor): The next state of the environment.
            - action (torch.Tensor): The index of the selected action.
            - reward (torch.Tensor): The reward received from the environment.
            - done (torch.Tensor): Whether the episode is finished.
            - info (dict): The information about the batch.
        """
        batch, info = self.buffer.sample(self.config.batch_size, return_info=True)
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        return state.to(self.device), next_state.to(self.device), action.to(self.device).squeeze(), reward.to(self.device).squeeze(), done.to(self.device).squeeze(), info

    def learn(self):
        """
        Update the network parameters using experiences.

        Returns: 
            - td_error (torch.Tensor): The TD error.
            - loss (float): The loss value.
        """
        # sample a batch of experiences from the replay buffer
        state, next_state, action, reward, done, info = self.sample()
        one_hot_action = F.one_hot(action, num_classes=self.num_actions).float()
        total_reward = reward

        if self.icm:
            # get the next state feature representation, the predicted one and the predicted action
            next_state_feat, pred_next_state_feat, pred_action = self.icm_model(state, next_state, one_hot_action)
            # compute the losses for the forward and inverse models
            inverse_loss = self.inverse_loss_fn(pred_action, action)
            forward_loss = self.forward_loss_fn(pred_next_state_feat, next_state_feat)
            # compute the intrinsic reward, logged in tensorboard
            intrinsic_reward = self.config.eta * F.mse_loss(next_state_feat, pred_next_state_feat, reduction='none').mean(-1)
            total_reward = reward + intrinsic_reward
        
        # get the Q values estimates
        q_est = self.net(state, model="online")[np.arange(0, self.config.batch_size), action]    
        with torch.no_grad():
            # get the Q values for the next state and select the best action
            next_state_Q = self.net(next_state, model="online")
            best_action = torch.argmax(next_state_Q, axis=1)
            # get the Q values from the target network and compute the target
            next_Q = self.net(next_state, model="target")[np.arange(0, self.config.batch_size), best_action]    
            q_tgt = (reward + (1 - done.float()) * self.config.gamma * next_Q).float()
        # compute the td error
        td_error = q_est - q_tgt 
        if self.prioritized:
            # compute the priority, the loss with importance sampling and update the priority in the buffer
            priority = td_error.abs() + self.config.epsilon_buffer
            importance_sampling_weights = torch.FloatTensor(info["_weight"]).reshape(-1, 1).to(self.device)
            loss = (importance_sampling_weights * td_error.pow(2)).mean()
            self.buffer.update_priority(info["index"], priority)
        else:
            # compute the loss
            loss = self.loss_fn(q_est, q_tgt)          
        if self.icm:
            # total loss for curiosity is a combination of ddqn loss and inverse/forward losses, weighted by beta and lambda
            loss = self.config.lambda_icm * loss + (1 - self.config.beta) * inverse_loss + self.config.beta * forward_loss
        
        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # log metrics in tensorboard
        if self.tb_writer and self.curr_step_global % self.loss_log_freq == 0:
            self.tb_writer.add_scalar("Loss/train", loss.item(), self.curr_step_global)
            self.tb_writer.add_scalar("Total_reward/train", total_reward.mean(), self.curr_step_global)
            self.tb_writer.add_scalar("Td_error/train", td_error.mean(), self.curr_step_global)                
            if self.icm:
                self.tb_writer.add_scalar("Forward_loss/train", forward_loss.item(), self.curr_step_global)
                self.tb_writer.add_scalar("Inverse_loss/train", inverse_loss.item(), self.curr_step_global)
                self.tb_writer.add_scalar("Intrinsic_reward/train", intrinsic_reward.mean(), self.curr_step_global)
        return td_error, loss.item()
        
    def train(self) -> None:
        """Train the agent for the given number of episodes."""
        
        episodes = self.config.episodes
        # Burn-in
        print("\nPopulating the replay buffer")
        reset_output = self.env.reset()
        state, _ = reset_output
        while self.curr_step_global < self.config.burn_in:
            # get the action and execute it
            action = self.act(state)
            next_state, reward, done, _, info = self.env.step(action) 
            # store the transition
            self.store(state, next_state, action, reward, done)
            # reset the environment
            if done:
                state, _  = self.env.reset()
        print("Burn in complete!\n")

        print("Starting...")
        print(f"Training for: {episodes} episodes\n")
        self.curr_step_global = 0
        for e in tqdm(range(episodes)):
            self.curr_step_local = 0
            state, _ = self.env.reset()
            self.ep = e
            while True:
                # select and perform action
                action = self.act(state)
                next_state, extrinsic_reward, done, _, info = self.env.step(action)
                # store the transition
                self.store(state, next_state, action, extrinsic_reward, done)
                # learn every update_freq time steps
                if self.curr_step_global % self.config.update_freq == 0:
                    self.learn()
                # sync target model every sync_freq time steps
                if self.curr_step_global % self.config.sync_freq == 0:
                    self.net.target_net.load_state_dict(self.net.online_net.state_dict())
                state = next_state
                # break if episode is finished
                if done or info["flag_get"]:
                    break
            self.config.exploration_rate *= self.config.exploration_rate_decay
            self.config.exploration_rate = max(self.config.exploration_rate_min, self.config.exploration_rate)
            # save model every save_freq episodes
            if self.ep % self.save_freq == 0:
                self.save()
        print("Training complete.\n")
        return 
    

    def save(self):
        """Save the model to a checkpoint."""
        save_path = self.save_dir / f"mario_net_{self.ep}.chkpt"

        state = {
            'model_state_dict': self.net.state_dict(),
            'icm_model_state_dict': self.icm_model.state_dict() if self.icm else None,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode': self.ep,
            'batch_size': self.config.batch_size,
            'update_freq': self.config.update_freq,
            'sync_freq': self.config.sync_freq,
            'exploration_rate': self.config.exploration_rate,
            'exploration_decay': self.config.exploration_rate_decay,
            'memory_size': self.config.memory_size,
            'burn_in': self.config.burn_in,
            'alpha': self.config.alpha if self.prioritized else None,
            'beta': self.config.beta if self.prioritized else None,
            'gamma': self.config.gamma,
            'curr_step_global': self.curr_step_global,
            'curr_step_local': self.curr_step_local,
            'log_dir': self.log_dir,
        }
        torch.save(state, save_path)


    def load(self, model_path: str=None):
        """Load the agent's state from a checkpoint."""
        if model_path == 'False':
            print("No model specified")
            raise ValueError(f"{model_path} does not exist")

        load_path = self.save_dir / model_path
        
        if not load_path.exists():
                raise ValueError(f"{load_path} does not exist")
        
        else:
            state = torch.load(load_path, map_location=self.device)
            
            self.net.load_state_dict(state['model_state_dict'])
            if self.icm and 'icm_model_state_dict' in state:
                self.icm_model.load_state_dict(state['icm_model_state_dict'])
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
            self.ep = state['episode']
            self.config.batch_size = state['batch_size']
            self.config.update_freq = state['update_freq']
            self.config.sync_freq = state['sync_freq']
            self.config.exploration_rate = state['exploration_rate']
            self.config.exploration_rate_decay = state['exploration_decay']
            self.config.memory_size = state['memory_size']
            self.config.burn_in = state['burn_in']
            if self.prioritized:
                self.config.alpha = state['alpha']
                self.config.beta = state['beta']
            self.config.gamma = state['gamma']
            self.curr_step_global = state.get('curr_step_global', 0)
            self.curr_step_local = state.get('curr_step_local', 0)
            self.log_dir = state['log_dir']

            # Print loaded hyperparameters
            print("Loaded Hyperparameters:")
            hyperparameters = [
                'batch_size', 'update_freq', 'sync_freq', 'exploration_rate', 
                'exploration_rate_decay', 'memory_size', 'burn_in', 'gamma', 'log_dir'
            ]
            if self.prioritized:
                hyperparameters.extend(['alpha', 'beta'])

            for key in hyperparameters:
                if key in state:
                    print(f"{key}: {state[key]}")
                else:
                    print(f"{key}: Not found in saved state")

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret

    def evaluate(self, 
                 env: gym.Env) -> None:
        """
        Evaluate the agent for 10 episodes.
        """
        rewards = []
        print(f'\nEvaluating for 10 episodes')
        print('Algorithm: {}'.format('DDQN_PER' if self.prioritized else 'DDQN'))
        for _ in tqdm(range(10)):
            total_reward = 0
            done = False
            state, _ = env.reset()
            while not done:
                action = self.act(state)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                if terminated or truncated:
                    break
            rewards.append(total_reward)
        print('Mean Reward:', np.mean(rewards))
        print()


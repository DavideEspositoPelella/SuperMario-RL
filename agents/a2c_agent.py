from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Tuple
from util.util import create_dir, init_tensorboard, close_tb, set_seed

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
import gym
from gym.wrappers import LazyFrames

from models.icm import ICMModel
from models.a2c import A2C
from config import Config

import torch.optim as optim
from torch.distributions import Categorical

import copy

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        return self.state
    



class A2CAgent(nn.Module):
    def __init__(self, 
                 env: gym.Env,
                 config: Config,
                 icm: bool=False,
                 tb_writer: SummaryWriter=None, 
                 log_dir: Path=Path('./logs/'),
                 save_dir: Path=Path('./checkpoints/')) -> None:
        """
        Initializes the A2C agent.

        Args:
            - env: The environment.
            - config (Config): The configuration object. 
            - icm (bool): Whether to use ICM. Default to False.
            - tb_writer (SummaryWriter): The TensorBoard SummaryWriter object. Default to None.
            - log_dir (Path): The directory where to save TensorBoard logs. Default to None.
            - save_dir (Path): The directory where to save trained models. Default to None.
        """

        super(A2CAgent, self).__init__()
        # configure the agent with the given parameters
        self.configure_agent(env=env, config=config, icm=icm)
        # define tensorboard logging and checkpoint saving
        self.define_logs_metric(tb_writer=tb_writer, log_dir=log_dir, save_dir=save_dir)

    def configure_agent(self,
                        env: gym.Env,
                        config: Config,
                        icm: bool=False) -> None:
        """
        Initializes the configuration settings.

        Args:
            - env (gym.Env): The environment.
            - config (Config): The configuration object.
            - icm (bool): Whether to use ICM. Default to False.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.env = env
        self.state_dim = (self.config.stack, self.config.resize_shape, self.config.resize_shape)
        self.num_actions=self.env.action_space.n
        self.icm = icm
        self.define_network_components()

    def define_network_components(self) -> None:
        """Initialize the networks, the loss and the optimizer according to the configuration settings."""
        # define the actor-critic network
        self.a2c = A2C(input_dim=self.state_dim, 
                       num_actions=self.num_actions).float().to(self.device)
        
        if self.config.adaptive:
            self.a2c_noised = A2C(input_dim=self.state_dim, 
                                   num_actions=self.num_actions).float().to(self.device)
            self.a2c_noised.load_state_dict(self.a2c.state_dict().copy())
            self.actor_noised = self.a2c_noised.actor_net
            self.scalar = self.config.scalar
            self.distances = []
            self.desired_distance = self.config.desired_distance
            self.scalar_decay = self.config.scalar_decay

   
        if self.config.ou_noise:
            self.ou_noise = OUNoise(size=self.num_actions)

        # define the optimizers for the actor and critic networks
        self.actor_optim = optim.RMSprop(self.a2c.actor_net.parameters(), lr=self.config.actor_lr)
        self.critic_optim = optim.RMSprop(self.a2c.critic_net.parameters(), lr=self.config.critic_lr)
        # define the ICM model, the optimizer and the loss functions
        if self.icm:
            self.icm_model = ICMModel(input_dim=self.state_dim, 
                                      num_actions=self.num_actions, 
                                      feature_size=self.config.feature_size, 
                                      device=self.device).float().to(self.device)
            self.optimizer = torch.optim.RMSprop(list(self.icm_model.parameters()) + list(self.a2c.parameters()), lr=self.config.lr)
            self.emb_loss_fn = torch.nn.MSELoss()
            self.inverse_loss_fn = torch.nn.CrossEntropyLoss()
            self.forward_loss_fn = torch.nn.MSELoss()

    def define_logs_metric(self, 
                           tb_writer: SummaryWriter=None, 
                           log_dir: Path=None,
                           save_dir: Path=None) -> None:
        """ 
        Initializes the tensorboard logging and checkpoint saving.
        
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
        self.episodes = 0
    
    def lazy_to_tensor(self, 
                       state: LazyFrames) -> torch.Tensor:
        """
        Convert the state from LazyFrames to a tensor.
        
        Args:
            - state (LazyFrames): The state to convert.
        
        Returns:
            - state (torch.Tensor): The converted state.
        """
        state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
        state = torch.tensor(state, device=self.device).unsqueeze(0)
        return state
    
    def act(self, 
            state: LazyFrames) -> int:
        """
        Select an action according to the current policy and add noise if specified.
        
        Args:
            - state (LazyFrames): The current state.
    
        Returns:
            - action (int): The index of the selected action.
        """
        state = self.lazy_to_tensor(state)
        # noise in weights
        if self.config.adaptive:
            with torch.no_grad():
                # get the action values from the noised actor for comparison
                '''action = self.actor_regular(state).cpu().data.numpy()'''
                self.actor_noised.load_state_dict(self.a2c.actor_net.state_dict().copy())
                # add noise to the copy
                self.actor_noised.add_weight_noise(self.scalar)
                # get noised action
                action_probs_noised = self.actor_noised(state)
                dist_noised = Categorical(logits=action_probs_noised)
                action_noised = dist_noised.sample().item()

                action_probs = self.a2c.actor_net(state)
                dist = Categorical(logits=action_probs)
                action = dist.sample().item()
                                
                # meassure the distance between the actions for noise update
                distance = np.sqrt(np.mean(np.square(action-action_noised)))
                
                # adjust the amount of noise given to the actor_noised
                if distance > self.desired_distance:
                    self.scalar *= self.scalar_decay
                if distance < self.desired_distance:
                    self.scalar /= self.scalar_decay
                
                # set the noised action as action
                action = action_noised

        # noise on action
        elif self.config.ou_noise:
            action_probs = self.a2c.actor_net(state)
            if self.config.ou_noise:
                noise = torch.tensor(self.ou_noise.sample(), device=self.device)
                action_probs += noise
                #action = np.clip(action + noise, 0, self.num_actions - 1)
            dist = Categorical(logits=action_probs)
            action = dist.sample().item()

        # no noise
        else:
            action_probs = self.a2c.actor_net(state)
            dist = Categorical(logits=action_probs)
            action = dist.sample().item()

        return action
    
    def returns_advantages(self, 
                           rewards: np.ndarray, 
                           dones: np.ndarray, 
                           values: np.ndarray, 
                           next_value) -> Tuple[np.ndarray, np.ndarray]:  
        """
        Compute the returns and advantages for the given rewards, values and next value.

        Args:
            - rewards (np.ndarray): The rewards for the current episode.
            - dones (np.ndarray): Whether the states are done.
            - values (np.ndarray): The states values.
            - next_value: The next states values.

        Returns:
            - returns (np.ndarray): The returns for the current episode.
            - advantages (np.ndarray): The advantages for the current episode.
        """      


        gae = 0
        gae_lambda = 0.95
        advantages = np.zeros_like(rewards)
        for t in reversed(range(len(rewards))):
            # Calculate delta: reward + discount_factor * next_value * (1 - done) - value
            delta = rewards[t] + self.config.gamma * next_value * (1 - int(dones[t])) - values[t]
            # Update gae: delta + discount_factor * lambda * gae * (1 - done)
            gae = delta + self.config.gamma * gae_lambda * gae * (1 - int(dones[t]))
            # Store advantage
            advantages[t] = gae
            # Update next_value to the value of the current state
            next_value = values[t]

        # The returns are the value estimates plus the advantages
        returns = advantages + values
        return returns, advantages
    
    def optimize_model(self, 
                       states: np.ndarray, 
                       actions: np.ndarray,
                       returns: np.ndarray,
                       advantages: np.ndarray, 
                       rewards: np.ndarray,
                       total_reward: float) -> None:
        """
        Compute the losses and optimize the model.
        
        Args:
            - states (np.ndarray): The states for the current episode.
            - actions (np.ndarray): The actions for the current episode.
            - returns (np.ndarray): The returns for the current episode.
            - advantages (np.ndarray): The advantages for the current episode.
            - rewards (np.ndarray): The rewards for the current episode.
        """
        # convert the numpy arrays to tensors and move to the device
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)
        returns_tensor = torch.tensor(returns[:, None], dtype=torch.float).to(self.device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float).to(self.device)
        states_tensor = torch.tensor(states, dtype=torch.float).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float).to(self.device)
        # compute the action probabilities and the state values
        action_probs, state_values = self.a2c(states_tensor)
        dist = Categorical(logits=action_probs)
        # compute the log probabilities of the actions
        log_probs = dist.log_prob(actions_tensor)
        # actor loss
        actor_loss = -(log_probs * advantages_tensor).mean()
        # critic loss
        critic_loss = F.mse_loss(state_values, returns_tensor)
        # entropy loss
        entropy_loss = dist.entropy().mean()
        # a2c loss
        loss = actor_loss + 0.5 * critic_loss - self.config.ent_coef * entropy_loss
        # icm loss and intrinsic reward
        if self.icm:
            init_states = states_tensor[:-1]
            next_states = states_tensor[1:]
            one_hot_action = F.one_hot(actions_tensor[:-1], num_classes=self.num_actions).float().to(self.device)
            # get the next state feature representation, the predicted one and the predicted action
            next_state_feat, pred_next_state_feat, pred_action = self.icm_model(init_states, next_states, one_hot_action)
            # compute the losses for the forward and inverse models
            inverse_loss = self.inverse_loss_fn(pred_action, actions_tensor[:-1])
            forward_loss = self.forward_loss_fn(pred_next_state_feat, next_state_feat)
            # compute the intrinsic reward, logged in tensorboard
            intrinsic_reward = self.config.eta * F.mse_loss(next_state_feat, pred_next_state_feat, reduction='none').mean(-1)
            rewards = rewards_tensor[:-1] + intrinsic_reward
            # total loss for curiosity is a combination of ddqn loss and inverse/forward losses, weighted by beta and lambda
            loss = self.config.lambda_icm * loss + (1 - self.config.beta) * inverse_loss + self.config.beta * forward_loss
            self.optimizer.zero_grad()
        # update the actor and critic networks
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        loss.backward()               
        torch.nn.utils.clip_grad_norm_(self.a2c.parameters(), 0.5)
        if self.icm:
            torch.nn.utils.clip_grad_norm_(self.icm_model.parameters(), 0.5)
        # update the icm model
        if self.icm:
            self.optimizer.step()
        else:
            self.actor_optim.step()
            self.critic_optim.step()
        # log the metrics
        if self.tb_writer:
            self.tb_writer.add_scalar("Actor_loss/train", actor_loss.item(), self.step)
            self.tb_writer.add_scalar("Critic_loss/train", critic_loss.item(), self.step)
            self.tb_writer.add_scalar("Entropy_loss/train", entropy_loss.item(), self.step)
            self.tb_writer.add_scalar("Total_loss/train", loss.item(), self.step)
            self.tb_writer.add_scalar("Advantage/train", advantages.mean().item(), self.step)
            self.tb_writer.add_scalar("Rewards/train", rewards.mean(), self.step)
            if self.icm:
                self.tb_writer.add_scalar("Forward_loss/train", forward_loss.item(), self.step)
                self.tb_writer.add_scalar("Inverse_loss/train", inverse_loss.item(), self.step)
                self.tb_writer.add_scalar("Intrinsic_reward/train", intrinsic_reward.mean(), self.step)

    def flush(self) -> None:
        """Flush the tensorboard writer."""
        if self.tb_writer:
            if self.icm:
                self.tb_writer.add_scalar("learning/train", 0, self.episodes)
            else:
                self.tb_writer.add_scalar("actor_lr/train", 0, self.episodes)
                self.tb_writer.add_scalar("critic_lr/train", 0, self.episodes)
                self.tb_writer.add_scalar("Total_Reward/train", 0, self.episodes)
                        
            
    def anneal_learning_rate(self, initial_lr, episode, total_episodes, min_lr=1e-4):
        """
        Anneal the learning rate linearly from the initial learning rate to the minimum learning rate.

        Args:
            initial_lr (float): The initial learning rate.
            episode (int): The current episode number.
            total_episodes (int): The total number of episodes for training.
            min_lr (float): The minimum learning rate to anneal to. Default is 1e-4.

        Returns:
            lr (float): The annealed learning rate.
        """
        decay = (initial_lr - min_lr) / total_episodes
        new_lr = initial_lr - decay * episode
        return max(new_lr, min_lr)

    def train(self) -> None:
        """Train the agent for the given number of episodes."""
        print("Start training.")
        self.step = 0
        anealing = True

        self.flush()

        for episode in tqdm(range(self.config.episodes)):
            self.episodes = episode

            if anealing:
                if self.icm:
                    # Anneal learning rate for curiosity
                    new_lr = self.anneal_learning_rate(self.config.lr, episode, self.config.episodes)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                else:
                    # Anneal learning rate for actor and critic
                    new_actor_lr = self.anneal_learning_rate(self.config.actor_lr, episode, self.config.episodes)
                    new_critic_lr = self.anneal_learning_rate(self.config.critic_lr, episode, self.config.episodes)
                    
                    # Update the learning rate for the optimizers
                    for param_group in self.actor_optim.param_groups:
                        param_group['lr'] = new_actor_lr
                    for param_group in self.critic_optim.param_groups:
                        param_group['lr'] = new_critic_lr

            if self.config.ou_noise:
                self.ou_noise.reset()

            state, _ = self.env.reset()
            total_reward = 0

            while True:
                # initialize the environment and all the variables for the current episode
                actions = np.empty((self.config.n_steps,), dtype=int)
                dones = np.empty((self.config.n_steps,), dtype=bool)
                rewards = np.empty((self.config.n_steps), dtype=float)
                values = np.empty((self.config.n_steps), dtype=float)
                states = np.empty((self.config.n_steps, 
                                   self.config.stack, 
                                   self.config.resize_shape, 
                                   self.config.resize_shape), dtype=float)
                # rollout phase
                done_idx = self.config.n_steps - 1
                for i in range(self.config.n_steps):
                    # store the current state, value and action
                    states[i] = state
                    state_tensor = self.lazy_to_tensor(state)
                    values[i] = self.a2c.critic_net(state_tensor).cpu().detach().numpy()
                    actions[i] = self.act(state)
                    # perform the action and store the reward
                    state, reward, terminated, truncated, info  = self.env.step(actions[i])
                    rewards[i] = reward
                    dones[i] = terminated or truncated
                    self.step += 1
                    total_reward += reward

                    # check if the agent reached the flag and save the model
                    if info['flag_get']:
                        print("\n################################################ \n \
                              Win! Flag reached at episode: ", self.episodes, "Total reward: ",\
                              total_reward,"\n################################################\n")
                        
                        if self.ou_noise:
                            self.save(f'win_OU_{self.episodes}')
                        elif self.config.adaptive:
                            self.save(f'win_adaptive_{self.episodes}')
                        else:
                            self.save(f'win_{self.episodes}')
                    
                    if dones[i]:
                        done_idx = i
                        break

                # handle termination before n_steps      
                if dones[done_idx]: 
                    next_value = np.zeros(1)
                    states = states[:done_idx+1]
                    rewards = rewards[:done_idx+1]
                    dones = dones[:done_idx+1]
                    values = values[:done_idx+1]
                    actions = actions[:done_idx+1]
                else:
                    state_tensor = self.lazy_to_tensor(state)
                    next_value = self.a2c.critic_net(state_tensor).cpu().detach().numpy()[0]
                # compute the returns and advantages
                returns, advantages = self.returns_advantages(rewards, dones, values, next_value)
                # optimize the model
                self.optimize_model(states, actions, returns, advantages, rewards, total_reward)
                # save the model
                if dones[-1]:
                    if self.tb_writer:
                        if self.icm:
                            self.tb_writer.add_scalar("learning/train", new_lr, self.episodes)
                        else:
                            self.tb_writer.add_scalar("actor_lr/train", new_actor_lr, self.episodes)
                            self.tb_writer.add_scalar("critic_lr/train", new_critic_lr, self.episodes)
                        self.tb_writer.add_scalar("Total_Reward/train", total_reward, self.episodes)
                        
                    break
            if self.episodes % self.save_freq == 0:
                self.save()
    
    def save(self,
             name: str=None) -> None:
        """Save the model and the configuration settings."""
        if name:
            save_path = self.save_dir / f"{name}.chkpt"
        else:
            save_path = self.save_dir / f"mario_net_{self.episodes}.chkpt"
        state = {
            'a2c_model_state_dict': self.a2c.state_dict(),
            'icm_model_state_dict': self.icm_model.state_dict() if self.icm else None,
            'optimizer_state_dict': self.optimizer.state_dict() if self.icm else None,
            'actor_optimizer_state_dict': self.actor_optim.state_dict(),
            'critic_optimizer_state_dict': self.critic_optim.state_dict(),
            'episode': self.episodes,
            'gamma': self.config.gamma,
            'actor_lr': self.config.actor_lr,
            'critic_lr': self.config.critic_lr,
            'eta': self.config.eta if self.icm else None,
            'beta_icm': self.config.beta_icm if self.icm else None,
            'lambda_icm': self.config.lambda_icm if self.icm else None,
            'n_steps': self.config.n_steps,
            'ent_coef': self.config.ent_coef,
            'skip_frame': self.config.skip_frame,
            'stack': self.config.stack,
            'resize_shape': self.config.resize_shape,
            'log_dir': self.log_dir}
        torch.save(state, save_path)
        
    def load(self, 
             model_path: str) -> None:
        """
        Load the agent's state from a checkpoint.
        
        Args:
            - model_path (str): The path to the checkpoint.
        """
        if model_path == 'False':
            print("No model specified")
            raise ValueError(f"{model_path} does not exist")
        load_path = self.save_dir / model_path
        if not load_path.exists():
                raise ValueError(f"{load_path} does not exist")
        else:
            state = torch.load(load_path, map_location=self.device)
            
            self.a2c.load_state_dict(state['a2c_model_state_dict'])
            if self.icm and 'icm_model_state_dict' in state:
                self.icm_model.load_state_dict(state['icm_model_state_dict'])
                self.optimizer.load_state_dict(state['optimizer_state_dict'])
            self.actor_optim.load_state_dict(state['actor_optimizer_state_dict'])
            self.critic_optim.load_state_dict(state['critic_optimizer_state_dict'])
            self.episodes = state['episode']
            self.config.actor_lr = state['actor_lr']
            self.config.critic_lr = state['critic_lr']
            self.config.eta = state['eta'] if self.icm else None
            self.config.beta_icm = state['beta_icm'] if self.icm else None
            self.config.lambda_icm = state['lambda_icm'] if self.icm else None
            self.config.n_steps = state['n_steps']
            self.config.ent_coef = state['ent_coef']
            self.config.skip_frame = state['skip_frame']
            self.config.stack = state['stack']
            self.config.resize_shape = state['resize_shape']
            self.log_dir = state['log_dir']

    def evaluate(self, 
                 env: gym.Env) -> None:
        """
        Evaluate the agent for 10 episodes.

        Args:
            - env (gym.Env): The environment.
        """
        winners = []
        best_reward = 0
        for seed in tqdm(range(3000)): 
            set_seed(183)
            rewards = []

            #print(f'\nEvaluating for 10 episodes')
            #print('Algorithm: A2C')
            for _ in range(1):
                total_reward = 0
                done = False
                state, _ = env.reset()
                while not done:
                    action = self.act(state)
                    state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    total_reward += reward

                rewards.append(total_reward)
            if info['flag_get']:
                print("\n################################################ \nWin! Flag reached at seed: ", seed, "\n################################################\n")
                winners.append(seed)
            if np.mean(rewards) > best_reward:
                best_reward = np.mean(rewards)
                print('New best Mean Reward:', np.mean(rewards), 'Seed:', seed)
            #print('Mean Reward:', np.mean(rewards))
            #print()
        print('Winners:', winners)
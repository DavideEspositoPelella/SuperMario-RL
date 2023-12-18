from tqdm import tqdm
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
import gym

from models.icm import ICMModel
from models.a2c import A2C
from config import Config

import torch.optim as optim
from torch.distributions import Categorical

class A2CAgent(nn.Module):
    def __init__(self, 
                 env: gym.Env,
                 config: Config,
                 prioritized: bool=False,
                 icm: bool=False,
                 tb_writer: SummaryWriter=None, 
                 log_dir: Path=Path('./logs/'),
                 save_dir: Path=Path('./checkpoints/')) -> None:
        """
        Initializes the A2C agent.

        Args:
            - env: The environment.
            - config (Config): The configuration object. 
            - prioritized (bool): Whether to use prioritized replay buffer. Default to False.
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
        self.state_dim = (4, 42, 42)
        self.num_actions=self.env.action_space.n
        self.icm = icm
        self.define_network_components()

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
        
    def define_network_components(self) -> None:
        """Initialize the networks, the loss and the optimizer according to the configuration settings."""
        self.a2c = A2C(input_dim=self.state_dim, 
                       num_actions=self.num_actions).float().to(self.device)
            
        self.actor_optim = optim.RMSprop(self.a2c.actor_net.parameters(), lr=self.config.actor_lr)
        self.critic_optim = optim.RMSprop(self.a2c.critic_net.parameters(), lr=self.config.critic_lr)
    
        if self.icm:
            self.icm_model = ICMModel(input_dim=self.state_dim, 
                                      num_actions=self.num_actions, 
                                      feature_size=self.config.feature_size, 
                                      device=self.device).float().to(self.device)
            
            self.optimizer = torch.optim.Adam(self.icm_model.parameters(), lr=self.config.lr)
            self.emb_loss_fn = torch.nn.MSELoss()
            self.inverse_loss_fn = torch.nn.CrossEntropyLoss()
            self.forward_loss_fn = torch.nn.MSELoss()
        
    def act(self, state):
        action_probs = self.a2c.actor_net(state)
        dist = Categorical(logits=action_probs)
        action = dist.sample().item()
        return action
    
    def lazy_to_tensor(self, state):
        state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
        state = torch.tensor(state, device=self.device).unsqueeze(0)
        return state
    
    def train(self):

        print("Start training.")
        
        for e in tqdm(range(self.config.episodes)):
            self.ep = e
            actions = np.empty((self.config.n_steps,), dtype=np.int)
            dones = np.empty((self.config.n_steps,), dtype=np.bool)
            rewards, values = np.empty((2, self.config.n_steps), dtype=np.float)
            states = np.empty((self.config.n_steps, 4, 42, 42), dtype=np.float)
            state, _ = self.env.reset()
            
            # Rollout phase
            for i in range(self.config.n_steps):
                states[i] = state
                state_tensor = self.lazy_to_tensor(state)
                values[i] = self.a2c.critic_net(state_tensor).cpu().detach().numpy()
                actions[i] = self.act(state_tensor)
                state, reward, terminated, truncated, _  = self.env.step(actions[i])
                rewards[i] = reward
                dones[i] = terminated or truncated
                if dones[i]:
                    state, _ = self.env.reset()
            
            if dones[-1]:
                next_value = 0
            else:
                state_tensor = self.lazy_to_tensor(state)
                next_value = self.a2c.critic_net(state_tensor).cpu().detach().numpy()[0]
            
            returns, advantages = self.returns_advantages(rewards, dones, values, next_value)
            self.optimize_model(states, actions, returns, advantages)
        return
    
    def returns_advantages(self, rewards, dones, values, next_value):        
        returns = np.append(np.zeros_like(rewards), next_value, axis=0)
        
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.config.gamma * returns[t + 1] * (1 - dones[t])
            
        returns = returns[:-1]
        advantages = returns - values
        return returns, advantages
    
    def optimize_model(self, 
                       states, 
                       actions,
                       returns,
                       advantages):
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)
        returns = torch.tensor(returns[:, None], dtype=torch.float).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float).to(self.device)
        states = torch.tensor(states, dtype=torch.float).to(self.device)

        action_probs, state_values = self.a2c(states)
        dist = Categorical(logits=action_probs)
        log_probs = dist.log_prob(actions_tensor)

        # actor loss
        actor_loss = -(log_probs * advantages).mean()
        # critic loss
        critic_loss = F.mse_loss(state_values, returns)
        # entropy loss
        entropy_loss = dist.entropy().mean()
        # a2c loss
        loss = actor_loss + 0.5 * critic_loss - self.config.ent_coef * entropy_loss
        
        if self.icm:
            init_states = states[:-1]
            next_states = states[1:]
            one_hot_action = F.one_hot(actions_tensor[:-1], num_classes=self.num_actions).float()
            # get the next state feature representation, the predicted one and the predicted action
            next_state_feat, pred_next_state_feat, pred_action = self.icm_model(init_states, next_states, one_hot_action)
            # compute the losses for the forward and inverse models
            inverse_loss = self.inverse_loss_fn(pred_action, actions_tensor[:-1])
            forward_loss = self.forward_loss_fn(pred_next_state_feat, next_state_feat)
            # compute the intrinsic reward, logged in tensorboard
            intrinsic_reward = self.config.eta * F.mse_loss(next_state_feat, pred_next_state_feat, reduction='none').mean(-1)
            # total loss for curiosity is a combination of ddqn loss and inverse/forward losses, weighted by beta and lambda
            loss = self.config.lambda_icm * loss + (1 - self.config.beta) * inverse_loss + self.config.beta * forward_loss
            self.optimizer.zero_grad()
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.a2c.actor_net.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.a2c.critic_net.parameters(), 1.0)
        
        self.actor_optim.step()
        self.critic_optim.step()
        if self.icm:
            self.optimizer.step()
    
        if self.tb_writer:
            self.tb_writer.add_scalar("Actor_loss/train", actor_loss.item(), self.ep)
            self.tb_writer.add_scalar("Critic_loss/train", critic_loss.item(), self.ep)
            self.tb_writer.add_scalar("Entropy_loss/train", entropy_loss.item(), self.ep)
            self.tb_writer.add_scalar("Total_loss/train", loss.item(), self.ep)
            self.tb_writer.add_scalar("Advantage/train", advantages.mean().item(), self.ep)
            if self.icm:
                self.tb_writer.add_scalar("Forward_loss/train", forward_loss.item(), self.ep)
                self.tb_writer.add_scalar("Inverse_loss/train", inverse_loss.item(), self.ep)
                self.tb_writer.add_scalar("Intrinsic_reward/train", intrinsic_reward.mean(), self.ep)
        
        
    def save(self, episode):
        episode=episode
        actor_filename = f"model_actor_ep{episode}.pt"
        critic_filename = f"model_critic_ep{episode}.pt"
        torch.save(self.a2c.actor_net.state_dict(), actor_filename)
        torch.save(self.a2c.critic_net.state_dict(), critic_filename)

    def load(self):
        self.actor.load_state_dict(torch.load("model_actor.pt"), map_location=self.device)
        self.critic.load_state_dict(torch.load("model_critic.pt"), map_location=self.device)

    def evaluate(self, env: gym.Env) -> None:
        rewards = []

        print(f'\nEvaluating for 5 episodes')
        print('Algorithm: A2C')
        for _ in tqdm(range(10)):
            total_reward = 0
            done = False
            state, _ = env.reset()
            while not done:
                action = self.act(state)
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward

            rewards.append(total_reward)

        print('Mean Reward:', np.mean(rewards))
        print()

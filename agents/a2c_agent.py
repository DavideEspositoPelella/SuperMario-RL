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
from config import Config

import torch.optim as optim
from torch.distributions import Categorical
from collections import namedtuple, deque

import os
os.environ["OMP_NUM_THREADS"] = "1"


class A2CNet(nn.Module):
    def __init__(self, 
                 num_channels: int=4,
                 output_dim: int=5) -> None:
        """         
        Initializes the network for the agent.

        Args:
            - num_channels (int): Number of channels in input. Default to 4.
            - output_dim (int): Output dimension. Default to 5. 
        """
        super(A2CNet, self).__init__()
        # convolutional layers
        self.actor_lr = 0.0001
        self.critic_lr = 0.0005
        actor_layers = [
         nn.Conv2d(in_channels=num_channels,
                   out_channels=32,
                   kernel_size=8, 
                   stride=4),
        nn.Conv2d(in_channels=32,
                  out_channels=64,
                  kernel_size=4,
                  stride=2),
        nn.Conv2d(in_channels=64,
                  out_channels=64,
                  kernel_size=3,
                  stride=1),
        nn.Flatten(),
        nn.Linear(64, 256),
        nn.Linear(256, output_dim)
        ]
        critic_layers = [
        nn.Conv2d(in_channels=num_channels,
                  out_channels=32,
                  kernel_size=8, 
                  stride=4),
        nn.Conv2d(in_channels=32,
                    out_channels=64,
                    kernel_size=4,
                    stride=2),
        nn.Conv2d(in_channels=64,
                    out_channels=64,
                    kernel_size=3,
                    stride=1),
        nn.Flatten(),
        nn.Linear(64, 256),
        nn.Linear(256,1)
        ]

        self.actor = nn.Sequential(*actor_layers)
        self.critic = nn.Sequential(*critic_layers)

        self.critic_optim = optim.RMSprop(self.critic.parameters(), lr=self.critic_lr)
        self.actor_optim = optim.RMSprop(self.actor.parameters(), lr=self.actor_lr)


    
    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        """ 
        Implements the forward pass.
        
        Args:
            - x (torch.Tensor): Input tensor.
        """
        # convolutional layers
        x = torch.Tensor(x).to(self.device)
        values = self.critic(x)
        logits = self.actor(x)
        return logits, values
    

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
        # initialize the configuration settings
        self.episode = 0
        self.gamma = 0.999
        self.lam = 0.95  # hyperparameter for GAE
        self.ent_coef = 0.01  # coefficient for the entropy bonus (to encourage exploration)
        self.actor_lr = 0.001
        self.critic_lr = 0.005
        self.n_updates = 1000

        self.configure_agent(env=env, config=config, prioritized=prioritized, icm=icm)
        self.define_logs_metric(tb_writer=tb_writer, log_dir=log_dir, save_dir=save_dir)

        self.actor_lr = 0.001
        self.critic_lr = 0.005
        actor_layers = [
        nn.Conv2d(in_channels=self.state_dim[0],
                  out_channels=32,
                  kernel_size=8, 
                  stride=4),
        nn.ReLU(),
        nn.Conv2d(in_channels=32,
                  out_channels=64,
                  kernel_size=4,
                  stride=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=64,
                  out_channels=64,
                  kernel_size=3,
                  stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Linear(256, self.num_actions)
        ]
        critic_layers = [
        nn.Conv2d(in_channels=self.state_dim[0],
                  out_channels=32,
                  kernel_size=8, 
                  stride=4),
        nn.ReLU(),
        nn.Conv2d(in_channels=32,
                    out_channels=64,
                    kernel_size=4,
                    stride=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=64,
                    out_channels=64,
                    kernel_size=3,
                    stride=1),
        nn.Flatten(),
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Linear(256,1)
        ]

        self.actor = nn.Sequential(*actor_layers)
        self.critic = nn.Sequential(*critic_layers)

        self.critic_optim = optim.RMSprop(self.critic.parameters(), lr=self.critic_lr)
        self.actor_optim = optim.RMSprop(self.actor.parameters(), lr=self.actor_lr)



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
        self.device = "cpu"
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
        #self.net = A3CNet(num_channels=4, 
        #                  output_dim=self.num_actions).float().to(self.device)    

        if self.icm:
            self.icm_model = ICMModel(input_dim=self.state_dim, 
                                      num_actions=self.num_actions, 
                                      feature_size=self.config.feature_size, 
                                      device=self.device).float().to(self.device)
            
            self.optimizer = torch.optim.Adam(list(self.net.parameters()) + list(self.icm_model.parameters()),lr=self.config.lr)
            self.emb_loss_fn = torch.nn.MSELoss()
            self.inverse_loss_fn = torch.nn.CrossEntropyLoss()
            self.forward_loss_fn = torch.nn.MSELoss()


    def select_action(self, x):
        """
        Returns a tuple of the chosen actions and the log-probs of those actions.

        Args:
            x: A batched vector of states.

        Returns:
            actions: A tensor with the actions, with shape [n_steps_per_update, n_envs].
            action_log_probs: A tensor with the log-probs of the actions, with shape [n_steps_per_update, n_envs].
            state_values: A tensor with the state values, with shape [n_steps_per_update, n_envs].
        """
        state_values, action_logits = self.forward(x)
        action_pd = torch.distributions.Categorical(logits=action_logits)  # implicitly uses softmax
        actions = action_pd.sample()
        action_log_probs = action_pd.log_prob(actions)
        entropy = action_pd.entropy()
        return (actions, action_log_probs, state_values, entropy)
    

    def forward(self, x):
            """
            Forward pass of the networks.

            Args:
                x: A batched vector of states.

            Returns:
                state_values: A tensor with the state values, with shape [n_envs,].
                action_logits_vec: A tensor with the action logits, with shape [n_envs, n_actions].
            """
            #x = torch.Tensor(x).to(self.device)
            if isinstance(x, np.ndarray) or hasattr(x, '__array__'):
                x = np.array(x)  # This works for numpy arrays and objects with a __array__ method
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, device=self.device, dtype=torch.float32)


            state_values = self.critic(x)  
            action_logits_vec = self.actor(x)  
            return (state_values, action_logits_vec)
    

    def act(self, state, exploration=True):
        #print(state.shape)
        #state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
        #print(state.shape)
        #state = torch.tensor(state, device=self.device).unsqueeze(0)
        #print(state.shape)
    
        value, actor_features = self.forward(state)

        dist = Categorical(logits=actor_features)
        chosen_action = dist.sample().item()  # Convert to Python scalar
        log_prob = dist.log_prob(torch.tensor(chosen_action, device=self.device))
        
        return chosen_action, log_prob, value
    
    def preprocess(self, state):
        state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
        state = torch.tensor(state, device=self.device).unsqueeze(0)
        return state
    
    def train(self):
        critic_losses = []
        actor_losses = []
        entropies = []
        n_updates = 100
        n_steps_per_update = 128
        self.episode = 0

        print("Start training.")
        scores_deque = deque(maxlen=100)

        n_training_episodes = 35000
        gamma = 0.99
        print_every = 10

        frequency_update = 20
        frequency_save = 300

        all_lengths = []
        average_lengths = []
        all_rewards = []
        entropy_term = 0
        D = torch.zeros(1000, 6)

        
        for i_episode in tqdm(range(1, n_training_episodes + 1)):
            saved_log_probs = []  # stores log probs during episode
            saved_rewards = []    # stores rewards during episode
            saved_actions = []    # stores actions
            saved_done = []       # stores done flag
            saved_values = []     # stores the values
            saved_states = []     # stores states

            states, _ = self.env.reset()
            done = False
            #Rollout phase
            while not done:

                # store the state
                saved_states.append(states)

                action, log_prob, value = self.act(self.preprocess(states))

                # store log_prob
                saved_log_probs.append(log_prob)

                #STEP Simulator
                states, rewards, terminated, truncated, info  = self.env.step(action)
                done = terminated or truncated

                saved_done.append(done)

                # STORE all
                saved_rewards.append(rewards)
                saved_actions.append(action)
                saved_values.append(value)

                if done:
                    break

                if i_episode % frequency_update:
                    # Learning Phase:
                    v_next = self.critic(self.preprocess(states))

                    A = self.GAE_advantage(saved_rewards,
                                            saved_log_probs,
                                            saved_done,
                                            v_next,
                                            0.9,
                                            0.95)

                    #R = A + saved_values.detach().numpy()
                    R = A + torch.vstack(saved_values)

                    # Convert lists to PyTorch tensors
                    #states_batch = torch.vstack(saved_states)
                    #actions_batch = torch.tensor(saved_actions, dtype=torch.long)
                    #rewards_batch = torch.tensor(saved_rewards, dtype=torch.float32)
                    #done_batch = torch.tensor(saved_done, dtype=torch.float32)

                   # advantages_batch = torch.tensor(A, dtype=torch.float32)

                    #returns_batch = torch.tensor(R, dtype=torch.float32)

                    # Flatten the batch
                    #M = (states_batch, actions_batch, rewards_batch, done_batch, advantages_batch, returns_batch)
                    self.optimize_model(self.preprocess(states), saved_actions, saved_rewards, saved_done, A, R)

                    scores_deque.append(sum(saved_rewards))
                    #reinitialize
                    saved_log_probs = []  
                    saved_rewards = []    
                    saved_actions = []    
                    saved_states = []   
                    saved_done = []    
                    saved_values = []     

            scores_deque.append(sum(saved_rewards))

            # Print training statistics
            if i_episode % print_every == 0:
                average_length = np.mean(scores_deque)
                all_lengths.append(len(saved_rewards))
                average_lengths.append(average_length)
                all_rewards.append(np.sum(saved_rewards))
                print(f"Episode {i_episode}\tAverage Score: {average_length:.2f}")

            # Save model weights
            if i_episode % frequency_save == 0:
                self.save(i_episode)

        print("Training Finished.")
        return 
    
    def GAE_advantage(self, rewards, values, dones, next_value, discount_factor, gae_parameter):
        T = len(rewards)
        advantages = torch.zeros(T, dtype=torch.float32, device=self.device)

        # Convert rewards, dones, and next_value to PyTorch tensors
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        next_value = torch.tensor(next_value, dtype=torch.float32, device=self.device)

        ones = torch.ones((T,)).to(self.device)
        values = torch.tensor(values, dtype=torch.float32, device=self.device)

        # Compute temporal differences
        deltas = rewards + (ones - dones) * discount_factor * next_value - values

        # Calculate GAE advantage
        advantage = 0
        for t in reversed(range(T)):
            advantage = deltas[t] + (discount_factor * gae_parameter) * (1 - dones[t]) * advantage
            advantages[t] = advantage

        return advantages.to(self.device)
    
    def optimize_model(self, states, actions, rewards, dones, advantages, returns):
        observations = states
        
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float, device=self.device)
        advantages = torch.tensor(advantages, dtype=torch.float, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float, device=self.device)

        # Forward pass
        values, actor_features = self.forward(observations)
        dist = Categorical(logits=actor_features)
        action_log_probs = dist.log_prob(actions)

        # Compute actor and critic loss
        actor_loss = -(action_log_probs * advantages).mean()
        
        critic_loss = F.mse_loss(values, returns)

        # Entropy loss for regularization
        entropy_loss = dist.entropy().mean()

        # Total loss with a coefficient for entropy regularization (encourage exploration)
        total_loss = -actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss

        # Backward pass and optimization for both actor and critic
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        total_loss.backward()

        # Clip gradients to prevent excessively large updates
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)

        self.actor_optim.step()
        self.critic_optim.step()

        
    def get_losses(
        self,
        rewards: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        entropy: torch.Tensor,
        masks: torch.Tensor,
        gamma: float,
        lam: float,
        ent_coef: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the loss of a minibatch (transitions collected in one sampling phase) for actor and critic
        using Generalized Advantage Estimation (GAE) to compute the advantages 
        Args:
            rewards: A tensor with the rewards for each time step in the episode, with shape [n_steps_per_update, n_envs].
            action_log_probs: A tensor with the log-probs of the actions taken at each time step in the episode, with shape [n_steps_per_update, n_envs].
            value_preds: A tensor with the state value predictions for each time step in the episode, with shape [n_steps_per_update, n_envs].
            masks: A tensor with the masks for each time step in the episode, with shape [n_steps_per_update, n_envs].
            gamma: The discount factor.
            lam: The GAE hyperparameter. (lam=1 corresponds to Monte-Carlo sampling with high variance and no bias,
                                          and lam=0 corresponds to normal TD-Learning that has a low variance but is biased
                                          because the estimates are generated by a Neural Net).
            device: The device to run the computations on (e.g. CPU or GPU).

        Returns:
            critic_loss: The critic loss for the minibatch.
            actor_loss: The actor loss for the minibatch.
        """
        T = len(rewards)
        advantages = torch.zeros(T, device=device)

        # compute the advantages using GAE
        gae = 0.0
        for t in reversed(range(T - 1)):
            td_error = (
                rewards[t] + gamma * masks[t] * value_preds[t + 1] - value_preds[t]
            )
            gae = td_error + gamma * lam * masks[t] * gae
            advantages[t] = gae

        # calculate the loss of the minibatch for actor and critic
        critic_loss = advantages.pow(2).mean()

        # give a bonus for higher entropy to encourage exploration
        actor_loss = (-(advantages.detach() * action_log_probs).mean() - ent_coef * entropy.mean())
        return (critic_loss, actor_loss)




    def update_parameters(
        self, critic_loss: torch.Tensor, actor_loss: torch.Tensor
    ) -> None:
        """
        Updates the parameters of the actor and critic networks.

        Args:
            critic_loss: The critic loss.
            actor_loss: The actor loss.
        """
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
    
    def save(self, episode):
        episode=episode
        actor_filename = f"model_actor_ep{episode}.pt"
        critic_filename = f"model_critic_ep{episode}.pt"
        torch.save(self.actor.state_dict(), actor_filename)
        torch.save(self.critic.state_dict(), critic_filename)

    def load(self):
        self.actor.load_state_dict(torch.load("model_actor.pt"), map_location=self.device)
        self.critic.load_state_dict(torch.load("model_critic.pt"), map_location=self.device)

    def evaluate(self, env: gym.Env) -> None:
        rewards = []

        print(f'\nEvaluating for 5 episodes')
        print('Algorithm: {}'.format('DDQN_PER' if self.prioritized else 'DDQN'))
        for _ in tqdm(range(5)):
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

from tqdm import tqdm
import numpy as np
from pathlib import Path
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import gym
from gym.spaces import Box
from gym.wrappers import FrameStack 
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import gym_super_mario_bros

from nes_py.wrappers import JoypadSpace

from torchrl.data import TensorDictPrioritizedReplayBuffer, LazyTensorStorage, TensorDictReplayBuffer
from tensordict import TensorDict


class Net(nn.Module):
    def __init__(self, 
                 num_channels: int,
                 output_dim: int) -> None:
        """         
        Initializes the network for the agent.

        Args:
            - num_actions (int): Number of actions (discrete environment). Default to 5.
            - in_channels (int): Number of input channels (number of frames stacked). Default to 4.
            - hidden_size (int): Hidden size for the network. Default to 64.
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
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, output_dim)
    
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
        x = self.fc2(x) 
        return x 
    

class DDQNetwork(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 output_dim: int) -> None:
        """
        Initializes the Deep Q-Network.

        Args:
            - input_dim (int): Input dimension.
            - output_dim (int): Output dimension.
        """
        c, h, w = input_dim
        super(DDQNetwork, self).__init__()
        # initialize the online and the target networks
        self.online_net = Net(c, output_dim) 
        self.target_net = Net(c, output_dim)
        self.target_net.load_state_dict(self.online_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad = False

    def forward(self, 
                input: torch.Tensor, 
                model:  str='online') -> torch.Tensor:
        """
        Implements the forward pass.
        
        Args:
            - input (torch.Tensor): Input tensor.
            - model (str): Model to use. Can be 'online' or 'target'. Default to 'online'.
        """
        if model == 'online':
            return self.online_net(input)
        elif model == 'target':
            return self.target_net(input)
        

class SkipFrame(gym.Wrapper):
    def __init__(self, 
                 env, 
                 skip_frame):
        super().__init__(env)
        self._skip_frame = skip_frame

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip_frame):
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        transform = torchvision.transforms.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(self.shape, antialias=True), torchvision.transforms.Normalize(0, 255)]
        )
        observation = transform(observation).squeeze(0)
        return observation


class ResetWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        if isinstance(obs, np.ndarray) and len(obs.shape) > 1:
            return [(single_obs, {}) for single_obs in obs], info
        return obs, info
    

class Config:
    def __init__(self, 
                 multi_env: bool=True,
                 num_envs: int=2,
                 skip_frame: int=2,
                 exploration_rate: float=1,
                 exploration_rate_decay: float=0.999,
                 exploration_rate_min: float=0.1,
                 memory_size: int=10000,
                 burn_in: int=100,
                 epsilon_buffer: float=1e-8, 
                 alpha: float=0.6, 
                 beta: float=0.4, 
                 gamma: float=0.99,
                 batch_size: int=32,  
                 lr: float=0.0001,
                 update_freq: int=3, 
                 sync_freq: int=1000, 
                 episodes: int=1000) -> None:
        """
        Initializes the configuration settings.

        Args:
            - multi_env (bool): Whether to use multiple environments. Default to True.
            - num_envs (int): Number of environments. Default to 2.
            - skip_frame (int): Number of frames to skip. Default to 2.
            - exploration_rate (float): Exploration rate. Default to 1.
            - exploration_rate_decay (float): Decay value for the exploration rate. Default to 0.999.
            - exploration_rate_min (float): Minimum value for the exploration rate. Default to 0.1.
            - memory_size (int): Size of the buffer. Default to 10000.
            - burn_in (int): Number of experiences to burn in. Default to 100.
            - alpha (float): Priority exponent. Default to 0.6.
            - beta (float): Importance sampling exponent. Default to 0.4.
            - epsilon_buffer (float): Small constant to avoid zero priority. Default to 1e-8.
            - gamma (float): Discount factor. Default to 0.99.
            - batch_size (int): Batch size for training. Default to 32.
            - lr (float): Learning rate. Default to 0.0001.
            - update_freq (int): Frequency of updating the online network. Default to 3.
            - sync_freq (int): Frequency of updating the target network. Default to 1000.
            - episodes (int): Maximum number of episodes. Default to 1000.
        """
        self.multi_env = multi_env
        self.num_envs = num_envs
        self.skip_frame = skip_frame
        self.exploration_rate = exploration_rate
        self.exploration_rate_decay = exploration_rate_decay
        self.exploration_rate_min = exploration_rate_min
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.alpha = alpha
        self.beta = beta
        self.epsilon_buffer = epsilon_buffer
        self.gamma = gamma
        self.batch_size = batch_size
        self.lr = lr
        self.update_freq = update_freq
        self.sync_freq = sync_freq
        self.episodes = episodes


class DDQN(nn.Module):

    def __init__(self, 
                 episodes: int=10000, 
                 prioritized: bool=True):
        super(DDQN, self).__init__()
        # initialize the configuration settings
        self.config = Config(multi_env = False,
                             num_envs=2,
                             skip_frame = 2,
                             exploration_rate=1,
                             exploration_rate_decay=0.999,
                             exploration_rate_min=0.1,
                             memory_size=10000,
                             burn_in=200,  
                             alpha=0.6,
                             beta=0.5,
                             epsilon_buffer=0.01,
                             gamma=0.99,
                             batch_size=32,                            
                             lr=0.0001,
                             update_freq=3,
                             sync_freq=1000,
                             episodes=episodes)
        self.prioritized = prioritized
        # create the environment
        self.env = self.make_env(multi=self.config.multi_env,
                                 num_envs=self.config.num_envs)
        # extract the state dimension and the number of actions 
        self.state_dim = (4, 84, 84)
        self.num_actions=self.env.action_space.n
        # initialize the networks
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = DDQNetwork(input_dim=self.state_dim, 
                              output_dim=self.num_actions).float().to(self.device)
        if self.prioritized:
            # initialize the prioritized replay buffer
            self.buffer = TensorDictPrioritizedReplayBuffer(storage=LazyTensorStorage(max_size = self.config.memory_size),
                                                            alpha=self.config.alpha, 
                                                            beta=self.config.beta,
                                                            priority_key='td_error',
                                                            eps=self.config.epsilon_buffer,
                                                            batch_size=self.config.batch_size)
        else:
            self.buffer = TensorDictReplayBuffer(storage=LazyTensorStorage(max_size = self.config.memory_size))
            self.loss_fn = torch.nn.SmoothL1Loss()
        # initialize the optimizer and the loss function
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr)
        self.curr_step = 0 #TODO remove this
        
        if self.prioritized:
            self.dir  = Path("checkpoints/ddqn_per")
        else:
            self.dir  = Path("checkpoints/ddqn")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.save_every = 100

        #TODO add logging
        self.print_every = 20

        
                
    def make_env(self, 
                 multi=True, 
                 num_envs=2):
        if multi:
            env = gym.vector.make('SuperMarioBrosRandomStages-v3', 
                                  stages=['1-4', '2-4', '3-4', '4-4'], 
                                  num_envs=num_envs, 
                                  asynchronous=False)
        else:
            env_ID = "SuperMarioBros-1-1-v0"
            if gym.__version__ < '0.26':
                env = gym_super_mario_bros.make(env_ID, new_step_api=True)

            else:
                env = gym_super_mario_bros.make(env_ID, render_mode='human', apply_api_compatibility=True)

        # apply reset wrapper and other wrappers
        if multi:
            env = ResetWrapper(env)
        env = SkipFrame(env, skip_frame=self.config.skip_frame)
        env = GrayScaleObservation(env)
        env = ResizeObservation(env, shape=84)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        if gym.__version__ < '0.26':
            env = FrameStack(env, 
                             num_stack=4, 
                             new_step_api=True)
        else:
            env = FrameStack(env, 
                             num_stack=4)
        return env

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

        self.curr_step += 1
        return action_idx
    
        
    def store(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
            
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])
        
        data = TensorDict({"state": state, 
                           "next_state": next_state, 
                           "action": action, 
                           "reward": reward, 
                           "done": done}, batch_size=[])
        self.buffer.add(data)

    def sample(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch, info = self.buffer.sample(self.config.batch_size, return_info=True)
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        return state.to(self.device), next_state.to(self.device), action.to(self.device).squeeze(), reward.to(self.device).squeeze(), done.to(self.device).squeeze(), info

    def learn(self):
        """Update online action value (Q) function with a batch of experiences"""
        if self.curr_step % self.config.sync_freq == 0:
            self.net.target_net.load_state_dict(self.net.online_net.state_dict())

        if self.curr_step < self.config.burn_in:
            return None, None

        if self.curr_step % self.config.update_freq != 0:
            return None, None

        state, next_state, action, reward, done, info = self.sample()

        q_est = self.net(state, model="online")[
            np.arange(0, self.config.batch_size), action]

        with torch.no_grad():
            next_state_Q = self.net(next_state, model="online")
            best_action = torch.argmax(next_state_Q, axis=1)
            next_Q = self.net(next_state, model="target")[
                np.arange(0, self.config.batch_size), best_action]
            q_tgt = (reward + (1 - done.float()) * self.config.gamma * next_Q).float()
        td_error = q_tgt - q_est 
        if self.prioritized:
            priority = td_error.abs() + self.config.epsilon_buffer
            importance_sampling_weights = torch.FloatTensor(info["_weight"]).reshape(-1, 1).to(self.device)
            loss = (importance_sampling_weights * td_error.pow(2)).mean()
            self.buffer.update_priority(info["index"], priority)
        else:
            loss = self.loss_fn(q_est, q_tgt)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return td_error, loss.item()
        
    def train(self):
        rewards = []
        episodes = self.config.episodes

        print("Starting...")
        print(f"Training for: {episodes} episodes\n")
        for e in tqdm(range(episodes)):
            self.curr_step = 0 #TODO remove this
            reset_output = self.env.reset()

            self.ep = e

            # Handle multi-environment
            if self.config.multi_env:
                # Extract the first observation from each sub-environment
                state = np.array([obs_info[0] for obs_info in reset_output])
            else:
                # Handle single environment
                state, _ = reset_output
            
            total_reward = 0
            while True:
                action = self.act(state)
                
                next_state, reward, done, _, info = self.env.step(action)
                
                total_reward += reward

                self.store(state, next_state, action, reward, done)

                q, loss = self.learn()

                state = next_state

                if done or info["flag_get"]:
                    break

            rewards.append(total_reward)
            
            if self.ep % self.print_every == 0:
                print(f"\nEpisode: {self.ep} Mean reward: {np.mean(rewards)}")

            
            if self.ep % self.save_every == 0:
                self.save()


        print("Training complete.\n")
        return
    
    def save(self):
        save_path = (self.dir / f"mario_net_{self.ep}.chkpt")
        torch.save(
            dict(model=self.net.state_dict(), 
                 exploration_rate=self.config.exploration_rate), 
                 save_path)
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    # TODO more modular loading
    def load(self, model_path: str=None):
        if model_path == "mario_net_0.chkpt":
            return
        
        load_path = self.dir / model_path
        
        if not load_path.exists():
                raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=self.device)
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')

        print(f"\nLoading model at    {load_path}   with exploration rate {exploration_rate}")
        self.net.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret

    def evaluate(self, episodes: int=5) -> None:
        env = self.make_env(multi=False)
        rewards = []

        print('\nEvaluating for 5 episodes')
        print('Algorithm: {}'.format('DDQN_PER' if self.prioritized else 'DDQN'))
        for episode in tqdm(range(5)):
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


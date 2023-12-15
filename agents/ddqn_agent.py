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
        self.net = DDQNetwork(input_dim=self.state_dim, 
                              output_dim=self.num_actions).float().to(self.device)
        if self.prioritized:
            self.buffer = TensorDictPrioritizedReplayBuffer(storage=LazyTensorStorage(max_size = self.config.memory_size),
                                                            alpha=self.config.alpha, 
                                                            beta=self.config.beta,
                                                            priority_key='td_error',
                                                            eps=self.config.epsilon_buffer,
                                                            batch_size=self.config.batch_size)
        else:
            self.buffer = TensorDictReplayBuffer(storage=LazyTensorStorage(max_size = self.config.memory_size))
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
        """#TODO: add docstring
        #TODO simplify this function
        """
        
        state, next_state, action, reward, done, info = self.sample()
        torch.clamp(reward, -1, 1)
        one_hot_action = F.one_hot(action, num_classes=self.num_actions).float()
        total_reward = reward

        if self.icm:
            next_state_feat, pred_next_state_feat, pred_action = self.icm_model(state, next_state, one_hot_action)
            inverse_loss = self.inverse_loss_fn(pred_action, action)
            forward_loss = self.forward_loss_fn(pred_next_state_feat, next_state_feat)  

            intrinsic_reward = self.config.eta * F.mse_loss(next_state_feat, pred_next_state_feat, reduction='none').mean(-1)
            total_reward = reward + intrinsic_reward

        q_est = self.net(state, model="online")[np.arange(0, self.config.batch_size), action]
            
        with torch.no_grad():
            next_state_Q = self.net(next_state, model="online")
            best_action = torch.argmax(next_state_Q, axis=1)
            next_Q = self.net(next_state, model="target")[np.arange(0, self.config.batch_size), best_action]    
            q_tgt = (reward + (1 - done.float()) * self.config.gamma * next_Q).float()
        td_error = q_tgt - q_est 

        if self.prioritized:
            priority = td_error.abs() + self.config.epsilon_buffer
            importance_sampling_weights = torch.FloatTensor(info["_weight"]).reshape(-1, 1).to(self.device)
            loss = (importance_sampling_weights * td_error.pow(2)).mean()
            self.buffer.update_priority(info["index"], priority)
        else:
            loss = self.loss_fn(q_est, q_tgt)
        loss = torch.clamp(loss, -1, 1)
            
        if self.icm:
            loss = self.config.lambda_icm * loss + (1 - self.config.beta) * inverse_loss + self.config.beta * forward_loss
            
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()
        
        if self.tb_writer and self.curr_step_global % self.loss_log_freq == 0:
            self.tb_writer.add_scalar("Loss/train", loss.item(), self.curr_step_global)
            self.tb_writer.add_scalar("Total_reward/train", total_reward.mean(), self.curr_step_global)
            self.tb_writer.add_scalar("Td_error/train", td_error.mean(), self.curr_step_global)                
            if self.icm:
                self.tb_writer.add_scalar("Forward_loss/train", forward_loss.item(), self.curr_step_global)
                self.tb_writer.add_scalar("Inverse_loss/train", inverse_loss.item(), self.curr_step_global)
                self.tb_writer.add_scalar("Intrinsic_reward/train", intrinsic_reward.mean(), self.curr_step_global)
                
        return td_error, loss.item()
        
    def train(self):
        """#TODO: add docstring"""
        episodes = self.config.episodes

        # Burn-in
        print("Populate replay buffer")
        reset_output = self.env.reset()
        state, _ = reset_output
        while self.curr_step_global < self.config.burn_in:
            action = self.act(state)
            next_state, reward, done, _, info = self.env.step(action) 
            self.store(state, next_state, action, reward, done)
            if done:
                state, _  = self.env.reset()
        print("Burn in complete")

        print("Starting...")
        print(f"Training for: {episodes} episodes\n")
        self.curr_step_global = 0
        self.curr_step_local = 0
        for e in tqdm(range(episodes)):
            state, _ = self.env.reset()
            self.ep = e

            while True:
                action = self.act(state)
                
                next_state, extrinsic_reward, done, _, info = self.env.step(action)

                self.store(state, next_state, action, extrinsic_reward, done)

                if self.curr_step_global % self.config.update_freq == 0:
                    td_err, loss = self.learn()
                
                if self.curr_step_global % self.config.sync_freq == 0:
                    self.net.target_net.load_state_dict(self.net.online_net.state_dict())

                state = next_state

                if done or info["flag_get"]:
                    break

            if self.ep % self.save_freq == 0:
                self.save()
        print("Training complete.\n")
        return
    
    def save(self):
        """Save the model to a checkpoint."""
        save_path = (self.save_dir / f"mario_net_{self.ep}.chkpt")

        torch.save(
            dict(model=self.net.state_dict(), 
                 icm_model=self.icm_model.state_dict() if self.icm else None,
                #  ddqn_agent=self
                 ), save_path)
        
    def load(self, model_path: str=None):
        """Load a model from a checkpoint"""

        if model_path == "mario_net_0.chkpt":
            # print("No model to load")
            return
        
        load_path = self.save_dir / model_path
        
        if not load_path.exists():
                raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=self.device)
        print(ckp.keys())
        state_dict = ckp.get('model')

        # print(f"\nLoading model at    {load_path}   with exploration rate {exploration_rate}")
        # self.net.load_state_dict(state_dict)
        # self.exploration_rate = exploration_rate

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret

    # TODO is this compliant with the new ICM module?
    def evaluate(self, episodes: int=5) -> None:
        env = self.make_env()
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


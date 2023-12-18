import torch
import torch.nn as nn
from typing import Tuple


class ICMModel(nn.Module):
    def __init__(self, 
                 input_dim: tuple, 
                 num_actions: int, 
                 feature_size: int, 
                 device: torch.device) -> None:
        """
        Initializes the ICM model.

        Args:
            - input_dim (tuple): The shape of the observation. Expects a tuple of the form (C, H, W).
            - num_actions (int): The number of actions available in the environment. We assume that the actions are discrete.
            - feature_size (int): The dimension used to embed the state.
            - device (torch.device): The device to use.
        """
        super(ICMModel, self).__init__()
        num_channels = input_dim[0]
        self.device = device
        # network to encode the observation
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=num_channels,
                      out_channels=32,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ELU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ELU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ELU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(num_features=32), 
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(288, feature_size)
        )
        # network to predict the action given the current and next state embeddings
        self.inverse_net = nn.Sequential(nn.Linear(feature_size * 2, 512),
                                         nn.ReLU(),
                                         nn.Linear(512, num_actions))
        # forward model to predict the next state embedding given the current state embedding and action
        self.forward_1 = nn.Sequential(
            nn.Linear(num_actions + feature_size, feature_size),
            nn.LeakyReLU())
        # residual network to preserve the initial information in forward model
        self.residual = [nn.Sequential(
            nn.Linear(num_actions + feature_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, feature_size)).to(self.device)] * 8
        self.forward_net_2 = nn.Sequential(
            nn.Linear(feature_size + num_actions, feature_size))
        # initialize the weights
        self.init_weights()
    
    def init_weights(self) -> None:
        """
        Initializes the weights of the network with Kaiming normal initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, 
                state: torch.Tensor, 
                next_state: torch.Tensor, 
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:    
        """
        Implements the forward pass.

        Args:
            - state (torch.Tensor): The current state.
            - next_state (torch.Tensor): The next state.
            - action (torch.Tensor): The action taken.
    
        Returns:
            - encoded_next_state (torch.Tensor): The encoded next state.
            - pred_next_state_feat (torch.Tensor): The predicted next state embedding.
            - pred_action (torch.Tensor): The predicted action.
        """
        # encode the state and next state
        encoded_state = self.feature(state)
        encoded_next_state = self.feature(next_state)
        # predict the action
        concat_states = torch.cat((encoded_state, encoded_next_state), 1)
        pred_action = self.inverse_net(concat_states)
        # predict the next state
        concat_state_action = torch.cat((encoded_state, action), 1)
        pred_next_state_feat_1 = self.forward_1(concat_state_action)
        # residual blocks
        for i in range(4):
            pred_next_state_feat = self.residual[i * 2](torch.cat((pred_next_state_feat_1, action), 1))
            pred_next_state_feat_1 = self.residual[i * 2 + 1](torch.cat((pred_next_state_feat, action), 1)) + pred_next_state_feat_1
        # final layer of the forward model
        pred_next_state_feat = self.forward_net_2(torch.cat((pred_next_state_feat_1, action), 1))
        return  encoded_next_state, pred_next_state_feat, pred_action
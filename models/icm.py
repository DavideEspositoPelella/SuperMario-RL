import torch
import torch.nn as nn
import torch.nn.functional as F



class ICMModel(nn.Module):
    def __init__(self, input_dim, num_actions, feature_size, device):
        super(ICMModel, self).__init__()
        num_channels = input_dim[0]
        self.device = device
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

        self.inverse_net = nn.Sequential(nn.Linear(feature_size * 2, 512),
                                         nn.ReLU(),
                                         nn.Linear(512, num_actions))

        self.forward_net = nn.Sequential(
            nn.Linear(num_actions + feature_size, feature_size),
            nn.ReLU(),
            nn.Linear(feature_size, feature_size))

    def forward(self, state, next_state, action):    
        encoded_state = self.feature(state)
        encoded_next_state = self.feature(next_state)
        
        concat_states = torch.cat((encoded_state, encoded_next_state), 1)
        pred_action = self.inverse_net(concat_states)

        concat_state_action = torch.cat((encoded_state, action), 1)
        pred_next_state_feat = self.forward_net(concat_state_action)
        
        return pred_next_state_feat, encoded_next_state, pred_action
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
        
        self.residual = [nn.Sequential(
            nn.Linear(num_actions + feature_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, feature_size),
        ).to(self.device)] * 8

        self.forward_1 = nn.Sequential(
            nn.Linear(num_actions + feature_size, feature_size),
            nn.LeakyReLU()
        )
        self.forward_net_2 = nn.Sequential(
            nn.Linear(feature_size + num_actions, feature_size))
        self.init_weights()
    
    def init_weights(self):
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

    def forward(self, state, next_state, action):    
        encoded_state = self.feature(state)
        encoded_next_state = self.feature(next_state)
        
        concat_states = torch.cat((encoded_state, encoded_next_state), 1)
        pred_action = self.inverse_net(concat_states)

        concat_state_action = torch.cat((encoded_state, action), 1)
        pred_next_state_feat_1 = self.forward_1(concat_state_action)

        # residual
        for i in range(4):
            pred_next_state_feat = self.residual[i * 2](torch.cat((pred_next_state_feat_1, action), 1))
            pred_next_state_feat_1 = self.residual[i * 2 + 1](
                torch.cat((pred_next_state_feat, action), 1)) + pred_next_state_feat_1
        
        return pred_next_state_feat, encoded_next_state, pred_action
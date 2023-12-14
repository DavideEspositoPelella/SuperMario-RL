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
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(num_features=32), 
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

        self.forward_net_1 = nn.Sequential(
            nn.Linear(num_actions + feature_size, feature_size),
            nn.LeakyReLU()
        )
        self.forward_net_2 = nn.Sequential(
            nn.Linear(num_actions + feature_size, feature_size),
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                nn.init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                nn.init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()

    def forward(self, state, next_state, action):
    
        encode_state = self.feature(state)
        encode_next_state = self.feature(next_state)
        pred_action = torch.cat((encode_state, encode_next_state), 1)
        pred_action = self.inverse_net(pred_action)

        pred_next_state_feature_orig = torch.cat((encode_state, action), 1)

        pred_next_state_feature_orig = self.forward_net_1(pred_next_state_feature_orig)

        for i in range(4):
            pred_next_state_feature = self.residual[i * 2](torch.cat((pred_next_state_feature_orig, action), 1))
            pred_next_state_feature_orig = self.residual[i * 2 + 1](
                torch.cat((pred_next_state_feature, action), 1)) + pred_next_state_feature_orig

        pred_next_state_feature = self.forward_net_2(torch.cat((pred_next_state_feature_orig, action), 1))

        real_next_state_feature = encode_next_state
        return real_next_state_feature, pred_next_state_feature, pred_action
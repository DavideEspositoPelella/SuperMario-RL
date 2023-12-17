import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

class Net(nn.Module):
    def __init__(self, 
                 num_channels: int=4,
                 output_dim: int=5) -> None:
        """         
        Initializes the network for the agent.

        Args:
            - num_channels (int): Number of channels in input. Default to 4.
            - output_dim (int): Output dimension. Default to 5. 
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
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, output_dim)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.constant_(m.bias, 0)

    
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
        c = input_dim[0]
        super(DDQNetwork, self).__init__()
        # initialize the online and the target networks
        self.online_net = Net(c, output_dim) 
        self.online_net.init_weights() 
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
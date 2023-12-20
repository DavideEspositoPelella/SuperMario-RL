import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class A2C(nn.Module):
    def __init__(self, 
                 input_dim: tuple=(4, 42, 42),
                 num_actions: int=5) -> None:
        """
        Initialize the A2C module.
        
        Args:  
            - input_dim (tuple): Input shape. Default to (4, 42, 42).
            - num_actions (int): Output dimension for the actor network (number of actions). Default to 5.
        """
        super(A2C, self).__init__()
        c = input_dim[0]
        self.actor_net = Actor(num_channels=c, output_dim=num_actions)
        self.critic_net = Critic(num_channels=c)
        self.actor_net.init_weights()
        self.critic_net.init_weights()
    
    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass.

        Args:
            - x (torch.Tensor): Input tensor.

        Returns:
            - x (torch.Tensor): Output tensor.
        """
        # actor network
        action_probs = self.actor_net(x)
        # critic network
        state_values = self.critic_net(x)
        return action_probs, state_values
        
class Actor(nn.Module):
    def __init__(self, 
                 num_channels: int=4, 
                 output_dim: int=5) -> None:
        """
        Initialize the actor module.

        Args:
            - num_channels (int): Number of input channels. Default to 4.
            - output_dim (int): Output dimension (number of actions). Default to 5.
        """
        super(Actor, self).__init__()
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
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, output_dim)
        
    def init_weights(self) -> None:
        """
        Initializes the weights of the network with Kaiming normal initialization.
        """
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

        Returns:
            - x (torch.Tensor): Output tensor.
        """
        # convolutional layers
        x = F.relu(self.conv1(x)) 
        x = F.relu(self.conv2(x)) 
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        # linear layers
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x
    
    def add_weight_noise(self, scalar=.1):
        for layer in [self.conv1, self.conv2, self.conv3, self.fc1, self.fc2]:
            layer.weight.data += torch.randn_like(layer.weight.data) * scalar


class Critic(nn.Module):
    def __init__(self, 
                 num_channels: int=4) -> None:
        """
        Initialize the critic module.

        Args:
            - num_channels (int): Number of input channels. Default to 4.
        """
        super(Critic, self).__init__()
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
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 1)
        
    def init_weights(self) -> None:
        """
        Initializes the weights of the network with Kaiming normal initialization.
        """
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

        Returns:
            - x (torch.Tensor): Output tensor.
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
    



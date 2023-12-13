import torch
import torch.nn as nn
import torch.nn.functional as F




class embedding(nn.Module):
    def __init__(self, 
                 num_channels: int,
                 output_dim: int) -> None:
        """
        Network which takes as input feature embedding of 
        actual and next state and outputs the predicted action
        """
        super(embedding, self).__init__()
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



class inverseModel(nn.Module):
    def __init__(self, 
                 feature_size: int,
                 num_actions: int) -> None:
        """
        Network which takes as input feature embedding of 
        actual and next state and outputs the predicted action
        """
        super(inverseModel, self).__init__()

        #self.feature_size = 512
        #self.embedding = embedding(num_channels, self.feature_size)

        self.inverse_net = nn.Sequential(
            nn.Linear(feature_size*2, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, num_actions)
        )

    def forward(self, 
                state: torch.Tensor, 
                next_state: torch.Tensor) -> torch.Tensor:
        """ 
        Implements the forward pass.
          
        Args:
            - state (torch.Tensor): Input tensor.
            - next_state (torch.Tensor): Input tensor.
        """
        #state_embedding = self.embedding(state)
        #next_state_embedding = self.embedding(next_state)
        x = torch.cat((state, next_state), dim=1)
        return self.inverse_net(x)


class forwardModel(nn.Module):
    def __init__(self, 
                 action_dim: int,
                 out_channels) -> None:
        """
        Network which takes as input feature embedding of 
        actual state and predicted action and outputs the 
        predicted next state
        """
        super(forwardModel, self).__init__()

        #self.feature_size = 512
        #self.embedding_state = embedding(num_channels, self.feature_size)

        self.forward_net = nn.Sequential(
            nn.Linear(out_channels + 1, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, out_channels)
        )

    def forward(self, 
                state: torch.Tensor, 
                action: torch.Tensor) -> torch.Tensor:
        """ 
        Implements the forward pass.
          
        Args:
            - state (torch.Tensor): Input tensor.
            - action (torch.Tensor): Input tensor.
        """
        # Ensure action tensor is 2D (batch size x 1)
        if action.dim() == 1:
            action = action.unsqueeze(1)  # Reshape action to be 2D

        #state_embedding = self.embedding_state(state)
        x = torch.cat((state, action), dim=1)
        return self.forward_net(x)
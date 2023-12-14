class Config:
    def __init__(self, 
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
                 episodes: int=1000,
                 feature_size: int=288,
                 n_scaling: float=1.0,
                 beta_icm: float = 0.2,
                 lambda_icm: float = 0.1) -> None:
        """
        Initializes the configuration settings.

        Args:
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
            - episodes (int): Number of episodes to train. Default to 2000.
            - feature_size (int): Size of the feature embedding. Default to 288.
            - n_scaling (float): Weights the importance of the policy loss against the intrinsic reward. Default to 0.5.
            - beta_icm (float): Weights the importance of the forward loss against the inverse loss. Default to 0.5.
            - lambda_icm (float): Discount factor for the intrinsic reward. Default to 0.5.
        """
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
        self.feature_size = feature_size
        self.n_scaling = n_scaling
        self.beta_icm = beta_icm
        self.lambda_icm = lambda_icm

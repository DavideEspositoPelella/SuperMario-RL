class Config:
    def __init__(self, 
                 skip_frame: int=2,
                 stack: int=4,
                 resize_shape: int=42,
                 batch_size: int=32,  
                 exploration_rate: float=1,
                 exploration_rate_decay: float=0.999,
                 exploration_rate_min: float=0.1,
                 memory_size: int=10000,
                 burn_in: int=100,
                 lr: float=0.0001,
                 update_freq: int=3, 
                 sync_freq: int=1000, 
                 alpha: float=0.6, 
                 beta: float=0.4, 
                 epsilon_buffer: float=1e-8, 
                 feature_size: int=288,
                 eta: float=1.0,
                 beta_icm: float = 0.2,
                 lambda_icm: float = 0.1,
                 gamma: float=0.99,
                 episodes: int=1000,
                 log_freq: int=100,
                 save_freq: int=100,
                 n_steps: int=2000,
                 actor_lr: float=0.0001,
                 critic_lr: float=0.0001,
                 ent_coef: float=0.01,
                 ou_noise: bool=True,
                 adaptive: bool=False,
                 desired_distance: float=0.7,
                 scalar: float=0.5,
                 scalar_decay: float=0.99) -> None:
        """
        Initializes the configuration settings.

        Args:
            - skip_frame (int): Number of frames to skip. Default to 2.
            - stack (int): Number of frames to stack. Default to 4.
            - resize_shape (int): Size of the resized frame. Default to 42.
            - batch_size (int): Batch size for training. Default to 32.
            - exploration_rate (float): Exploration rate. Default to 1.
            - exploration_rate_decay (float): Decay value for the exploration rate. Default to 0.999.
            - exploration_rate_min (float): Minimum value for the exploration rate. Default to 0.1.
            - memory_size (int): Size of the buffer. Default to 10000.
            - burn_in (int): Number of experiences to burn in. Default to 100.
            - lr (float): Learning rate. Default to 0.0001.
            - update_freq (int): Frequency of updating the online network. Default to 3.
            - sync_freq (int): Frequency of updating the target network. Default to 1000.
            - alpha (float): Priority exponent. Default to 0.6.
            - beta (float): Importance sampling exponent. Default to 0.4.
            - epsilon_buffer (float): Small constant to avoid zero priority. Default to 1e-8.
            - feature_size (int): Size of the feature embedding (ICM). Default to 288.
            - eta (float): Scaling factor for the intrinsic reward (ICM). Default to 1.0.
            - beta_icm (float): Weights the importance of the forward loss against the inverse loss (ICM). Default to 0.5.
            - lambda_icm (float): Discount factor for the intrinsic reward (ICM). Default to 0.5.
            - gamma (float): Discount factor. Default to 0.99.
            - episodes (int): Number of episodes to train. Default to 2000.
            - log_freq (int): Log frequency. Default to 100.
            - save_freq (int): Save frequency. Default to 100. 
            - n_steps (int): Number of steps for rollouts (A2C). Default to 2000.
            - actor_lr (float): Learning rate for the actor (A2C). Default to 0.0001.
            - critic_lr (float): Learning rate for the critic (A2C). Default to 0.0001.
            - ent_coef (float): Entropy coefficient for A2C. Default to 0.01.
            - ou_noise (bool): Whether to use Ornstein-Uhlenbeck noise for exploration (A2C). Default to True.
            - adaptive (bool): Whether to use adaptive OU noise (A2C). Default to False.
            - desired_distance (float): Desired distance for adaptive OU noise (A2C). Default to 0.7.
            - scalar (float): Scalar for adaptive OU noise (A2C). Default to 0.5.
            - scalar_decay (float): Decay value for the scalar (A2C). Default to 0.99.
            
        """
        # env params
        self.skip_frame = skip_frame
        self.stack = stack
        self.resize_shape = resize_shape
        # ddqn params
        self.batch_size = batch_size
        self.exploration_rate = exploration_rate
        self.exploration_rate_decay = exploration_rate_decay
        self.exploration_rate_min = exploration_rate_min
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.lr = lr
        self.update_freq = update_freq
        self.sync_freq = sync_freq
        # ddqn prioritized buffer params
        self.alpha = alpha
        self.beta = beta
        self.epsilon_buffer = epsilon_buffer
        # icm params
        self.feature_size = feature_size
        self.eta = eta
        self.beta_icm = beta_icm
        self.lambda_icm = lambda_icm
        # common params
        self.gamma = gamma
        self.episodes = episodes
        self.log_freq = log_freq   
        self.save_freq = save_freq
        # a2c params
        self.n_steps = n_steps
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.ent_coef = ent_coef
        self.ou_noise = ou_noise
        self.adaptive = adaptive
        self.desired_distance = 0.7
        self.scalar = scalar
        self.scalar_decay = scalar_decay
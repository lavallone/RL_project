from dataclasses import dataclass

@dataclass
class Hparams:
    seed: int = 99 # random seed
    
    #n_frames: int = 8 # number of frames to stack
    train_episodes: int = 200 # training episodes
    eval_episodes: int = 5 # number of evaluation episodes during training
    
    ## TD tabular (sarsa and q-learning)
    td_train_steps: int = 100000
    td_alpha: float = 0.2
    td_gamma: float = 0.99
    td_lambda_: float = 0.99
    td_min_eps: float = 0.01
    
    ## DDQN
    ddqn_train_steps: int = 1000000
    ddqn_gamma: float = 0.99
    ddqn_min_eps: float = 0.01
    ddqn_network_update_frequency: int = 5
    ddqn_network_sync_frequency: int = 100
    ddqn_batch_size: int = 512
    ddqn_lr: float = 1e-3
    ddqn_adam_eps: float = 1e-8
    ddqn_n_concat_states: int = 1
    
    
    ## PPO
    ppo_train_episodes: int = 1000
    ppo_n_epochs: int = 10 # number of learning epochs each episode
    ppo_n_rollout_steps: int = 2000 # rollout steps
    ppo_log_every: int = 5
    ppo_eval_episodes: int = 1
    # params
    ppo_gamma: float = 0.99 # used for GAE
    ppo_lambda_: float = 0.9 # used for GAE 
    ppo_clip_epsilon: float = 0.2 # eps for loss clipping
    ppo_c_1: float = 0.5 # weight for value function loss 
    ppo_c_2: float = 0.01 # weight for entropy loss
    # learning part
    ppo_batch_size: int = 256 # size of training batches
    ppo_lr: float = 1e-4 # adam optimizer learning rate
    ppo_adam_eps: float = 1e-8 # epsilon for adam
    ppo_n_concat_states: int = 1
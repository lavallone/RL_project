from dataclasses import dataclass

@dataclass
class Hparams:
    seed: int = 99 # random seed
    
    log_every: int = 10
    eval_episodes: int = 1
    
    ## TD tabular (sarsa and q-learning)
    td_train_steps: int = 1000000
    td_alpha: float = 0.2
    td_gamma: float = 0.99
    td_lambda_: float = 0.99
    td_min_eps: float = 0.01
    
    ## DDQN
    dqn_train_steps: int = 1000000
    dqn_gamma: float = 0.99
    dqn_min_eps: float = 0.01
    dqn_network_update_frequency: int = 5
    dqn_network_sync_frequency: int = 100
    dqn_batch_size: int = 512
    dqn_lr: float = 1e-3
    dqn_adam_eps: float = 1e-8
    dqn_n_concat_states: int = 1
    
    ## PPO
    ppo_train_episodes: int = 100
    ppo_n_epochs: int = 20 # number of learning epochs each episode
    ppo_n_rollout_steps: int = 1000 # rollout steps
    # params
    ppo_gamma: float = 0.99 # used for GAE
    ppo_lambda_: float = 0.95 # used for GAE 
    ppo_clip_epsilon: float = 0.1#0.2 # eps for loss clipping
    ppo_c_1: float = 1#0.5 # weight for value function loss 
    ppo_c_2: float = 0.01 #0.01 # weight for entropy loss
    # learning part
    ppo_batch_size: int = 64 # size of training batches
    ppo_lr: float = 1e-4 # adam optimizer learning rate
    ppo_adam_eps: float = 1e-8 # epsilon for adam
    ppo_n_concat_states: int = 1
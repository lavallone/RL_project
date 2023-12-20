import shutil
from pathlib import Path
import numpy as np
import wandb
import random
from gym.wrappers import Monitor
import time
from gym_breakout_pygame.breakout_env import BreakoutConfiguration

from src.utils.env import make_env
from src.agents.tabular import TabularAgent
from src.agents.dqn import DDQNAgent
from src.agents.ppo import PPOAgent

def run_agent(args, hparams):
    # SETTING-UP 
    if args.is_wandb == "wandb":
        wandb.login()
        wandb_run_name = args.is_rb+"/"+args.algorithm+"_"+args.action_type+"_"+str(args.rows)+"x"+str(args.cols)+"_"+args.rb_type
    output_dir = Path( "runs/"+args.is_rb+"/"+args.algorithm+"_"+args.action_type+"_"+str(args.rows)+"x"+str(args.cols)+"_"+args.rb_type )
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=False)

    # SETTING ENVIRONMENT 
    ball_enabled, fire_enabled = False, False
    if args.action_type == "fire":  fire_enabled = True
    elif args.action_type == "fire_ball":
        fire_enabled = True
        ball_enabled = True
    elif args.action_type == "ball": ball_enabled = True
    
    targets = []
    if args.rb_type == "2targets" and args.is_rb == "rb":
        t0 = (random.randint(0, args.cols-1), 0)
        t1 = (random.randint(0, args.cols-1), args.rows-1)
        targets = [t1, t0]
        print(f"Targets are {targets}")
    elif args.rb_type == "3targets" and args.is_rb == "rb":
        t0 = (random.randint(0, args.cols-1), args.rows-1)
        t1 = (random.randint(0, args.cols-1), args.rows-2)
        t2 = (random.randint(0, args.cols-1), 0)
        #targets = [t0, t1, t2]
        targets = [t2, t1, t0]
        print(f"Targets are {targets}")

    # core environment configuration
    config = BreakoutConfiguration(brick_rows=args.rows, brick_cols=args.cols,
                                   brick_reward=args.brick_reward, step_reward=args.step_reward,
                                   ball_enabled=ball_enabled, fire_enabled=fire_enabled, targets=targets)
    # here we may add or may not the restraining bolt
    env = make_env(config, output_dir, args.goal_reward, args.action_type, args.rb_type, restraining_bolt=args.is_rb, targets=targets)
    env.seed(hparams.seed)
    
    # SETTING AGENT
    if args.algorithm == "sarsa" or args.algorithm == "q":
        agent = TabularAgent(env, td_type=args.algorithm, alpha=hparams.td_alpha, gamma=hparams.td_gamma, lambda_= hparams.td_lambda_, min_eps = hparams.td_min_eps, train_steps=hparams.td_train_steps)
    elif args.algorithm == "dqn" or args.algorithm == "ddqn":
        agent = DDQNAgent(env, output_dir=output_dir, type_=args.algorithm, train_steps=hparams.dqn_train_steps, log_every=hparams.dqn_log_every, eval_episodes=hparams.dqn_eval_episodes, gamma=hparams.dqn_gamma, min_eps=hparams.dqn_min_eps, network_update_frequency=hparams.dqn_network_update_frequency, \
                          network_sync_frequency=hparams.dqn_network_sync_frequency, batch_size=hparams.dqn_batch_size, lr=hparams.dqn_lr, adam_eps=hparams.dqn_adam_eps, n_concat_states = hparams.dqn_n_concat_states)
    elif args.algorithm == "ppo":
        agent = PPOAgent(env, output_dir=output_dir, train_episodes=hparams.ppo_train_episodes, n_epochs=hparams.ppo_n_epochs, n_rollout_steps=hparams.ppo_n_rollout_steps, log_every=hparams.ppo_log_every, eval_episodes=hparams.ppo_eval_episodes, gamma=hparams.ppo_gamma, lambda_=hparams.ppo_lambda_, \
                         clip_epsilon=hparams.ppo_clip_epsilon, c_1=hparams.ppo_c_1, c_2=hparams.ppo_c_2, batch_size=hparams.ppo_batch_size, lr=hparams.ppo_lr, adam_eps=hparams.ppo_adam_eps, n_concat_states = hparams.ppo_n_concat_states)
    # train
    if args.is_wandb == "wandb":
        with wandb.init(entity="lavallone", project="RL", name=wandb_run_name, mode="offline"):
            agent.train()
        wandb.finish()
    else: 
        agent.train()
    # test
    agent.test(Monitor(env, output_dir / "videos"), n_episodes=hparams.eval_episodes, visualize=True)
    
    env.close()
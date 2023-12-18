import shutil
from pathlib import Path
import numpy as np
import wandb
import random
from gym.wrappers import Monitor
from gym_breakout_pygame.breakout_env import BreakoutConfiguration

from src.utils.env import make_env
from src.agents.tabular import TabularAgent
from src.agents.ddqn import DDQNAgent
from src.agents.ppo import PPOAgent

# from old.brains import Sarsa, QLearning
# from old.callbacks import ModelCheckpoint
# from old.core import TrainEpisodeLogger, Agent
# from old.policies import EpsGreedyQPolicy, AutomataPolicy, LinearAnnealedPolicy


def run_agent(args, hparams):
    # SETTING-UP 
    #wandb.login()
    #run_name = "RB_"+args.algorithm+"_"+args.action_type+"_"+str(args.rows)+"x"+str(args.cols)+"_"+str(hparams.train_steps)+"_"+args.rb_type
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

    # core environment configuration (qui posso lavorare per fare i mattoncini di colori diversi)
    config = BreakoutConfiguration(brick_rows=args.rows, brick_cols=args.cols,
                                   brick_reward=args.brick_reward, step_reward=args.step_reward,
                                   ball_enabled=ball_enabled, fire_enabled=fire_enabled, targets=targets)
    # here we may add or may not the restraining bolt
    env = make_env(config, output_dir, args.goal_reward, args.action_type, args.rb_type, restraining_bolt=args.is_rb, targets=targets)
    env.seed(hparams.seed)
    
    # SETTING AGENT
    if args.algorithm == "sarsa" or args.algorithm == "q":
        agent = TabularAgent(env, td_type=args.algorithm, alpha=hparams.td_alpha, gamma=hparams.td_gamma, lambda_= hparams.td_lambda_, min_eps = hparams.td_min_eps, train_steps=hparams.td_train_steps)
    elif args.algorithm == "ddqn":
        agent = DDQNAgent(env, train_steps=hparams.ddqn_train_steps, gamma=hparams.ddqn_gamma, min_eps=hparams.ddqn_min_eps, network_update_frequency=hparams.ddqn_network_update_frequency, \
                          network_sync_frequency=hparams.ddqn_network_sync_frequency, batch_size=hparams.ddqn_batch_size, lr=hparams.ddqn_lr, adam_eps=hparams.ddqn_adam_eps, n_concat_states = hparams.ddqn_n_concat_states)
    elif args.algorithm == "ppo":
        agent = PPOAgent(env, train_episodes=hparams.ppo_train_episodes, n_epochs=hparams.ppo_n_epochs, n_rollout_steps=hparams.ppo_n_rollout_steps, log_every=hparams.ppo_log_every, eval_episodes=hparams.eval_episodes, gamma=hparams.ppo_gamma, lambda_=hparams.ppo_lambda_, \
                         clip_epsilon=hparams.ppo_clip_epsilon, c_1=hparams.ppo_c_1, c_2=hparams.ppo_c_2, batch_size=hparams.ppo_batch_size, lr=hparams.ppo_lr, adam_eps=hparams.ppo_adam_eps, n_concat_states = hparams.ppo_n_concat_states)
    # train
    agent.train()
    # test
    agent.test(Monitor(env, output_dir / "videos"), n_episodes=hparams.eval_episodes, visualize=True)

    # if arguments.is_rb == "rb": policy = AutomataPolicy((-2, ), nb_steps=arguments.train_steps/10, value_max=1.0, value_min=configuration.min_eps)
    # elif arguments.is_rb == "no_rb": policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.0, value_min=configuration.min_eps, value_test=.0, nb_steps=arguments.train_steps/10)
    # algorithm = Sarsa if arguments.algorithm == "sarsa" else QLearning
    # agent = Agent(algorithm(None,
    #                     env.action_space,
    #                     gamma=configuration.gamma,
    #                     alpha=configuration.alpha,
    #                     lambda_=configuration.lambda_),
    #               policy=policy, #the agent is set to the choosen policy
    #               test_policy=EpsGreedyQPolicy(eps=0.01))
    
    # #with wandb.init(entity="lavallone", project="Restraining Bolts", name=run_name, mode="offline"):
    # # here it starts the learning
    # _ = agent.fit(
    #     env,
    #     nb_steps=arguments.train_steps,
    #     visualize=configuration.visualize_training,
    #     callbacks=[
    #         ModelCheckpoint(str(agent_dir / "checkpoints" / "agent-{}.pkl")),
    #         TrainEpisodeLogger()
    #     ]
    # )
    # #wandb.finish()
    
    env.close()
    return output_dir
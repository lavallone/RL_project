#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This is the main entry-point for the experiments with the Breakout environment."""
import logging
import random
import numpy as np
import torch
from argparse import ArgumentParser

from src.run import run_agent
from src.hyperparameters import Hparams

logging.getLogger("temprl").setLevel(level=logging.ERROR)
logging.getLogger("matplotlib").setLevel(level=logging.ERROR)
logging.getLogger("rl_algorithm").setLevel(level=logging.ERROR)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def parse_args():
    parser = ArgumentParser()
    # experiments parameters
    parser.add_argument("--cols", type=int, default=3, help="Number of columns.", required=True)
    parser.add_argument("--rows", type=int, default=3, help="Number of rows.", required=True)
    parser.add_argument("--is_rb", type=str, default="rb", help="If the Restrainnig Bolt is applied or not to the agent ('no_rb' or 'rb').", required=True)
    parser.add_argument("--action_type", type=str, default="fire", help="Allowed actions for the agent. ('fire', 'fire_ball' or 'ball').", required=True)
    # this can become the type of applied restraining bolt
    parser.add_argument("--rb_type", type=str, default="sx2dx", help="Type of restraining bolt('sx2dx', 'dx2sx', 'down2up', 'up2down', '2targets' and '3targets').", required=True)
    # tabular q-learning lambda, DQN or PPO
    parser.add_argument("--algorithm", type=str, default="sarsa", help="RL algorithm ('sarsa', 'q', 'ddqn' or 'ppo').", required=True)
    
    parser.add_argument("--brick-reward", type=int, default=5, help="The reward for breaking a brick.")
    parser.add_argument("--step-reward", type=float, default=-0.01, help="The reward (cost) when nothing happens.")
    parser.add_argument("--goal-reward", type=int, default=1000, help="The reward for satisfying the temporal goal.")
    return parser.parse_args()


def main(args):
    
    hparams = Hparams()
    set_seed(hparams.seed)
    print("\n>>>>>>>>>>>>>>>>> Starting agent <<<<<<<<<<<<<<<<<\n")
    run_agent(args, hparams)


if __name__ == '__main__':
    args = parse_args()
    main(args)
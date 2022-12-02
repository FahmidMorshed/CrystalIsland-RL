import dataclasses
import logging
import os.path
from collections import deque
from copy import deepcopy

import torch
import numpy as np
import pandas as pd
from torch import FloatTensor

from src import utils, evaluation
from src.env.crystalisland import CrystalIsland
import src.env.constants as envconst
from src.model.gail import GAIL
from src.model.rl import BCQ, RANDOM
from src.model.rl import FQE
from src.model_args import ModelArguments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

external_config = {
    'dryrun': False,
    # 'simulate_episodes': 1000,
    'run_name': 'dec2',
    # 'train_steps': 1000,
    # 'update_frequency': 100,
    # 'log_frequency': 100,
    # 'gail_train_steps': 100,
}

def run():
    args = utils.parse_config(ModelArguments, external_config)
    args.seed = 1
    utils.set_all_seeds(args.seed)
    args.run_name += "_" + str(args.seed)

    train, test = utils.load_data(args.student_data_loc, test_size=.3)

    clf = utils.reward_predictor(train, args.seed, print_eval=False)
    action_probs = utils.get_action_probs()

    env = CrystalIsland(solution_predictor=clf, action_probs=action_probs)
    gail = GAIL(args, train, test, env)
    gail.train()
    gail.simulate(1000)
    print()


def dummy():
    args = utils.parse_config(ModelArguments, external_config)
    args.seed = 1
    utils.set_all_seeds(args.seed)
    args.run_name += "_" + str(args.seed)

    train, test = utils.load_data(args.student_data_loc, test_size=.3)

    clf = utils.reward_predictor(train, args.seed, print_eval=False)
    action_probs = utils.get_action_probs()
    env = CrystalIsland(solution_predictor=clf, action_probs=action_probs)
    state = env.reset()
    for i in range(1000):
        action = np.random.choice(range(20))
        next_state, reward, done, info = env.step(action)

        state = next_state
        if done:
            print(state, action, reward, info)
            state = env.reset()


if __name__ == "__main__":
    run()

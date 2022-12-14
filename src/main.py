import dataclasses
import logging
import os.path
from collections import deque, Counter
from copy import deepcopy

import torch
import numpy as np
import pandas as pd
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from matplotlib import pyplot as plt
from pyod.models.auto_encoder_torch import AutoEncoder
from pyod.utils.example import visualize
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from stable_baselines3 import PPO
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from torch import FloatTensor
import imitation

from src import utils, evaluation
from src.env.crystalisland import CrystalIsland
import src.env.constants as envconst
from src.model import policy
from src.model.gail import GAIL
# from src.model.gail import GAIL
from src.model.rl import BCQ, RANDOM
from src.model.rl import FQE
from src.model_args import ModelArguments
import pyod
from pyod.models.knn import KNN
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.metrics import classification_report, f1_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

external_config = {
    'dryrun': False,
    'run_name': 'dec14',
}

def run():

    args = utils.parse_config(ModelArguments, external_config)
    args.seed = 1
    utils.set_all_seeds(args.seed)
    args.run_name += "_" + str(args.seed)
    train, test = utils.load_data_by_solved(args.student_data_loc, test_size=.2)

    # clf = utils.solve_predictor(args.student_data_loc, args.seed, print_eval=True)
    env = CrystalIsland(solution_predictor=None)

    print("\n=====RUNNING RANDOM=====\n")
    rp = policy.RandomPolicy(args, train, test, env)
    rp.train()
    rp.eval_score(0)

    print("\n=====RUNNING ACTION PRIOR=====\n")
    ap = policy.ActionPriorPolicy(args, train, test, env)
    ap.train()
    ap.eval_score(0)

    print("\n=====RUNNING BEHAVIOR CLONING DT=====\n")
    bcdt = policy.BehaviorCloning(args, train, test, env)
    bcdt.train()
    bcdt.eval_score(0)

    print("\n=====RUNNING BEHAVIOR CLONING MLP=====\n")
    clf = MLPClassifier(random_state=args.seed, hidden_layer_sizes=(128, 128), max_iter=10000, shuffle=True)
    bcnn = policy.BehaviorCloning(args, train, test, env, clf=clf, name='bcnn')
    bcnn.train()
    bcnn.eval_score(0)

    print("\n=====RUNNING GAIL=====\n")
    gail = GAIL(args, train, test, env)
    gail.train(200)
    gail.eval_score(0)


if __name__ == "__main__":
    run()

import dataclasses
import logging
import os.path
from collections import deque, Counter
from copy import deepcopy

import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyod.models.auto_encoder_torch import AutoEncoder
from pyod.utils.example import visualize
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from torch import FloatTensor

from src import utils, evaluation
from src.env.crystalisland import CrystalIsland
import src.env.constants as envconst
from src.model import policy
from src.model.gail import GAIL
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
    'run_name': 'dec7',
}

def run():
    args = utils.parse_config(ModelArguments, external_config)
    args.seed = 7
    utils.set_all_seeds(args.seed)
    args.run_name += "_" + str(args.seed)
    train, test = utils.load_data(args.student_data_loc, test_size=.2)

    anomaly_detector = utils.get_anomaly_detector(train, test)

    action_probs = utils.get_action_probs()
    clf = utils.reward_predictor(train, args.seed, print_eval=True)
    env = CrystalIsland(solution_predictor=clf, action_probs=action_probs, anomaly_detector=anomaly_detector)

    gail = GAIL(args, train, test, env)
    gail.train(500)
    gail.eval_score(0)

    rp = policy.RandomPolicy(args, train, test, env)
    rp.train()
    rp.eval_score(0)

    bp = policy.BehaviorPolicy(args, train, test, env)
    bp.train()
    bp.eval_score(0)

    bc_rf = policy.BehaviorCloning(args, train, test, env)
    bc_rf.train()
    bc_rf.eval_score(0)

    bc_dt = policy.BehaviorCloning(args, train, test, env, clf=DecisionTreeClassifier())
    bc_dt.train()
    bc_dt.eval_score(0)


if __name__ == "__main__":

    run()

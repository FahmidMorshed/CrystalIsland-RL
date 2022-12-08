import dataclasses
import logging
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from src import evaluation, utils

logger = logging.getLogger(__name__)
class Policy:
    def __init__(self,
                 args: dataclasses,
                 train_df: pd.DataFrame,
                 test_df: pd.DataFrame,
                 env,
                 name):
        self.args = args
        self.train_df = train_df
        self.test_df = test_df
        self.env = env
        self.name = name

    def get_action(self, state: np.ndarray):
        probs = self.get_probs(state)
        return np.random.choice(range(self.args.action_dim), p=probs)

    def get_probs(self, state: np.ndarray):
        raise NotImplementedError

    def act_log_prob(self, state, action):
        probs = self.get_probs(state)
        prob = probs[action] if probs[action]>0.00001 else 0.00001
        return np.log(prob)

    def train(self):
        return
    def eval_score(self, epoch):
        df = utils.simulate_env(self, total_episode=100)
        rewards, actions = evaluation.avg_rewards_actions(self, df)
        perp = evaluation.perplexity(self)
        kld = evaluation.kld(self)
        anomaly = evaluation.anomaly(self, df)
        logger.info("{6} || EPOCH {0} | Rewards: {1: .0f} | "
                    "Perplexity: {2: .4f} | KLD: {3: .4f} | Anomaly: {4: .1f}%| Actions: {5}"
                    .format(epoch, rewards, perp, kld, anomaly, actions, self.name))

class RandomPolicy(Policy):
    def __init__(self, args: dataclasses, train_df: pd.DataFrame, test_df: pd.DataFrame, env, name="random"):
        super().__init__(args, train_df, test_df, env, name)

        self.probs = [1.0 / args.action_dim, ] * args.action_dim

    def get_probs(self, state: np.ndarray):
        return self.probs

class BehaviorPolicy(Policy):
    def __init__(self, args: dataclasses, train_df: pd.DataFrame, test_df: pd.DataFrame, env, name="behavior_prob"):
        super().__init__(args, train_df, test_df, env, name)
        self.probs = list(self.train_df.groupby('action').count()['step'] / len(self.train_df))

    def get_probs(self, state: np.ndarray):
        return self.probs


class BehaviorCloning(Policy):
    def __init__(self, args: dataclasses, train_df: pd.DataFrame, test_df: pd.DataFrame, env, name="behavior_rf", clf=None):
        super().__init__(args, train_df, test_df, env, name)

        if clf is None:
            self.clf = RandomForestClassifier()
        else:
            self.clf = clf

    def get_probs(self, state: np.ndarray):
        probs = list(self.clf.predict_proba([state])[0])
        return probs

    def train(self):
        X = np.stack(self.train_df['state'])
        y = np.array(self.train_df['action'])
        self.clf.fit(X, y)

        X_test = np.stack(self.test_df['state'])
        y_test = np.array(self.test_df['action'])
        y_pred = self.clf.predict(X_test)
        # logger.info(classification_report(y_test, y_pred))

        return

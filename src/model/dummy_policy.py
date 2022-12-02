import dataclasses
from copy import deepcopy

import numpy as np
import pandas as pd


class RandomPolicy:
    def __init__(self,
                 args: dataclasses,
                 train_df: pd.DataFrame,
                 test_df: pd.DataFrame,
                 env):
        self.args = args
        self.train_df = train_df
        self.test_df = test_df
        self.env = env
        self.probs = [1.0 / args.action_dim, ] * args.action_dim

    def get_action(self, state: np.ndarray):
        return np.random.choice(range(self.args.action_dim))

    def get_probs(self, state: np.ndarray):
        return self.probs

    def act_log_prob(self, state, action):
        probs = self.get_probs(state)
        return np.log(probs[action])


class BehaviorPolicy:
    def __init__(self,
                 args: dataclasses,
                 train_df: pd.DataFrame,
                 test_df: pd.DataFrame,
                 env):
        self.args = args
        self.train_df = train_df
        self.test_df = test_df
        self.env = env
        self.probs = list(self.train_df.groupby('action').count()['step'] / len(self.train_df))

    def get_action(self, state: np.ndarray):
        return np.random.choice(range(self.args.action_dim), p=self.probs)

    def get_probs(self, state: np.ndarray):
        return self.probs

    def act_log_prob(self, state, action):
        probs = self.get_probs(state)
        return np.log(probs[action])


class BehaviorPolicy2:
    def __init__(self,
                 args: dataclasses,
                 train_df: pd.DataFrame,
                 test_df: pd.DataFrame,
                 env):
        self.args = args
        self.train_df = train_df
        self.test_df = test_df
        self.env = env
        self.probs = self.train_df.apply(lambda x: {tuple(x['state']): list(x['act_prob'])}, axis=1)

    def get_action(self, state: np.ndarray):
        probs = self.get_probs(state)
        return np.random.choice(range(self.args.action_dim), p=probs)

    def get_probs(self, state: np.ndarray):
        probs = self.probs.get(tuple(state), [1.0 / self.args.action_dim, ] * self.args.action_dim)
        return probs

    def act_log_prob(self, state, action):
        probs = self.get_probs(state)
        return np.log(probs[action])

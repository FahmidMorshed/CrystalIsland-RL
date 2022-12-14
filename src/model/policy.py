import dataclasses
import logging
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

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
        raise NotImplementedError

    def get_all_probs(self, all_states: np.ndarray):
        raise NotImplementedError

    def get_all_act_log_probs(self, all_states, all_actions):
        all_probs = self.get_all_probs(all_states)
        all_act_probs = np.take_along_axis(all_probs, np.expand_dims(all_actions, axis=1), axis=1).squeeze()
        return np.log(all_act_probs)

    def train(self):
        return

    def eval_score(self, epoch):
        rewards, actions = evaluation.avg_rewards_actions(self)
        perp = evaluation.perplexity(self)
        kld_pq, kld_qp, jsd = evaluation.divergence(self)
        detector, anomaly_f1w = evaluation.anomaly(self)
        clf, clf_f1w = evaluation.classify(self)
        logger.info("{6} || EPOCH {0} | Rewards: {1: .0f} | "
                    "Perplexity: {2: .4f} | Div (pq qp js): {3: .2f} {7: .2f} {8: .2f} | Anomaly: {4: .1f}% | "
                    "Classify: {9: .2f}% | Actions: {5}"
                    .format(epoch, rewards, perp, kld_pq, anomaly_f1w, actions, self.name[:4], kld_qp, jsd, clf_f1w))

class RandomPolicy(Policy):
    def __init__(self, args: dataclasses, train_df: pd.DataFrame, test_df: pd.DataFrame, env, name="random"):
        super().__init__(args, train_df, test_df, env, name)

        self.probs = [1.0 / args.action_dim, ] * args.action_dim

    def get_action(self, state: np.ndarray):
        return np.random.choice(range(self.args.action_dim), p=self.probs)

    def get_all_probs(self, all_states: np.ndarray):
        all_probs = np.stack([self.probs] * len(all_states))
        return all_probs



class ActionPriorPolicy(Policy):
    def __init__(self, args: dataclasses, train_df: pd.DataFrame, test_df: pd.DataFrame, env, name="actprior"):
        super().__init__(args, train_df, test_df, env, name)
        self.probs = self.env.envconst.action_probs

    def get_action(self, state: np.ndarray):
        return np.random.choice(range(self.args.action_dim), p=self.probs)

    def get_all_probs(self, all_states: np.ndarray):
        all_probs = np.stack([self.probs] * len(all_states))
        return all_probs


class BehaviorCloning(Policy):
    def __init__(self, args: dataclasses, train_df: pd.DataFrame, test_df: pd.DataFrame, env, name="bcdt", clf=None):
        super().__init__(args, train_df, test_df, env, name)

        if clf is None:
            self.clf = DecisionTreeClassifier()
        else:
            self.clf = clf

    def get_action(self, state: np.ndarray):
        return self.clf.predict([state])[0]

    def get_all_probs(self, all_states: np.ndarray):
        all_probs = self.clf.predict_proba(all_states)
        all_probs[all_probs == 0.0] = 0.0000001
        return all_probs

    def train(self):
        X = np.stack(self.train_df['state'])
        y = np.array(self.train_df['action'])
        self.clf.fit(X, y)

        X_test = np.stack(self.test_df['state'])
        y_test = np.array(self.test_df['action'])
        y_pred = self.clf.predict(X_test)
        # logger.info(classification_report(y_test, y_pred, zero_division=0))

        return

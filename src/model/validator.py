import dataclasses
import logging
from collections import deque

import numpy as np
import pandas as pd
import torch
from torch import nn

from src import utils
from src.model import nets
from sklearn import metrics

from src.model.crystalisland import CrystalIsland

logger = logging.getLogger(__name__)


# TODO maybe include heuristic based data along with random data
class SimpleValidator:
    def __init__(self, args: dataclasses, train_df: pd.DataFrame, test_df: pd.DataFrame, s0: np.ndarray, name="filter"):
        self.args = args

        self.train_df = train_df
        self.test_df = test_df
        self.s0 = s0
        self.train_state_actions, _, _ = utils.state_action_tensor(np.stack(self.train_df['state']),
                                                                   np.array(self.train_df['action']),
                                                                   self.args.action_dim)
        self.test_state_actions, _, _ = utils.state_action_tensor(np.stack(self.test_df['state']),
                                                                  np.array(self.test_df['action']),
                                                                  self.args.action_dim)

        self.env = CrystalIsland(self.args, self.s0)

        self.model_nlg = nets.Discriminator(args)
        self.model_auth = nets.Discriminator(args)

        self.optimizer_nlg = torch.optim.Adam(self.model_nlg.parameters(), lr=self.args.lr)

        self.optimizer_auth = torch.optim.Adam(self.model_auth.parameters(), lr=self.args.lr)

        self.loss = nn.MSELoss()

        self.name = name

    def validate_episode(self, states: np.ndarray, actions: np.ndarray):
        is_authentic = False
        state_actions, _, _ = utils.state_action_tensor(states, actions, self.args.action_dim)

        is_auth = self.model_auth(state_actions)
        is_auth = sum((is_auth > 0.5).float()) / len(is_auth)  # 1 is for authentic
        if is_auth > self.args.validator_auth_threshold:
            is_authentic = True

        end_state_actions = state_actions[-1]
        is_high = self.model_nlg(end_state_actions)
        is_high = (is_high > 0.5).item()

        return is_authentic, is_high

    def validate_df(self, df):
        result = []
        for student_id, episode_df in df.groupby('student_id'):
            states = np.stack(episode_df['state'])
            actions = np.array(episode_df['action'])
            is_authentic, is_high = self.validate_episode(states, actions)
            result.append({'student_id': student_id, 'is_authentic': is_authentic, 'is_high': is_high})

        result_df = pd.DataFrame(result, columns=['student_id', 'is_authentic', 'is_high'])
        return result_df

    def _train_nlg(self):
        train_ids = np.array(self.train_df.loc[self.train_df['done']].index)
        test_ids = np.array(self.test_df.loc[self.test_df['done']].index)
        train_X = self.train_state_actions[train_ids]
        test_X = self.test_state_actions[test_ids]

        train_nlg = np.array(self.train_df['reward'])
        train_y = train_nlg[train_ids]
        if set(train_y) != {-100.0, 100.0}:
            logger.error("train_y has do not have different reward values")
        train_y = torch.from_numpy((train_y == 100)).float()
        test_nlg = np.array(self.test_df['reward'])
        test_y = test_nlg[test_ids]
        if set(test_y) != {-100.0, 100.0}:
            logger.error("test_y has do not have different reward values")
        test_y = torch.from_numpy((test_y == 100)).float()

        curr_loss = 0
        prev_loss = 999.0
        prev_diffs = deque(maxlen=100)
        for i in range(100):
            prev_diffs.append(9999.0)
        self.model_nlg.train()
        for ep in range(self.args.train_steps):
            idx = np.random.choice(range(len(train_y)), 128)
            train_X_batch = train_X[idx]
            train_y_batch = train_y[idx]
            pred_y = self.model_nlg(train_X_batch)
            pred_y = pred_y.squeeze()
            loss = self.loss(pred_y, train_y_batch)

            curr_loss = loss.item()

            self.optimizer_nlg.zero_grad()
            loss.backward()
            self.optimizer_nlg.step()

            logger.info("reward training epoch {0} | current loss {1: .8f}".format(ep, curr_loss))
            if curr_loss < 0.01:
                logger.info("-- early converged | epoch {0}/{1} --".format(ep, self.args.train_steps))
                break

        logger.info("reward training complete | current loss {0: .4f}".format(curr_loss))
        self.model_nlg.eval()
        pred_y = self.model_nlg(test_X).detach()
        pred_y = pred_y.squeeze()
        pred_y = (pred_y > 0.5).float()
        print("-- validator reward test result --")
        print("confusion metrix\n", metrics.confusion_matrix(test_y, pred_y))
        print("accuracy:", metrics.accuracy_score(test_y, pred_y))
        print("report\n", metrics.classification_report(test_y, pred_y))

    def _train_auth(self):
        train_X_true = self.train_state_actions
        test_X_true = self.test_state_actions

        train_y_true = torch.ones((train_X_true.shape[0], 1), device=self.args.device)
        test_y_true = torch.ones((test_X_true.shape[0], 1), device=self.args.device)

        rand_df = self.env.gen_random_data(len(train_y_true))
        train_X_rand, _, _ = utils.state_action_tensor(np.stack(rand_df['state']),
                                                       np.array(rand_df['action']),
                                                       self.args.action_dim)
        train_y_rand = torch.zeros((train_X_rand.shape[0], 1), device=self.args.device)

        train_X = torch.cat([train_X_true, train_X_rand], dim=0)
        train_y = torch.cat([train_y_true, train_y_rand], dim=0)

        rand_df = self.env.gen_random_data(len(test_y_true))
        test_X_rand, _, _ = utils.state_action_tensor(np.stack(rand_df['state']),
                                                      np.array(rand_df['action']),
                                                      self.args.action_dim)
        test_y_rand = torch.zeros((test_X_rand.shape[0], 1), device=self.args.device)

        test_X = torch.cat([test_X_true, test_X_rand], dim=0)
        test_y = torch.cat([test_y_true, test_y_rand], dim=0)

        curr_loss = 0.0
        prev_loss = 9999.0
        prev_diffs = deque(maxlen=100)
        for i in range(100):
            prev_diffs.append(9999.0)
        self.model_auth.train()
        for ep in range(self.args.train_steps):
            idx = np.random.choice(range(len(train_y)), 128)
            train_X_batch = train_X[idx]
            train_y_batch = train_y[idx]

            pred_y = self.model_auth(train_X_batch)
            loss = self.loss(pred_y, train_y_batch)

            curr_loss = loss.item()

            self.optimizer_auth.zero_grad()
            loss.backward()
            self.optimizer_auth.step()

            logger.info("auth training epoch {0} | current loss {1: .8f}".format(ep, curr_loss))
            if curr_loss < 0.01:
                logger.info("-- early converged | epoch {0}/{1} --".format(ep, self.args.train_steps))
                break

        logger.info("auth training complete | current loss {0: .4f}".format(curr_loss))
        self.model_auth.eval()
        pred_y = self.model_auth(test_X).detach()
        pred_y = (pred_y > 0.5).float()
        print("-- validator reward test result --")
        print("confusion metrix\n", metrics.confusion_matrix(test_y, pred_y))
        print("accuracy:", metrics.accuracy_score(test_y, pred_y))
        print("report\n", metrics.classification_report(test_y, pred_y))

    def train(self, force_train=False):
        if force_train is False:
            is_loaded = self.load()
            if is_loaded:
                return

        logger.info("-- training validator --")
        self._train_nlg()
        self._train_auth()
        if self.args.dryrun is False:
            self.save()

    def save(self):
        torch.save(self.model_nlg.state_dict(), "../checkpoint/" + self.args.run_name + "_" + self.name + "_valid_nlg.ckpt")
        torch.save(self.model_auth.state_dict(), "../checkpoint/" + self.args.run_name + "_" + self.name + "_valid_auth.ckpt")
        logger.info("validator models saved!")

    def load(self):
        is_loaded = False
        try:
            self.model_nlg.load_state_dict(torch.load("../checkpoint/" + self.args.run_name + "_" + self.name + "_valid_nlg.ckpt",
                                                      map_location=lambda x, y: x))
            self.model_auth.load_state_dict(torch.load("../checkpoint/" + self.args.run_name + "_" + self.name + "_valid_auth.ckpt",
                                                       map_location=lambda x, y: x))
            logger.info("loaded validator with run_name {0}".format(self.args.run_name))
            is_loaded = True
        except FileNotFoundError:
            logger.info("no validator with run_name {0}".format(self.args.run_name))
        return is_loaded


class LSTMValidator:
    def __init__(self, args: dataclasses, train_df: pd.DataFrame, test_df: pd.DataFrame, s0: np.ndarray):
        self.args = args

        self.train_df = train_df
        self.test_df = test_df
        self.s0 = s0
        self.train_state_actions, _, _ = utils.state_action_tensor(np.stack(self.train_df['state']),
                                                                   np.array(self.train_df['action']),
                                                                   self.args.action_dim)
        self.test_state_actions, _, _ = utils.state_action_tensor(np.stack(self.test_df['state']),
                                                                  np.array(self.test_df['action']),
                                                                  self.args.action_dim)

        self.env = CrystalIsland(self.args, self.s0)

        self.model_nlg = nets.Discriminator(args)
        self.model_auth = nets.Discriminator(args)

        self.optimizer_nlg = torch.optim.Adam(self.model_nlg.parameters(), lr=self.args.lr)

        self.optimizer_auth = torch.optim.Adam(self.model_auth.parameters(), lr=self.args.lr)

        self.loss = nn.MSELoss()

    def validate_episode(self, states: np.ndarray, actions: np.ndarray):
        is_authentic = False
        state_actions, _, _ = utils.state_action_tensor(states, actions, self.args.action_dim)

        is_auth = self.model_auth(state_actions)
        is_auth = sum((is_auth > 0.5).float()) / len(is_auth)  # 1 is for authentic
        if is_auth > self.args.validator_auth_threshold:
            is_authentic = True

        end_state_actions = state_actions[-1]
        is_high = self.model_nlg(end_state_actions)
        is_high = (is_high > 0.5).item()

        return is_authentic, is_high

    def validate_df(self, df):
        result = []
        for student_id, episode_df in df.groupby('student_id'):
            states = np.stack(episode_df['state'])
            actions = np.array(episode_df['action'])
            is_authentic, is_high = self.validate_episode(states, actions)
            result.append({'student_id': student_id, 'is_authentic': is_authentic, 'is_high': is_high})

        result_df = pd.DataFrame(result, columns=['student_id', 'is_authentic', 'is_high'])
        return result_df

    def _train_nlg(self):
        train_ids = np.array(self.train_df.loc[self.train_df['done']].index)
        test_ids = np.array(self.test_df.loc[self.test_df['done']].index)
        train_X = self.train_state_actions[train_ids]
        test_X = self.test_state_actions[test_ids]

        train_nlg = np.array(self.train_df['reward'])
        train_y = train_nlg[train_ids]
        if set(train_y) != {-100.0, 100.0}:
            logger.error("train_y has do not have different reward values")
        train_y = torch.from_numpy((train_y == 100)).float()
        test_nlg = np.array(self.test_df['reward'])
        test_y = test_nlg[test_ids]
        if set(test_y) != {-100.0, 100.0}:
            logger.error("test_y has do not have different reward values")
        test_y = torch.from_numpy((test_y == 100)).float()

        curr_loss = 0
        early_stop = 0
        self.model_nlg.train()
        for ep in range(self.args.train_steps):
            pred_y = self.model_nlg(train_X)
            pred_y = pred_y.squeeze()
            loss = self.loss(pred_y, train_y)

            if loss.item() == curr_loss or curr_loss < 0.01:
                early_stop += 1
            else:
                early_stop = 0
            curr_loss = loss.item()

            self.optimizer_nlg.zero_grad()
            loss.backward()
            self.optimizer_nlg.step()

            logger.info("reward training epoch {0} | current loss {1: .8f}".format(ep, curr_loss))
            if early_stop > 10:
                break

        logger.info("reward training complete | current loss {0: .4f}".format(curr_loss))
        self.model_nlg.eval()
        pred_y = self.model_nlg(test_X).detach()
        pred_y = pred_y.squeeze()
        pred_y = (pred_y > 0.5).float()
        print("-- validator reward test result --")
        print("confusion metrix\n", metrics.confusion_matrix(test_y, pred_y))
        print("accuracy:", metrics.accuracy_score(test_y, pred_y))
        print("report\n", metrics.classification_report(test_y, pred_y))

    def _train_auth(self):
        train_X_true = self.train_state_actions
        test_X_true = self.test_state_actions

        train_y_true = torch.ones((train_X_true.shape[0], 1), device=self.args.device)
        test_y_true = torch.ones((test_X_true.shape[0], 1), device=self.args.device)

        rand_df = self.env.gen_random_data(len(train_y_true))
        train_X_rand, _, _ = utils.state_action_tensor(np.stack(rand_df['state']),
                                                       np.array(rand_df['action']),
                                                       self.args.action_dim)
        train_y_rand = torch.zeros((train_X_rand.shape[0], 1), device=self.args.device)

        train_X = torch.cat([train_X_true, train_X_rand], dim=0)
        train_y = torch.cat([train_y_true, train_y_rand], dim=0)

        rand_df = self.env.gen_random_data(len(test_y_true))
        test_X_rand, _, _ = utils.state_action_tensor(np.stack(rand_df['state']),
                                                      np.array(rand_df['action']),
                                                      self.args.action_dim)
        test_y_rand = torch.zeros((test_X_rand.shape[0], 1), device=self.args.device)

        test_X = torch.cat([test_X_true, test_X_rand], dim=0)
        test_y = torch.cat([test_y_true, test_y_rand], dim=0)

        curr_loss = 0
        early_stop = 0
        self.model_auth.train()
        for ep in range(self.args.train_steps):
            pred_y = self.model_auth(train_X)
            loss = self.loss(pred_y, train_y)

            if loss.item() == curr_loss or curr_loss < 0.01:
                early_stop += 1
            else:
                early_stop = 0
            curr_loss = loss.item()

            self.optimizer_auth.zero_grad()
            loss.backward()
            self.optimizer_auth.step()

            logger.info("auth training epoch {0} | current loss {1: .8f}".format(ep, curr_loss))
            if early_stop > 10:
                break

        logger.info("auth training complete | current loss {0: .4f}".format(curr_loss))
        self.model_auth.eval()
        pred_y = self.model_auth(test_X).detach()
        pred_y = (pred_y > 0.5).float()
        print("-- validator reward test result --")
        print("confusion metrix\n", metrics.confusion_matrix(test_y, pred_y))
        print("accuracy:", metrics.accuracy_score(test_y, pred_y))
        print("report\n", metrics.classification_report(test_y, pred_y))

    def train(self):
        self._train_nlg()
        self._train_auth()
        if self.args.dryrun is False:
            self.save()

    def save(self):
        torch.save(self.model_nlg.state_dict(), "../checkpoint/" + self.args.run_name + "_valid_nlg.ckpt")
        torch.save(self.model_auth.state_dict(), "../checkpoint/" + self.args.run_name + "_valid_auth.ckpt")
        logger.info("validator models saved!")

    def load(self):
        self.model_nlg.load_state_dict(torch.load("../checkpoint/" + self.args.run_name + "_valid_nlg.ckpt",
                                                  map_location=lambda x, y: x))
        self.model_auth.load_state_dict(torch.load("../checkpoint/" + self.args.run_name + "_valid_auth.ckpt",
                                                   map_location=lambda x, y: x))
        logger.info("validator models loaded!")


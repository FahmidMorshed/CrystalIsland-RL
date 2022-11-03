import dataclasses
import logging
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src import utils
from src.model import nets, ope
from src.model.fqe import FQE

logger = logging.getLogger(__name__)


class BCQ(object):
    def __init__(self, args: dataclasses, df: pd.DataFrame, df_test: pd.DataFrame, dr_estimator: FQE, name="bcq"):
        self.args = args
        self.df = deepcopy(df)
        self.df_test = deepcopy(df_test)
        s0 = np.stack(self.df.groupby('student_id').first()['state'])
        self.s0 = torch.tensor(s0, dtype=torch.float32, device=self.args.device) # todo calculate from df

        self.Q = nets.QNetwork(args)
        self.Q_target = nets.QNetwork(args)
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.Q_optim = torch.optim.Adam(self.Q.parameters(), lr=self.args.lr_bcq)

        self.buffer_tensor = utils.gen_buffer_tensor(deepcopy(self.df))
        self.loss = torch.nn.MSELoss()

        self.behavior_cloning = nets.BCNetwork(args)
        self.behavior_loss = torch.nn.functional.nll_loss
        self.bc_optim = torch.optim.Adam(self.behavior_cloning.parameters(), lr=self.args.lr_bcq)

        self.dr_estimator = dr_estimator

        self.name = name

    def select_action(self, state: np.ndarray):
        # take action according to the Q network
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.args.device)
        with torch.no_grad():
            q1 = self.Q(state_tensor)

            imt, i = self.behavior_cloning(state_tensor)
            # Use large negative number to mask actions from argmax
            imt = (imt / imt.max(1, keepdim=True)[0] > self.args.bcq_threshold).float()
            next_action = (imt * q1 + (1 - imt) * -1e8).argmax(1, keepdim=True)
            return next_action.squeeze().numpy()

    def train_behavior_cloning(self, force_train=False):
        if force_train is False:
            is_loaded = self.load_bc()
            if is_loaded:
                return

        logger.info('-- training bcq behavior cloning --')
        for epoch in range(self.args.behavior_cloning_train_steps):
            state, action, reward, next_state, done, next_state_action_musk = \
                utils.sample_buffer_tensor(self.buffer_tensor, self.args.bc_batch)
            action_eye = torch.eye(self.args.np_action_dim)[action.long()]

            imt, i = self.behavior_cloning(state)
            term1 = self.loss(imt, action_eye)
            loss = term1
            self.bc_optim.zero_grad()
            loss.backward()
            self.bc_optim.step()

            loss_val = term1.mean().detach()
            if (epoch + 1) % 1000 == 0:
                logger.info("training behavior cloning | step {0} or {1} | "
                            "loss {2: .4f}".format(epoch+1, self.args.behavior_cloning_train_steps, loss_val))

        logger.info('--finished training behavior cloning--')

        if self.args.dryrun is False:
            self.save_bc()

    def train(self, force_train=False):
        if force_train is False:
            is_loaded = self.load()
            if is_loaded:
                return

        logger.info("-- training bcq --")
        for epoch in range(self.args.bcq_train_steps):
            state, action, reward, next_state, done, next_state_action_musk = \
                utils.sample_buffer_tensor(self.buffer_tensor, self.args.bcq_batch)

            # Compute the target Q value
            with torch.no_grad():
                q1 = self.Q(next_state)

                # filter next_action
                q1 = next_state_action_musk * q1

                imt, i = self.behavior_cloning(next_state)
                # Use large negative number to mask actions from argmax
                imt = (imt / imt.max(1, keepdim=True)[0] > self.args.bcq_threshold).float()
                next_action = (imt * q1 + (1 - imt) * -1e8).argmax(1, keepdim=True)

                q2 = self.Q_target(next_state)

                q = 0.5 * torch.min(q1, q2) + (1 - 0.5) * torch.max(q1, q2)
                target_Q = reward.unsqueeze(dim=-1) + (1-done.unsqueeze(dim=-1)) * \
                           self.args.np_discount * q.gather(1, next_action).reshape(-1, 1)

            # Get current Q estimate
            current_Q = self.Q(state)
            current_Q = current_Q.gather(1, action.unsqueeze(dim=-1))

            # Compute Q loss
            Q_loss = self.loss(current_Q, target_Q)

            # Optimize the Q
            self.Q_optim.zero_grad()
            Q_loss.backward()
            self.Q_optim.step()

            if (epoch+1) % self.args.bcq_update_frequency == 0:
                self.print_logs(epoch, Q_loss)
                self.Q_target.load_state_dict(self.Q.state_dict())

        if self.args.dryrun is False:
            self.save()

    def save(self):
        torch.save(self.Q.state_dict(), "../checkpoint/" + self.args.run_name + "_" + self.name + "_bcq_q.ckpt")
        logger.info('-- saved bcq with run_name {0} --'.format(self.args.run_name))

    def load(self):
        is_loaded = False
        try:
            self.Q.load_state_dict(torch.load("../checkpoint/" + self.args.run_name + "_" + self.name + "_bcq_q.ckpt",
                                              map_location=lambda x, y: x))
            self.Q_target.load_state_dict(self.Q.state_dict())
            logger.info('-- loaded bcq with run_name {0} --'.format(self.args.run_name))
            is_loaded = True
        except FileNotFoundError:
            logger.info('-- no bcq with run_name {0} --'.format(self.args.run_name))
        return is_loaded

    def save_bc(self):
        torch.save(self.behavior_cloning.state_dict(), "../checkpoint/" + self.args.run_name + "_" + self.name + "_bcq_bc.ckpt")
        logger.info('-- saved behavior cloning with run_name {0} --'.format(self.args.run_name))

    def load_bc(self):
        is_loaded = False
        try:
            self.behavior_cloning.load_state_dict(torch.load("../checkpoint/" + self.args.run_name + "_" + self.name + "_bcq_bc.ckpt",
                                                             map_location=lambda x, y: x))

            logger.info('-- loaded behavior cloning with run_name {0} --'.format(self.args.run_name))
            is_loaded = True
        except FileNotFoundError:
            logger.info('-- no behavior cloning with run_name {0} --'.format(self.args.run_name))
        return is_loaded

    def ecr(self, states):
        current_Q = self.Q(states)
        max_q_val, idx = current_Q.max(dim=1)
        ecr = max_q_val.mean().item()
        return ecr

    def estimate_q(self, states: np.ndarray, actions: np.ndarray):
        state_tensor = torch.tensor(states, dtype=torch.float32, device=self.args.device)
        action_tensor = torch.tensor(actions, dtype=torch.int64, device=self.args.device)
        with torch.no_grad():
            current_Q = self.Q(state_tensor)
        action_Qs = current_Q.gather(1, action_tensor.unsqueeze(dim=-1))
        return action_Qs.squeeze()

    def estimate_v(self, states: np.ndarray):
        state_tensor = torch.tensor(states, dtype=torch.float32, device=self.args.device)
        with torch.no_grad():
            current_Q = self.Q(state_tensor)
        action_probs = self._compute_action_probs(state_tensor)
        v_val = torch.sum(current_Q * action_probs, dim=-1)
        return v_val

    def action_probs(self, states: np.ndarray, actions: np.ndarray):
        state_tensor = torch.tensor(states, dtype=torch.float32, device=self.args.device)
        action_tensor = torch.tensor(actions, dtype=torch.int64, device=self.args.device)

        with torch.no_grad():
            current_Q = self.Q(state_tensor)
        action_probs = F.softmax(current_Q, dim=1)
        action_probs = action_probs.gather(1, action_tensor.unsqueeze(dim=-1))
        return action_probs.detach()

    def _compute_action_probs(self, state_tensor: torch.Tensor) -> torch.tensor:
        """Compute action distribution over the action space.
        """
        with torch.no_grad():
            current_Q = self.Q(state_tensor)
        action_probs = F.softmax(current_Q, dim=1)
        return action_probs

    def print_logs(self, epoch, Q_loss):
        ecr = self.ecr(self.s0)
        _, mean_is, mean_wis = ope.importance_sampling_estimate(self)
        _, mean_dr = ope.doubly_robust_estimate(self, self.dr_estimator)
        mean_dm = ope.direct_method_estimate(self, self.dr_estimator)
        logger.info("epoch: {0}/{1} | Q loss: {2: .4f} | ecr: {3: .4f} | "
                    "is: {4: .4f} | wis: {5: .4f} | "
                    "dr: {6: .4f} | dm: {7: .4f}".format(epoch+1, self.args.bcq_train_steps, Q_loss, ecr,
                                                         mean_is, mean_wis, mean_dr, mean_dm))

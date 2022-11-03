import dataclasses
import logging
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src import utils
from src.model import nets

logger = logging.getLogger(__name__)


class FQE:
    """Pytorch implementation of the Fitted Q-Evaluation (FQE) model from
    https://arxiv.org/abs/1911.06854
    """
    def __init__(self, args: dataclasses, df: pd.DataFrame, name="fqe"):
        self.args = args
        self.df = deepcopy(df)
        s0 = np.stack(self.df.groupby('student_id').first()['state'])
        self.s0 = torch.tensor(s0, dtype=torch.float32, device=self.args.device)  # todo calculate from df

        self.Q = nets.QNetwork(args)
        self.Q_target = nets.QNetwork(args)
        self.Q_optim = torch.optim.Adam(self.Q.parameters(), lr=self.args.lr_fqe)

        self.buffer_tensor = utils.gen_buffer_tensor(deepcopy(self.df))
        self.loss = torch.nn.MSELoss()

        self.name = name

    def train(self, force_train=False):
        if force_train is False:
            is_loaded = self.load()
            if is_loaded:
                return

        logger.info("-- training fqe --")
        for epoch in range(self.args.fqe_train_steps):
            state, action, reward, next_state, done, next_state_action_musk = \
                utils.sample_buffer_tensor(self.buffer_tensor, self.args.fqe_batch)

            # todo check
            next_action_probs = self._compute_action_probs(next_state)

            # Compute Q-values for next state
            with torch.no_grad():
                next_q_values = self.Q_target(next_state)

                # Compute estimated state value next_v = E_{a ~ pi(s)} [Q(next_obs,a)]
                next_v = torch.sum(next_q_values * next_action_probs, dim=-1)
                target_Q = reward + (1 - done) * self.args.np_discount * next_v

            # Get current Q estimate
            current_Q = self.Q(state)
            current_Q = current_Q.gather(1, action.unsqueeze(dim=-1)).squeeze()

            # Compute Q loss
            Q_loss = self.loss(current_Q, target_Q)

            # Optimize the Q
            self.Q_optim.zero_grad()
            Q_loss.backward()
            self.Q_optim.step()

            if (epoch + 1) % self.args.fqe_update_frequency == 0:
                self.print_logs(epoch, Q_loss)
                self.Q_target.load_state_dict(self.Q.state_dict())

        if self.args.dryrun is False:
            self.save()

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

    def _compute_action_probs(self, state_tensor: torch.Tensor) -> torch.tensor:
        """Compute action distribution over the action space.
        """
        with torch.no_grad():
            current_Q = self.Q(state_tensor)
        action_probs = F.softmax(current_Q, dim=1)
        return action_probs

    def print_logs(self, epoch, Q_loss):
        ecr = self.ecr(self.s0)
        logger.info("epoch: {0}/{1} | Q loss: {2: .4f} | "
                    "ecr: {3: .4f}".format(epoch+1, self.args.fqe_train_steps, Q_loss, ecr))

    def save(self):
        torch.save(self.Q.state_dict(), "../checkpoint/" + self.args.run_name + "_" + self.name + "_fqe_q.ckpt")

        logger.info('-- saved fqe with run_name {0} --'.format(self.args.run_name))

    def load(self):
        loaded = False
        try:
            self.Q.load_state_dict(torch.load("../checkpoint/" + self.args.run_name + "_" + self.name + "_fqe_q.ckpt",
                                              map_location=lambda x, y: x))
            self.Q_target.load_state_dict(self.Q.state_dict())

            logger.info('-- loaded fqe with run_name {0} --'.format(self.args.run_name))
            loaded = True
        except FileNotFoundError:
            logger.info('-- no fqe with run_name {0} --'.format(self.args.run_name))

        return loaded

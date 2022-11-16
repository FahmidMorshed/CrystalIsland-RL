import dataclasses
import logging
from collections import deque
from copy import deepcopy
import pandas as pd
import torch
import numpy as np
from src import utils
from src.model import nets, ope

logger = logging.getLogger(__name__)


class BCQ(object):
    def __init__(self, args: dataclasses, df: pd.DataFrame = None, estimator=None, name="test"):
        self.args = args
        self.name = name

        self.Q = nets.QNetwork(self.args)
        self.Q_target = nets.QNetwork(self.args)
        self.Q_optim = torch.optim.Adam(self.Q.parameters(), lr=self.args.lr)

        self.loss = torch.nn.MSELoss()

        self.BC = nets.BCNetwork(self.args)
        self.bc_loss = torch.nn.MSELoss()  # torch.nn.functional.nll_loss
        self.bc_optim = torch.optim.Adam(self.BC.parameters(), lr=self.args.lr)

        if self.args.eval_only is False:
            if df is None:
                logger.error("must have a dataframe in the training mode")
                return
            if estimator is None:
                logger.error("must have a estimator in the training mode")
                return
            self.df = deepcopy(df)
            s0 = np.stack(self.df.groupby('episode').first()['state'])
            self.s0 = torch.tensor(s0, dtype=torch.float32, device=self.args.device)

            self.buffer_tensor = utils.gen_buffer_tensor(self.df)

            self.estimator = estimator
        else:
            self.load_q()
            self.load_bc()

    def train_behavior_cloning(self, force_train=False):
        if self.args.eval_only:
            logger.error("must run with eval_only as False")
            return

        if force_train is False:
            is_loaded = self.load_bc()
            if is_loaded:
                return

        prev_loss = 9999.0
        prev_diffs = deque(maxlen=100)
        for i in range(100):
            prev_diffs.append(9999.0)
        logger.info('-- training bcq behavior cloning --')
        for epoch in range(self.args.train_steps):
            state, action, reward, next_state, done, _ = \
                utils.sample_buffer_tensor(self.buffer_tensor, self.args.batch_size)
            action_eye = torch.eye(self.args.np_action_dim)[action.long()]

            imt, i = self.BC(state)
            term1 = self.loss(imt, action_eye)
            loss = term1
            self.bc_optim.zero_grad()
            loss.backward()
            self.bc_optim.step()

            curr_loss = term1.mean().detach()
            if (epoch + 1) % self.args.log_frequency == 0:
                logger.info("training behavior cloning | step {0} or {1} | "
                            "loss {2: .4f}".format(epoch + 1, self.args.train_steps, curr_loss))

            has_converged = utils.converged(curr_loss, prev_loss, prev_diffs)
            if has_converged:
                logger.info("-- early converged | epoch {0}/{1} --".format(epoch, self.args.train_steps))
                break
            else:
                prev_loss = curr_loss

        logger.info('--finished training behavior cloning--')

        if self.args.dryrun is False:
            self.save_bc()

    def train(self, force_train=False):
        if self.args.eval_only:
            logger.error("must run with eval_only as False")
            return

        if force_train is False:
            is_loaded = self.load_q()
            if is_loaded:
                return

        prev_loss = 9999.0
        prev_diffs = deque(maxlen=100)
        for i in range(100):
            prev_diffs.append(9999.0)
        logger.info("-- training bcq --")
        for epoch in range(self.args.train_steps):
            state, action, reward, next_state, done, next_state_action_musk = \
                utils.sample_buffer_tensor(self.buffer_tensor, self.args.batch_size)

            # Compute the target Q value
            with torch.no_grad():
                q1 = self.Q(next_state)

                # filter next_action
                q1 = next_state_action_musk * q1

                imt, i = self.BC(next_state)
                # Use large negative number to mask actions from argmax
                imt = (imt / imt.max(1, keepdim=True)[0] > self.args.bcq_threshold).float()
                next_action = (imt * q1 + (1 - imt) * -1e8).argmax(1, keepdim=True)

                q2 = self.Q_target(next_state)

                target_Q = reward + (1 - done) * self.args.discount_factor * q2.gather(1, next_action).squeeze()

            # Get current Q estimate
            current_Q = self.Q(state)
            current_Q = current_Q.gather(1, action.unsqueeze(dim=-1)).squeeze()

            # Compute Q loss
            Q_loss = self.loss(current_Q, target_Q)

            # Optimize the Q
            self.Q_optim.zero_grad()
            Q_loss.backward()
            self.Q_optim.step()

            if (epoch + 1) % self.args.update_frequency == 0:
                self.Q_target.load_state_dict(self.Q.state_dict())

            if (epoch + 1) % self.args.log_frequency == 0:
                self.print_logs(epoch, Q_loss)

            curr_loss = Q_loss.item()
            has_converged = utils.converged(curr_loss, prev_loss, prev_diffs)
            if has_converged:
                logger.info("-- early converged | epoch {0}/{1} --".format(epoch, self.args.train_steps))
                break
            else:
                prev_loss = curr_loss

        if self.args.dryrun is False:
            self.save_q()

    def ecr(self, states):
        current_Q = self.Q(states)
        max_q_val, idx = current_Q.max(dim=1)
        ecr = max_q_val.mean().item()
        return ecr

    def print_logs(self, epoch, loss):
        if self.args.eval_only:
            ecr = 0
        else:
            ecr = self.ecr(self.s0)

        mean_dm = ope.direct_method_estimate(self, self.estimator)
        _, mean_is, mean_wis = ope.importance_sampling_estimate(self, self.estimator)
        _, mean_dr = ope.doubly_robust_estimate(self, self.estimator)
        logger.info("epoch: {0}/{1} | loss: {2: .4f} | ecr: {3: .4f} | dm: {4: .4f} | is: {5: .4f} | "
                    "wis: {6: .4f} | dr: {7: .4f}"
                    .format(epoch + 1, self.args.train_steps, loss, ecr, mean_dm, mean_is, mean_wis, mean_dr))
        return ecr, mean_dm, mean_is, mean_wis, mean_dr

    def save_bc(self):
        torch.save(self.BC.state_dict(),
                   "../checkpoint/" + self.args.run_name + "_" + self.name + "_bcq_bc.ckpt")
        logger.info('-- saved behavior cloning with run_name {0} --'.format(self.args.run_name))

    def load_bc(self):
        is_loaded = False
        try:
            self.BC.load_state_dict(
                torch.load("../checkpoint/" + self.args.run_name + "_" + self.name + "_bcq_bc.ckpt",
                           map_location=lambda x, y: x))

            logger.info('-- loaded behavior cloning with run_name {0} --'.format(self.args.run_name))
            is_loaded = True
        except FileNotFoundError:
            logger.info('-- no behavior cloning with run_name {0} --'.format(self.args.run_name))
        return is_loaded

    def save_q(self):
        torch.save(self.Q.state_dict(), "../checkpoint/" + self.args.run_name + "_" + self.name + "_bcq_q.ckpt")
        logger.info('-- saved bcq with run_name {0} --'.format(self.args.run_name))

    def load_q(self):
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

    def select_action(self, state_tensor: torch.Tensor) -> torch.Tensor:
        # take action according to the target Q network
        with torch.no_grad():
            q1 = self.Q_target(state_tensor)

            imt, i = self.BC(state_tensor)
            # Use large negative number to mask actions from argmax
            imt = (imt / imt.max(1, keepdim=True)[0] > self.args.bcq_threshold).float()
            next_action = (imt * q1 + (1 - imt) * -1e8).argmax(1, keepdim=True)
        return next_action.squeeze().detach()

    def action_probs(self, state_tensor: torch.Tensor, action_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            current_Q = self.Q(state_tensor)
        action_probs = torch.nn.functional.softmax(current_Q, dim=1)
        action_probs = action_probs.gather(1, action_tensor.unsqueeze(dim=-1))
        return action_probs.detach()

    def get_action_for_state_in_np(self, state: np.ndarray) -> int:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.args.device)
        state_tensor = state_tensor.unsqueeze(dim=0)
        action_tensor = self.select_action(state_tensor)
        action = action_tensor.item()
        return action


class FQE(object):
    """
    This will be used on test set for calculating DM and DR metrics.
    Pytorch implementation of the Fitted Q-Evaluation (FQE) model from
    https://arxiv.org/abs/1911.06854
    """

    def __init__(self, args: dataclasses, df: pd.DataFrame):
        self.args = args
        self.df = deepcopy(df)
        s0 = np.stack(self.df.groupby('episode').first()['state'])
        self.s0 = torch.tensor(s0, dtype=torch.float32, device=self.args.device)

        self.Q = nets.QNetwork(args)
        self.Q_target = nets.QNetwork(args)
        self.Q_optim = torch.optim.Adam(self.Q.parameters(), lr=self.args.lr)

        self.buffer_tensor = utils.gen_buffer_tensor(df)
        self.loss = torch.nn.MSELoss()

    def train(self, force_train=False):
        if force_train is False:
            is_loaded = self.load()
            if is_loaded:
                return

        prev_loss = 9999.0
        prev_diffs = deque(maxlen=100)
        for i in range(100):
            prev_diffs.append(9999.0)
        logger.info("-- training fqe --")
        for epoch in range(self.args.train_steps):
            state, action, reward, next_state, done, _ = \
                utils.sample_buffer_tensor(self.buffer_tensor, self.args.batch_size)

            next_action_probs = self._compute_action_probs(next_state)

            # Compute Q-values for next state
            with torch.no_grad():
                next_q_values = self.Q_target(next_state)

                # Compute estimated state value next_v = E_{a ~ pi(s)} [Q(next_obs,a)]
                next_v = torch.sum(next_q_values * next_action_probs, dim=-1)
                target_Q = reward + (1 - done) * self.args.discount_factor * next_v

            # Get current Q estimate
            current_Q = self.Q(state)
            current_Q = current_Q.gather(1, action.unsqueeze(dim=-1)).squeeze()

            # Compute Q loss
            Q_loss = self.loss(current_Q, target_Q)

            # Optimize the Q
            self.Q_optim.zero_grad()
            Q_loss.backward()
            self.Q_optim.step()

            if (epoch + 1) % self.args.update_frequency == 0:
                self.Q_target.load_state_dict(self.Q.state_dict())

            if (epoch + 1) % self.args.log_frequency == 0:
                self.print_logs(epoch, Q_loss)

            curr_loss = Q_loss.item()
            has_converged = utils.converged(curr_loss, prev_loss, prev_diffs)
            if has_converged:
                logger.info("-- early converged | epoch {0}/{1} --".format(epoch, self.args.train_steps))
                break
            else:
                prev_loss = curr_loss

        if self.args.dryrun is False:
            self.save()

    def save(self):
        torch.save(self.Q.state_dict(), "../checkpoint/" + self.args.run_name + "_fqe_q.ckpt")

        logger.info('-- saved fqe with run_name {0} --'.format(self.args.run_name))

    def load(self):
        is_loaded = False
        try:
            self.Q.load_state_dict(torch.load("../checkpoint/" + self.args.run_name + "_fqe_q.ckpt",
                                              map_location=lambda x, y: x))
            self.Q_target.load_state_dict(self.Q.state_dict())

            logger.info('-- loaded fqe with run_name {0} --'.format(self.args.run_name))
            is_loaded = True
        except FileNotFoundError:
            logger.info('-- no fqe with run_name {0} --'.format(self.args.run_name))

        return is_loaded

    def ecr(self, states):
        current_Q = self.Q(states)
        max_q_val, idx = current_Q.max(dim=1)
        ecr = max_q_val.mean().item()
        return ecr

    def print_logs(self, epoch, loss):
        ecr = self.ecr(self.s0)
        logger.info("epoch: {0}/{1} | Q loss: {2: .4f} | "
                    "ecr: {3: .4f}".format(epoch + 1, self.args.train_steps, loss, ecr))

    def _compute_action_probs(self, state_tensor: torch.Tensor) -> torch.tensor:
        """Compute action distribution over the action space.
        """
        with torch.no_grad():
            current_Q = self.Q(state_tensor)
        action_probs = torch.nn.functional.softmax(current_Q, dim=1)
        return action_probs

    def estimate_q(self, state_tensor: torch.Tensor, action_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            current_Q = self.Q_target(state_tensor)
            action_Qs = current_Q.gather(1, action_tensor.unsqueeze(dim=-1))
        return action_Qs.squeeze().detach()

    def estimate_v(self, state_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            current_Q = self.Q(state_tensor)
            action_probs = self._compute_action_probs(state_tensor)
            v_val = torch.sum(current_Q * action_probs, dim=-1)
        return v_val.detach()


class RANDOM(object):
    def __init__(self, args: dataclasses, df: pd.DataFrame, estimator):
        self.args = args
        self.df = deepcopy(df)
        s0 = np.stack(self.df.groupby('episode').first()['state'])
        self.s0 = torch.tensor(s0, dtype=torch.float32, device=self.args.device)

        self.estimator = estimator

    def select_action(self, state_tensor: torch.Tensor) -> torch.Tensor:
        # take random action
        action_tensor = torch.randint(0, self.args.np_action_dim, size=(state_tensor.shape[0],))
        return action_tensor

    def action_probs(self, state_tensor: torch.Tensor, action_tensor: torch.Tensor) -> torch.Tensor:
        # have equal probability
        prob = 1.0 / (self.args.np_action_dim * 1.0)
        action_probs = torch.full(size=(state_tensor.shape[0],), fill_value=prob)
        return action_probs

    def print_logs(self):
        mean_dm = ope.direct_method_estimate(self, self.estimator)
        _, mean_is, mean_wis = ope.importance_sampling_estimate(self, self.estimator)
        _, mean_dr = ope.doubly_robust_estimate(self, self.estimator)
        logger.info("BASELINE WITH UNIFORM RANDOM POLICY | dm: {0: .4f} | is: {1: .4f} | "
                    "wis: {2: .4f} | dr: {3: .4f}"
                    .format(mean_dm, mean_is, mean_wis, mean_dr))

        return 0, mean_dm, mean_is, mean_wis, mean_dr

import dataclasses
import logging
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.model import nets, ope

logger = logging.getLogger(__name__)


class BCQ(object):
    def __init__(self, args: dataclasses, df: pd.DataFrame, df_test: pd.DataFrame, s0: np.ndarray, ):
        self.args = args
        self.df = deepcopy(df)
        self.s0 = torch.tensor(s0, dtype=torch.float32, device=self.args.device)
        self.Q = nets.QNetwork(args)
        self.Q_target = nets.QNetwork(args)
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.Q_optim = torch.optim.Adam(self.Q.parameters(), lr=self.args.lr_bcq)

        self.buffer_tensor = self._gen_buffer_tensor()
        self.df_test = deepcopy(df_test)

        self.loss = torch.nn.MSELoss()

        self.behavior_cloning = nets.BCNetwork(args)
        self.behavior_loss = torch.nn.functional.nll_loss
        self.bc_optim = torch.optim.Adam(self.behavior_cloning.parameters(), lr=self.args.lr_bcq)

    def select_action(self, state: np.ndarray, epsilon=0):
        # take action according to the Q network
        if np.random.uniform(0, 1) > epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).reshape(self.args.np_state_dim).to(self.args.device)
                q, imt, i = self.Q(state)
                imt = imt.exp()
                imt = (imt / imt.max(1, keepdim=True)[0] > self.args.bcq_threshold).float()
                # Use large negative number to mask actions from argmax
                return int((imt * q + (1. - imt) * -1e8).argmax(1))
        # take random action
        else:
            return np.random.randint(self.args.np_action_dim)

    def _musk_tensor_for_next_state_actions(self, next_states: torch.Tensor):
        action_tiggers = next_states[:, -4:].tolist()
        musk = []
        for row in action_tiggers:
            allowed_action = [0.] * 10
            if row == [0., 0., 1., 0.]:  # knowledge quiz
                allowed_action[5] = 1.
                allowed_action[6] = 1.
            elif row == [0., 1., 0., 0.]:  # teresa
                allowed_action[2] = 1.
                allowed_action[3] = 1.
                allowed_action[4] = 1.
            elif row == [1., 0., 0., 0.]:  # bryce
                allowed_action[0] = 1.
                allowed_action[1] = 1.
            elif row == [0., 0., 0., 1.]:  # diagnosis
                allowed_action[7] = 1
                allowed_action[8] = 1
                allowed_action[9] = 1

            musk.append(allowed_action)
        return torch.tensor(musk, dtype=torch.float32, device=self.args.device)

    def train_behavior_cloning(self):
        prev_loss = 0.
        early_stop = 0

        logger.info('-- training behavior cloning --')
        for epoch in range(self.args.bcq_train_steps):
            state, action, reward, next_state, done, next_state_action_musk = self.sample_buffer_tensor(all_data=False)
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
                            "loss {2: .4f}".format(epoch, self.args.bcq_train_steps, loss_val))
                if loss_val < 0.02:
                    early_stop += 1
                    if early_stop > 5:
                        break
                else:
                    early_stop = 0
            prev_loss = loss_val
        logger.info('--finished training behavior cloning--')

    def train(self):
        self.train_behavior_cloning()
        for epoch in range(self.args.bcq_train_steps):
            state, action, reward, next_state, done, next_state_action_musk = self.sample_buffer_tensor()

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

            if (epoch + 1) % 1000 == 0:
                self.print_logs(epoch, Q_loss)
            if (epoch+1) % self.args.bcq_update_frequency == 0:
                self.Q_target.load_state_dict(self.Q.state_dict())


        if self.args.dryrun is False:
            self.save()

    def save(self):
        torch.save(self.Q.state_dict(), "../checkpoint/" + self.args.run_name + "_bcq_q.ckpt")

    def load(self):
        self.Q.load_state_dict(torch.load("../checkpoint/" + self.args.run_name + "_bcq_q.ckpt",
                                          map_location=lambda x, y: x))
        self.Q_target.load_state_dict(self.Q.state_dict())

    def save_bc(self):
        torch.save(self.behavior_cloning.state_dict(), "../checkpoint/" + self.args.run_name + "_bcq_bc.ckpt")

    def load_bc(self):
        self.behavior_cloning.load_state_dict(torch.load("../checkpoint/" + self.args.run_name + "_bcq_bc.ckpt",
                                                         map_location=lambda x, y: x))

    def _gen_buffer_tensor(self) -> dict:
        """
        creates tensor for state, action, reward, next_state, and done
        :return: returns a dictionary of five tensor sets
        """
        df = deepcopy(self.df)
        states = np.stack(df['state'])
        actions = np.array(df['action'])
        rewards = np.array(df['reward'])
        dones = np.array(df['done'])

        next_states = np.stack(df['state'][1:])
        next_states = np.vstack([next_states, np.zeros(next_states.shape[1])])
        idx = np.where(dones == True)
        next_states[idx] = np.zeros(next_states.shape[1])

        state_tensor = torch.tensor(states, dtype=torch.float32, device=self.args.device)
        action_tensor = torch.tensor(actions, dtype=torch.int64, device=self.args.device)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.args.device)
        next_state_tensor = torch.tensor(next_states, dtype=torch.float32, device=self.args.device)
        done_tensor = torch.tensor(dones, dtype=torch.float32, device=self.args.device)

        musk_tensor = self._musk_tensor_for_next_state_actions(next_state_tensor)

        ret_dict = {'state': state_tensor, 'action': action_tensor, 'reward': reward_tensor,
                    'next_state': next_state_tensor, 'done': done_tensor, 'musk': musk_tensor}
        return ret_dict

    def sample_buffer_tensor(self, all_data=False) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        sample transactions of given batch size in tensor format
        :return: five tensors that contains samples of state, action, reward, next_state, done
        """
        total_rows = self.buffer_tensor['state'].size()[0]
        idx = np.random.choice(range(total_rows), self.args.bcq_batch)
        if all_data:
            idx = np.array(range(total_rows))

        state, action, reward, next_state, done, next_state_action_musk = (self.buffer_tensor['state'][idx],
                                                   self.buffer_tensor['action'][idx],
                                                   self.buffer_tensor['reward'][idx],
                                                   self.buffer_tensor['next_state'][idx],
                                                   self.buffer_tensor['done'][idx],
                                                   self.buffer_tensor['musk'][idx])

        return state, action, reward, next_state, done, next_state_action_musk

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
        v_val = current_Q.sum(dim=1)
        return v_val

    def action_probs(self, states: np.ndarray, actions: np.ndarray):
        state_tensor = torch.tensor(states, dtype=torch.float32, device=self.args.device)
        action_tensor = torch.tensor(actions, dtype=torch.int64, device=self.args.device)

        with torch.no_grad():
            current_Q = self.Q(state_tensor)
        action_probs = F.softmax(current_Q, dim=1)
        action_probs = action_probs.gather(1, action_tensor.unsqueeze(dim=-1))
        return action_probs.detach()

    def print_logs(self, epoch, Q_loss):
        ecr = self.ecr(self.s0)
        _, dr_behavior, dr_target = ope.doubly_robust_estimate(self.df_test, self,
                                                               np_discount=self.args.np_discount)
        _, is_behavior, is_target = 0, 0, 0
        _, wis_behavior, wis_target = 0, 0, 0
        logger.info("epoch: {0}/{1} | updating target | Q loss: {2: .4f} | "
                    "s0 ecr: {3: .4f} | dr behavior: {4: .4f} | dr target: "
                    "{5: .4f} | is behavior: {6: .4f} | is target: "
                    "{7: .4f} | wis behavior: {8: .4f} | wis target: "
                    "{9: .4f}".format(epoch, self.args.bcq_train_steps, Q_loss, ecr, dr_behavior,
                                      dr_target, is_behavior, is_target, wis_behavior, wis_target))

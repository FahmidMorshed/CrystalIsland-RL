import dataclasses
import logging
from copy import deepcopy

import numpy as np
import pandas as pd
import torch

from src.model import nets

logger = logging.getLogger(__name__)

class BCQ(object):
    def __init__(self, args: dataclasses, df: pd.DataFrame):
        self.args = args
        self.df = deepcopy(df)
        self.Q = nets.QNetwork(args)
        self.Q_target = nets.QNetwork(args)
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.Q_optim = torch.optim.Adam(self.Q.parameters(), lr=self.args.lr_bcq)

        self.buffer_tensor = self._gen_buffer_tensor()

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

    def train(self):
        for epoch in range(self.args.bcq_train_steps):
            # TODO
            # Sample replay buffer
            state, action, reward, next_state, done = self.sample_buffer_tensor()

            # Compute the target Q value
            with torch.no_grad():
                q, imt, i = self.Q(next_state)
                imt = imt.exp()
                imt = (imt / imt.max(1, keepdim=True)[0] > self.args.bcq_threshold).float()

                # Use large negative number to mask actions from argmax
                next_action = (imt * q + (1 - imt) * -1e8).argmax(1, keepdim=True)

                q, imt, i = self.Q_target(next_state)
                target_Q = reward + (1 - done) * self.args.np_discount * q.gather(1, next_action).reshape(-1, 1)

            # Get current Q estimate
            current_Q, imt, i = self.Q(state)
            current_Q = current_Q.gather(1, action.unsqueeze(dim=-1))

            # Compute Q loss
            smooth_l1_loss = torch.nn.SmoothL1Loss()
            q_loss = smooth_l1_loss(current_Q, target_Q)
            nll_loss = torch.nn.NLLLoss()
            i_loss = nll_loss(imt, action)

            Q_loss = q_loss + i_loss + 1e-2 * i.pow(2).mean()

            # Optimize the Q
            self.Q_optim.zero_grad()
            Q_loss.backward()
            self.Q_optim.step()

            if epoch % self.args.bcq_update_frequency == 0:
                logger.info("epoch: {0}/{1} | updating target q network | last Q loss: {2: .4f}".format(
                    epoch, self.args.bcq_train_steps, Q_loss))
                self.Q_target.load_state_dict(self.Q.state_dict())


        self.Q_target.load_state_dict(self.Q.state_dict())

        if self.args.dryrun is False:
            self.save()

    def save(self):
        torch.save(self.Q.state_dict(), "../checkpoint/" + self.args.run_name + "_bcq_q.ckpt")

    def load(self):
        self.Q.load_state_dict(torch.load("../checkpoint/" + self.args.run_name + "_bcq_q.ckpt",
                                          map_location=lambda x, y: x))
        self.Q_target.load_state_dict(self.Q.state_dict())

    def _gen_buffer_tensor(self) -> dict:
        """
        creates tensor for state, action, reward, next_state, and done
        :return: returns a dictionary of five tensor sets
        """
        df = deepcopy(self.df)
        states = np.stack(df['state'])
        actions = np.array(df['action'])
        rewards = np.array(df['nlg'])
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

        ret_dict = {'state': state_tensor, 'action': action_tensor, 'reward': reward_tensor,
                    'next_state': next_state_tensor, 'done': done_tensor}
        return ret_dict

    def sample_buffer_tensor(self) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        sample transactions of given batch size in tensor format
        :return: five tensors that contains samples of state, action, reward, next_state, done
        """
        total_rows = self.buffer_tensor['state'].size()[0]
        idx = np.random.choice(range(total_rows), self.args.bcq_batch)

        state, action, reward, next_state, done = (self.buffer_tensor['state'][idx],
                                                   self.buffer_tensor['action'][idx],
                                                   self.buffer_tensor['reward'][idx],
                                                   self.buffer_tensor['next_state'][idx],
                                                   self.buffer_tensor['done'][idx])

        return state, action, reward, next_state, done

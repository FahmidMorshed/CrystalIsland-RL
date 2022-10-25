import dataclasses

import torch
import torch.nn as nn
import logging

from torch import Tensor
from torch.distributions import Categorical
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PolicyModel(nn.Module):
    def __init__(self, args: dataclasses):
        super(PolicyModel, self).__init__()
        self.args = args
        self.actor = nn.Sequential(
            nn.Linear(self.args.state_dim, self.args.units),
            nn.Tanh(),
            nn.Linear(self.args.units, self.args.units),
            nn.Tanh(),
            nn.Linear(self.args.units, self.args.action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(self.args.state_dim, self.args.units),
            nn.Tanh(),
            nn.Linear(self.args.units, self.args.units),
            nn.Tanh(),
            nn.Linear(self.args.units, 1)
        )

    def act(self, state: Tensor):
        action_prob = self.actor(state)
        action_dist = Categorical(action_prob)
        action = action_dist.sample()
        action_log_prob = action_dist.log_prob(action)
        return action, action_log_prob

    def evaluate(self, state: Tensor, action: Tensor):
        action_prob = self.actor(state)
        action_dist = Categorical(action_prob)
        action_log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        value = self.critic(state)
        return value, action_log_prob, entropy

    def forward(self):
        raise NotImplementedError


class Discriminator(nn.Module):
    def __init__(self, args: dataclasses):
        super(Discriminator, self).__init__()
        self.args = args
        self.model = nn.Sequential(
            nn.Linear(self.args.state_dim + self.args.action_dim, self.args.units),
            nn.Tanh(),
            nn.Linear(self.args.units, self.args.units),
            nn.Tanh(),
            nn.Linear(self.args.units, 1),
            nn.Sigmoid()
        )

    def forward(self, state_action: Tensor):
        reward = self.model(state_action)
        return reward

# this section is for Narrative Planner RL
# any state/action refer onward represents narrative planner state/action, not student state/action
class QNetwork(nn.Module):
    def __init__(self, args: dataclasses):
        super(QNetwork, self).__init__()
        self.args = args
        self.q1 = nn.Linear(self.args.np_state_dim, self.args.units)
        self.q2 = nn.Linear(self.args.units, self.args.units)
        self.q3 = nn.Linear(self.args.units, self.args.np_action_dim)

        self.i1 = nn.Linear(self.args.np_state_dim, self.args.units)
        self.i2 = nn.Linear(self.args.units, self.args.units)
        self.i3 = nn.Linear(self.args.units, self.args.np_action_dim)

    def forward(self, state: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        q = F.relu(self.q1(state))
        q = F.relu(self.q2(q))
        q = self.q3(q)

        i = F.relu(self.i1(state))
        i = F.relu(self.i2(i))
        i = F.relu(self.i3(i))
        i_log_softmax = F.log_softmax(i, dim=1)

        return q, i_log_softmax, i


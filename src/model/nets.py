import dataclasses

import torch.nn as nn
import logging

from torch import Tensor
from torch.distributions import Categorical

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


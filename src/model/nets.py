import dataclasses

import torch
from torch.nn import Module, Sequential, Linear, Tanh, Embedding
from torch.distributions import Categorical


class PolicyNetwork(Module):
    def __init__(self, args: dataclasses) -> None:
        super().__init__()
        self.args = args

        self.net = Sequential(
            Linear(self.args.state_dim, self.args.units),
            Tanh(),
            Linear(self.args.units, self.args.units),
            Tanh(),
            Linear(self.args.units, self.args.units),
            Tanh(),
            Linear(self.args.units, self.args.action_dim),
        )

    def forward(self, states):
        probs = torch.softmax(self.net(states), dim=-1)
        distb = Categorical(probs)
        return distb


class ValueNetwork(Module):
    def __init__(self, args: dataclasses) -> None:
        super().__init__()
        self.args = args
        self.net = Sequential(
            Linear(self.args.state_dim, self.args.units),
            Tanh(),
            Linear(self.args.units, self.args.units),
            Tanh(),
            Linear(self.args.units, self.args.units),
            Tanh(),
            Linear(self.args.units, 1),
        )

    def forward(self, states):
        return self.net(states)


class Discriminator(Module):
    def __init__(self, args: dataclasses) -> None:
        super().__init__()
        self.args = args

        self.act_emb = Embedding(
            self.args.action_dim, self.args.state_dim
        )
        self.net_in_dim = 2 * self.args.state_dim

        self.net = Sequential(
            Linear(self.net_in_dim, self.args.units),
            Tanh(),
            Linear(self.args.units, self.args.units),
            Tanh(),
            Linear(self.args.units, self.args.units),
            Tanh(),
            Linear(self.args.units, 1),
        )

    def forward(self, states, actions):
        return torch.sigmoid(self.get_logits(states, actions))

    def get_logits(self, states, actions):
        actions = self.act_emb(actions.long())
        sa = torch.cat([states, actions], dim=-1)
        return self.net(sa)


# for Narrative Planner RL
class QNetwork(Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q1 = Linear(self.state_dim, self.args.units)
        self.q2 = Linear(self.args.units, self.args.units)
        self.q3 = Linear(self.args.units, self.action_dim)

    def forward(self, state: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        q = torch.relu(self.q1(state))
        q = torch.relu(self.q2(q))
        q = self.q3(q)
        return q


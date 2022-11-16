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

    def forward(self, state: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        q = torch.tanh(self.q1(state))
        q = torch.tanh(self.q2(q))
        q = self.q3(q)
        return q


class BCNetwork(nn.Module):
    def __init__(self, args: dataclasses):
        super(BCNetwork, self).__init__()
        self.args = args

        self.i1 = nn.Linear(self.args.np_state_dim, self.args.units)
        self.i2 = nn.Linear(self.args.units, self.args.units)
        self.i3 = nn.Linear(self.args.units, self.args.np_action_dim)

    def forward(self, state: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):

        i = F.relu(self.i1(state))
        i = F.relu(self.i2(i))
        i = F.relu(self.i3(i))
        i_log_softmax = F.softmax(i, dim=1)

        return i_log_softmax, i


class LSTMAttention(nn.Module):
    def __init__(self, args: dataclasses):
        super(LSTMAttention, self).__init__()
        self.args = args
        self.lstm = nn.LSTM(self.args.state_dim, self.args.validator_hidden_dim, self.args.validator_n_layers,
                            dropout=self.args.validator_dropout, batch_first=True)

        self.atten = nn.MultiheadAttention(self.args.validator_hidden_dim, self.args.validator_attn_head)
        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layers
        self.fc = nn.Linear(self.args.validator_hidden_dim, 1)  # predict binary high/low nlg, thus 1
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)

        lstm_out, hidden = self.lstm(x, hidden)

        attn_output, attn_output_weights = self.atten(lstm_out, lstm_out, lstm_out)

        attn_output = attn_output.contiguous().view(-1, self.args.validator_hidden_dim)

        # # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.args.validator_hidden_dim)

        # dropout and fully-connected layer
        out = self.dropout(attn_output)
        out = self.fc(out)
        # sigmoid function
        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]  # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        hidden = (weight.new(self.args.validator_n_layers, batch_size, self.args.validator_hidden_dim).zero_(),
                  weight.new(self.args.validator_n_layers, batch_size, self.args.validator_hidden_dim).zero_())

        return hidden
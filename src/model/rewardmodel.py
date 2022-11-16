import dataclasses
import logging
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn import metrics
from src.model import nets

logger = logging.getLogger(__name__)


def create_states_labels(df, max_len=200):
    all_states = []
    labels = []

    for student, d in df.groupby('student_id'):
        states = d['state']
        states = np.stack(states)
        curr_len = len(states)
        if curr_len <= max_len:
            pad_len = max_len - curr_len
            states = np.pad(states, pad_width=[(0, pad_len), (0, 0)], mode='constant', constant_values=0.)
        else:
            states = states[-max_len:, :]  # todo we can also take the last ones

        all_states.append(states)

        labels.append(d.iloc[-1]['reward'])

    labels = np.array([1 if r == 100. else 0 for r in labels])
    states = np.stack(all_states)
    return states, labels


class OutcomePredictor():
    def __init__(self, args: dataclasses, df: pd.DataFrame, df_test: pd.DataFrame):
        self.args = args
        self.df = deepcopy(df)
        self.df_test = deepcopy(df_test)

        self.batch_size = 64

        train_X, train_y = create_states_labels(self.df)
        test_X, test_y = create_states_labels(self.df_test)
        train_data = TensorDataset(torch.from_numpy(train_X).type(torch.FloatTensor),
                                   torch.from_numpy(train_y).type(torch.LongTensor))
        test_data = TensorDataset(torch.from_numpy(test_X).type(torch.FloatTensor),
                                  torch.from_numpy(test_y).type(torch.LongTensor))

        # make sure to SHUFFLE your data
        self.train_loader = DataLoader(train_data, shuffle=True, batch_size=self.batch_size)
        self.test_loader = DataLoader(test_data, shuffle=True, batch_size=self.batch_size)

        self.model = nets.LSTMAttention(args)
        self.loss = nn.BCELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

    def train(self):
        self.model.train()
        # init hidden state

        clip = 5  # gradient clipping
        for ep in range(self.args.train_steps):
            losses = []
            for train_X, train_y in self.train_loader:
                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                h = self.model.init_hidden(train_X.size(0))
                pred_y, h = self.model(train_X, h)

                loss = self.loss(pred_y.squeeze(), train_y.float())

                self.optim.zero_grad()
                loss.backward()
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                self.optim.step()

                losses.append(loss.item())

            logger.info("reward training epoch {0} | current loss {1: .8f}".format(ep, np.mean(losses)))

            # show test set result
            if (ep + 1) % 10 == 0:
                test_h = self.model.init_hidden(self.batch_size)
                test_losses = []
                test_accs = []
                test_f1s = []
                self.model.eval()
                for test_X, test_y in self.test_loader:
                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    test_h = self.model.init_hidden(test_X.size(0))

                    pred_y, test_h = self.model(test_X, test_h)
                    test_loss = self.loss(pred_y.squeeze(), test_y.float())
                    test_losses.append(test_loss.item())

                    pred_y_np = (pred_y >= 0.5).int().numpy()
                    test_y_np = test_y.numpy()

                    acc = metrics.accuracy_score(test_y_np, pred_y_np)
                    f1 = metrics.f1_score(test_y_np, pred_y_np)
                    test_accs.append(acc)
                    test_f1s.append(f1)

                logger.info("test set loss {0: .8f} | acc: {1: .4f} | f1: {2: .4f}".format(
                    np.mean(test_losses), np.mean(test_accs), np.mean(test_f1s)))

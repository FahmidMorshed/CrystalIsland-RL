from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch import FloatTensor

from src import utils


def avg_rewards_actions(policy, df):
    rewards = df.groupby('episode')['reward'].sum().tolist()
    d = df.groupby('episode')['action'].apply(list)
    action_counts = [Counter(acts) for ep, acts in d.items()]
    avg_action_counts = pd.DataFrame(action_counts).mean().reset_index().sort_values(by='index').set_index('index').round(1).to_dict()[0]
    return np.mean(rewards), avg_action_counts


def perplexity(policy):
    return np.exp(-np.mean(policy.test_df.apply(lambda x: policy.act_log_prob(x['state'], x['action']), axis=1)))


def kld(policy):
    q = FloatTensor(np.stack(policy.test_df.apply(lambda x: policy.get_probs(x['state']), axis=1)))
    p = FloatTensor(np.stack(policy.test_df['act_prob']))
    return (p * (torch.log(p) - torch.log(q))).sum(-1).mean().item()


def anomaly(policy, df):
    X = utils.actions_by_ep(df)
    y = policy.env.anomaly_detector.predict(X)  # outlier labels (0 or 1)
    anomaly_in_percent = sum(y) / len(y) * 100.0
    return anomaly_in_percent

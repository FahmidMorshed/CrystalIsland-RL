from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch import FloatTensor


def steps_rewards_actions(policy):
    actions = []
    steps = []
    rewards = []
    step = 0
    state = policy.env.reset()
    action_counts = []
    curr_reward = 0
    for i in range(len(policy.test_df)):
        step += 1
        action = policy.get_action(state)

        actions.append(action)
        state, reward, done, info = policy.env.step(action)

        curr_reward += reward
        if done:
            steps.append(step)
            act_count = Counter(actions)
            action_counts.append(act_count)
            rewards.append(curr_reward)
            curr_reward = 0
            actions = []
            state = policy.env.reset()
            step = 0

    avg_action_counts = pd.DataFrame(action_counts).mean().reset_index().sort_values(by='index').set_index('index').round(1).to_dict()[0]
    return np.mean(steps), np.mean(rewards), avg_action_counts


def perplexity(policy):
    return np.exp(-np.mean(policy.test_df.apply(lambda x: policy.act_log_prob(x['state'], x['action']), axis=1)))


def kld(policy):
    q = FloatTensor(np.stack(policy.test_df.apply(lambda x: policy.get_probs(x['state']), axis=1)))
    p = FloatTensor(np.stack(policy.test_df['act_prob']))
    return (p * (torch.log(p) - torch.log(q))).sum(-1).mean().item()

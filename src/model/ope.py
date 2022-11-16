import dataclasses
from copy import deepcopy
from typing import List, Dict

import numpy as np
import pandas as pd
import torch


def doubly_robust_estimate(policy, estimator) -> (List[Dict], float):
    """The Doubly Robust estimator.
        Let s_t, a_t, and r_t be the state, action, and reward at timestep t.
        This method takes a traiend Q-model for the evaluation policy \pi_e on behavior
        data generated by \pi_b.
        For behavior policy \pi_b and evaluation policy \pi_e, define the
        cumulative importance ratio at timestep t as:
        p_t = \sum_{t'=0}^t (\pi_e(a_{t'} | s_{t'}) / \pi_b(a_{t'} | s_{t'})).
        Consider an episode with length T. Let V_T = 0.
        For all t in {0, T - 1}, use the following recursive update:
        V_t^DR = (\sum_{a \in A} \pi_e(a | s_t) Q(s_t, a))
            + p_t * (r_t + \gamma * V_{t+1}^DR - Q(s_t, a_t))
        This estimator computes the expected return for \pi_e for an episode as:
        V^{\pi_e}(s_0) = V_0^DR
        and returns the mean and standard deviation over episodes.
        For more information refer to https://arxiv.org/pdf/1911.06854.pdf"""
    df_test = estimator.df
    all_estimates = []
    for episode, df in df_test.groupby('episode'):
        estimates_per_episode = {}
        rewards, old_prob = np.array(df["reward"]), np.array(df["action_prob"])
        ep_length = len(df)

        states = np.stack(df['state'])
        actions = np.array(df['action'])
        state_tensor = torch.tensor(states, dtype=torch.float32, device=policy.args.device)
        action_tensor = torch.tensor(actions, dtype=torch.int64, device=policy.args.device)

        new_prob = policy.action_probs(state_tensor, action_tensor)
        new_prob = new_prob.squeeze().numpy()

        v_target = 0.0
        q_values = estimator.estimate_q(state_tensor, action_tensor)
        q_values = q_values.numpy()
        v_values = estimator.estimate_v(state_tensor)
        v_values = v_values.numpy()

        if new_prob.shape == ():
            new_prob = new_prob.reshape(1, )
            q_values = q_values.reshape(1, )

        for t in reversed(range(ep_length)):
            v_target = v_values[t] + (new_prob[t] / old_prob[t]) * (
                    rewards[t] + policy.args.discount_factor * v_target - q_values[t]
            )
        v_target = v_target.item()

        estimates_per_episode["episode"] = episode
        estimates_per_episode["v_target"] = v_target

        all_estimates.append(estimates_per_episode)

    mean_dr = sum(est['v_target'] for est in all_estimates) / len(all_estimates)

    return all_estimates, mean_dr


def importance_sampling_estimate(policy, estimator) -> (List[Dict], float, float):
    """The step-wise IS estimator.
    Let s_t, a_t, and r_t be the state, action, and reward at timestep t.
    For behavior policy \pi_b and evaluation policy \pi_e, define the
    cumulative importance ratio at timestep t as:
    p_t = \sum_{t'=0}^t (\pi_e(a_{t'} | s_{t'}) / \pi_b(a_{t'} | s_{t'})).
    This estimator computes the expected return for \pi_e for an episode as:
    V^{\pi_e}(s_0) = \sum_t \gamma ^ {t} * p_t * r_t
    and returns the mean and standard deviation over episodes.
    For more information refer to https://arxiv.org/pdf/1911.06854.pdf"""

    # implementation follows 3.2.2 in https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/paper-1.pdf
    df_test = estimator.df
    all_estimates = []
    w_t = {}
    for episode, df in df_test.groupby('episode'):
        estimates_per_episode = {}
        rewards, old_prob = np.array(df["reward"]), np.array(df["action_prob"])
        ep_length = len(df)

        states = np.stack(df['state'])
        actions = np.array(df['action'])
        state_tensor = torch.tensor(states, dtype=torch.float32, device=policy.args.device)
        action_tensor = torch.tensor(actions, dtype=torch.int64, device=policy.args.device)

        new_prob = policy.action_probs(state_tensor, action_tensor)
        new_prob = new_prob.squeeze().numpy()

        if new_prob.shape == ():
            new_prob = new_prob.reshape(1, )
        # calculate importance ratios
        p = []
        for t in range(ep_length):
            if t == 0:
                pt_prev = 1.0
            else:
                pt_prev = p[t - 1]
            pt = pt_prev * new_prob[t] / old_prob[t]
            p.append(pt)

            w_t[t] = pt + w_t.get(t, 0.)

        # in delayed reward, all r_t is 0 except the last one
        v_is = (policy.args.discount_factor ** ep_length) * p[-1] * rewards[-1]
        v_is = v_is.item()

        estimates_per_episode["episode"] = episode
        estimates_per_episode["v_is"] = v_is
        estimates_per_episode["total_steps"] = ep_length

        all_estimates.append(estimates_per_episode)

    for t, w in w_t.items():
        w_t[t] = w / len(all_estimates)
    for estimates_per_episode in all_estimates:
        ep_length = estimates_per_episode["total_steps"]
        v_is = estimates_per_episode["v_is"]
        # in delayed reward, all r_t is 0 except the last one
        v_wis = (v_is + 1e-8) / (w_t[ep_length - 1] + 1e-8)
        estimates_per_episode['v_wis'] = v_wis

    mean_is = sum(est['v_is'] for est in all_estimates) / len(all_estimates)
    mean_wis = sum(est['v_wis'] for est in all_estimates) / len(all_estimates)

    return all_estimates, mean_is, mean_wis


def direct_method_estimate(policy, estimator) -> (List[Dict], float):
    s0 = estimator.s0
    actions = policy.select_action(s0)
    q_values = estimator.estimate_q(s0, actions)

    return q_values.mean().item()

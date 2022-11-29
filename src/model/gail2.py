import dataclasses
import logging
from copy import deepcopy

import numpy as np
import torch

from torch.nn import Module

from src.model.nets2 import PolicyNetwork, ValueNetwork, Discriminator
from src.utils import get_flat_grads, get_flat_params, set_params, \
    conjugate_gradient, rescale_and_linesearch

from torch import FloatTensor
import pandas as pd
from collections import Counter

logger = logging.getLogger(__name__)

class GAIL(Module):
    def __init__(
        self,
        args: dataclasses,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        env
    ) -> None:
        super().__init__()
        self.args = args

        self.train_df = train_df
        self.exp_obs = np.stack(self.train_df['state'])
        self.exp_acts = np.array(self.train_df['action'])
        self.num_steps_per_iter = len(train_df)

        self.test_df = test_df
        self.test_obs = np.stack(self.test_df['state'])
        self.test_acts = np.array(self.test_df['action'])
        self.test_step = len(test_df)
        self.test_avg_step = round(test_df.groupby('episode').count()['done'].mean(), 2)

        self.env = env

        self.pi = PolicyNetwork(self.args)
        self.v = ValueNetwork(self.args)
        self.d = Discriminator(self.args)

    def get_networks(self):
        return [self.pi, self.v]

    def act(self, state):
        self.pi.eval()

        state = FloatTensor(state)
        distb = self.pi(state)

        action = distb.sample().detach().cpu().numpy()

        return action

    def act_log_prob(self, state, action):
        self.pi.eval()

        state = FloatTensor(state)
        distb = self.pi(state)

        probs = distb.probs.detach().cpu().numpy()
        return np.log(probs[action])

    def test_perplexity(self):
        return sum(self.test_df.apply(lambda x: self.act_log_prob(x['state'], x['action']), axis=1))

    def train(self, force_train=False):
        if force_train is False:
            is_loaded = self.load()
            if is_loaded:
                return

        opt_d = torch.optim.Adam(self.d.parameters())

        exp_obs = FloatTensor(self.exp_obs)
        exp_acts = FloatTensor(self.exp_acts)
        exp_rwd_mean = round(self.train_df.groupby('episode').sum(numeric_only=True)['reward'].mean(), 2)
        print("TARGET REWARD:", exp_rwd_mean)

        rwd_iter_means = []
        for i in range(self.args.train_steps):
            rwd_iter = []

            obs = []
            acts = []
            rets = []
            advs = []
            gms = []

            steps = 0
            while steps < self.num_steps_per_iter:
                ep_obs = []
                ep_acts = []
                ep_rwds = []
                ep_costs = []
                ep_disc_costs = []
                ep_gms = []
                ep_lmbs = []

                t = 0
                done = False

                ob = self.env.reset()

                while not done and steps < self.num_steps_per_iter:
                    act = self.act(ob)

                    diff = (t - self.args.episode_len_75p) / (self.args.episode_len_90p - self.args.episode_len_75p)
                    if np.random.uniform() < diff:
                        act = 21  # forcing worksheet submit

                    ep_obs.append(ob)
                    obs.append(ob)

                    ep_acts.append(act)
                    acts.append(act)

                    ob, rwd, done, info = self.env.step(int(act))

                    ep_rwds.append(rwd)
                    ep_gms.append(self.args.discount_factor ** t)
                    ep_lmbs.append(self.args.discount_factor ** t)

                    t += 1
                    steps += 1

                if done:
                    rwd_iter.append(np.sum(ep_rwds))

                ep_obs = FloatTensor(np.array(ep_obs))
                ep_acts = FloatTensor(np.array(ep_acts))
                ep_rwds = FloatTensor(ep_rwds)
                # ep_disc_rwds = FloatTensor(ep_disc_rwds)
                ep_gms = FloatTensor(ep_gms)
                ep_lmbs = FloatTensor(ep_lmbs)

                ep_costs = (-1) * torch.log(self.d(ep_obs, ep_acts))\
                    .squeeze().detach()
                ep_disc_costs = ep_gms * ep_costs

                ep_disc_rets = FloatTensor(
                    [sum(ep_disc_costs[i:]) for i in range(t)]
                )
                ep_rets = ep_disc_rets / ep_gms

                rets.append(ep_rets)

                self.v.eval()
                curr_vals = self.v(ep_obs).detach()
                next_vals = torch.cat(
                    (self.v(ep_obs)[1:], FloatTensor([[0.]]))
                ).detach()
                ep_deltas = ep_costs.unsqueeze(-1)\
                    + self.args.discount_factor * next_vals\
                    - curr_vals

                ep_advs = FloatTensor([
                    ((ep_gms * ep_lmbs)[:t - j].unsqueeze(-1) * ep_deltas[j:])
                    .sum()
                    for j in range(t)
                ])
                advs.append(ep_advs)

                gms.append(ep_gms)

            rwd_iter_means.append(np.mean(rwd_iter))

            obs = FloatTensor(np.array(obs))
            acts = FloatTensor(np.array(acts))
            rets = torch.cat(rets)
            advs = torch.cat(advs)
            gms = torch.cat(gms)

            # this is optional
            advs = (advs - advs.mean()) / advs.std()

            self.d.train()
            exp_scores = self.d.get_logits(exp_obs, exp_acts)
            nov_scores = self.d.get_logits(obs, acts)

            opt_d.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                exp_scores, torch.zeros_like(exp_scores)
            ) \
                + torch.nn.functional.binary_cross_entropy_with_logits(
                    nov_scores, torch.ones_like(nov_scores)
                )
            loss.backward()
            opt_d.step()

            self.v.train()
            old_params = get_flat_params(self.v).detach()
            old_v = self.v(obs).detach()

            def constraint():
                return ((old_v - self.v(obs)) ** 2).mean()

            grad_diff = get_flat_grads(constraint(), self.v)

            def Hv(v):
                hessian = get_flat_grads(torch.dot(grad_diff, v), self.v)\
                    .detach()

                return hessian

            g = get_flat_grads(
                ((-1) * (self.v(obs).squeeze() - rets) ** 2).mean(), self.v
            ).detach()
            s = conjugate_gradient(Hv, g).detach()

            Hs = Hv(s).detach()
            alpha = torch.sqrt(2 * self.args.epsilon / torch.dot(s, Hs))

            new_params = old_params + alpha * s

            set_params(self.v, new_params)

            self.pi.train()
            old_params = get_flat_params(self.pi).detach()
            old_distb = self.pi(obs)

            def L():
                distb = self.pi(obs)

                return (advs * torch.exp(
                            distb.log_prob(acts)
                            - old_distb.log_prob(acts).detach()
                        )).mean()

            def kld():
                distb = self.pi(obs)

                old_p = old_distb.probs.detach()
                p = distb.probs

                return (old_p * (torch.log(old_p) - torch.log(p)))\
                    .sum(-1)\
                    .mean()

            grad_kld_old_param = get_flat_grads(kld(), self.pi)

            def Hv(v):
                hessian = get_flat_grads(
                    torch.dot(grad_kld_old_param, v),
                    self.pi
                ).detach()

                return hessian + self.args.cg_damping * v

            g = get_flat_grads(L(), self.pi).detach()

            s = conjugate_gradient(Hv, g).detach()
            Hs = Hv(s).detach()

            new_params = rescale_and_linesearch(
                g, s, Hs, self.args.max_kl, L, kld, old_params, self.pi
            )

            disc_causal_entropy = ((-1) * gms * self.pi(obs).log_prob(acts))\
                .mean()
            grad_disc_causal_entropy = get_flat_grads(
                disc_causal_entropy, self.pi
            )
            new_params += self.args.lambda_ * grad_disc_causal_entropy

            set_params(self.pi, new_params)

            if (i+1) % 10 == 0:
                rand, test, curr, rand_step, test_step, curr_step, rand_act_count, test_act_count, curr_act_count, \
                    curr_perp, rand_perp = self.eval_score
                logger.info(
                    "epoc: {0} | Reward Mean: {1: .2f} || D Score (avg_step) (perp) | "
                    "Rand: {2: .2f} ({3: .1f}) ({4: .2f}) | "
                    "Test: {5: .2f} ({6: .1f}) ({7: .2f}) | "
                    "Curr: {8: .2f} ({9: .1f}) ({10: .2f})"
                    .format(i + 1, np.mean(rwd_iter),
                            rand, rand_step, rand_perp,
                            test, test_step, 1.0,
                            curr, curr_step, curr_perp)
                )
                logger.info("Action Distribution: \nRand {0}\nTest {1}\nCurr {2}\n"
                            .format(rand_act_count, test_act_count, curr_act_count))
            else:
                logger.info(
                    "epoc: {0} | Reward Mean: {1: .2f}"
                    .format(i + 1, np.mean(rwd_iter))
                )

        if self.args.dryrun is False:
            self.save()

        return exp_rwd_mean, rwd_iter_means

    # an internal eval function to understand progress
    @property
    def eval_score(self):
        # random d score
        rand_states = []
        rand_actions = []
        rand_steps = []
        step = 0
        state = self.env.reset()
        for i in range(self.test_step):
            step += 1
            action = np.random.choice(range(self.args.action_dim))

            diff = (step - self.args.episode_len_75p) / (self.args.episode_len_90p - self.args.episode_len_75p)
            if np.random.uniform() < diff:
                action = 21  # forcing worksheet submit
            rand_states.append(state)
            rand_actions.append(action)
            state, reward, done, info = self.env.step(action)
            if done:
                state = self.env.reset()
                rand_steps.append(step)
                step = 0

        rand_act_count = Counter(rand_actions)
        rand_states = FloatTensor(np.array(rand_states))
        rand_actions = FloatTensor(np.array(rand_actions))

        # expert d score
        test_act_count = Counter(self.test_acts)
        test_obs = FloatTensor(self.test_obs)
        test_acts = FloatTensor(self.test_acts)

        # novice d score
        curr_states = []
        curr_actions = []
        curr_steps = []
        step = 0
        state = self.env.reset()
        for i in range(self.test_step):
            step += 1
            action = int(self.act(state))
            diff = (step - self.args.episode_len_75p) / (self.args.episode_len_90p - self.args.episode_len_75p)
            if np.random.uniform() < diff:
                action = 21  # forcing worksheet submit
            curr_states.append(state)
            curr_actions.append(action)
            state, reward, done, info = self.env.step(action)

            if done:
                state = self.env.reset()
                curr_steps.append(step)
                step = 0

        curr_act_count = Counter(curr_actions)
        curr_states = FloatTensor(np.array(curr_states))
        curr_actions = FloatTensor(np.array(curr_actions))

        with torch.no_grad():
            rand = self.d.get_logits(rand_states, rand_actions).mean().detach().item()
            test = self.d.get_logits(test_obs, test_acts).mean().detach().item()
            curr = self.d.get_logits(curr_states, curr_actions).mean().detach().item()

        self.env.reset()

        curr_perp = self.test_perplexity()
        rand_perp = len(self.test_df) * np.log(1.0 / 24.0)

        return rand, test, curr, np.mean(rand_steps), self.test_avg_step, np.mean(curr_steps), \
               sorted(rand_act_count.items()), sorted(test_act_count.items()), sorted(curr_act_count.items()), curr_perp, rand_perp

    def save(self):
        torch.save(self.pi.state_dict(), "../checkpoint/" + self.args.run_name + "_policy.ckpt")
        torch.save(self.d.state_dict(), "../checkpoint/" + self.args.run_name + "_discriminator.ckpt")
        torch.save(self.v.state_dict(), "../checkpoint/" + self.args.run_name + "_value.ckpt")

    def load(self):
        is_loaded = False
        try:
            self.pi.load_state_dict(torch.load("../checkpoint/" + self.args.run_name + "_policy.ckpt",
                                                   map_location=lambda x, y: x))
            self.d.load_state_dict(torch.load("../checkpoint/" + self.args.run_name + "_discriminator.ckpt",
                                              map_location=lambda x, y: x))
            self.v.load_state_dict(torch.load("../checkpoint/" + self.args.run_name + "_value.ckpt",
                                              map_location=lambda x, y: x))
            logger.info('-- loaded gail with run_name {0} --'.format(self.args.run_name))
            is_loaded = True
        except FileNotFoundError:
            logger.info('-- no gail with run_name {0} --'.format(self.args.run_name))
        return is_loaded

    def simulate(self, total_episode):
        logger.info("-- creating simulated data --")
        data = []
        for ep in range(total_episode):
            state = self.env.reset()
            ep_step = 0
            done = False
            while not done:
                action = int(self.act(state))
                diff = (ep_step - self.args.episode_len_75p) / (self.args.episode_len_90p - self.args.episode_len_75p)
                if np.random.uniform() < diff:
                    action = 21  # forcing worksheet submit
                next_state, reward, done, info = self.env.step(action)

                data.append({'episode': str(ep), 'step': ep_step, 'state': state, 'action': action, 'reward': reward,
                             'next_state': next_state, 'done': done, 'info': info})
                state = deepcopy(next_state)
                ep_step += 1

        df = pd.DataFrame(data, columns=['episode', 'step', 'state', 'action', 'reward', 'next_state', 'done', 'info'])

        if self.args.dryrun is False:
            df.to_pickle('../simulated_data/' + self.args.run_name + '_sim.pkl')
        return df

import dataclasses
import logging
import numpy as np
import torch

from src.model.policy import Policy
from src.model.nets import PolicyNetwork, ValueNetwork, Discriminator

from torch import FloatTensor
import pandas as pd
from collections import Counter

logger = logging.getLogger(__name__)


class GAIL(Policy):
    def __init__(
            self,
            args: dataclasses,
            train_df: pd.DataFrame,
            test_df: pd.DataFrame,
            env,
            name="gail"
    ) -> None:
        super().__init__(args, train_df, test_df, env, name)

        self.exp_obs = np.stack(self.train_df['state'])
        self.exp_acts = np.array(self.train_df['action'])
        self.num_steps_per_iter = len(train_df)

        action_count = Counter(np.array(self.test_df['action']))
        total_eps = len(test_df.groupby('episode').count())
        self.avg_action_counts = {act: round(count / total_eps, 1) for (act, count) in sorted(action_count.items())}
        self.test_reward = round(self.train_df.groupby('episode').sum(numeric_only=True)['reward'].mean(), 2)

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

    def get_action(self, state):
        return int(self.act(state))

    def get_probs(self, state: np.ndarray):
        self.pi.eval()

        state = FloatTensor(state)
        distb = self.pi(state)

        probs = distb.probs.detach().cpu().numpy()
        return probs

    def act_log_prob(self, state, action):
        probs = self.get_probs(state)
        return np.log(probs[action])

    def train(self, train_steps, force_train=False):
        logger.info("TEST DATA | Rewards: {0: .0f} |  Actions: {1}".format(self.test_reward, self.avg_action_counts))
        if force_train is False:
            is_loaded = self.load()
            if is_loaded:
                return

        opt_d = torch.optim.Adam(self.d.parameters())

        exp_obs = FloatTensor(self.exp_obs)
        exp_acts = FloatTensor(self.exp_acts)

        rwd_iter_means = []
        for i in range(train_steps):
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

                ep_step = 0
                done = False

                state = self.env.reset()

                while not done and steps < self.num_steps_per_iter:
                    action = self.act(state)

                    ep_obs.append(state)
                    obs.append(state)

                    ep_acts.append(action)
                    acts.append(action)

                    state, reward, done, info = self.env.step(int(action))

                    ep_rwds.append(reward)
                    ep_gms.append(self.args.discount_factor ** ep_step)
                    ep_lmbs.append(self.args.discount_factor ** ep_step)

                    ep_step += 1
                    steps += 1

                if done:
                    rwd_iter.append(np.sum(ep_rwds))

                ep_obs = FloatTensor(np.array(ep_obs))
                ep_acts = FloatTensor(np.array(ep_acts))
                ep_rwds = FloatTensor(ep_rwds)
                # ep_disc_rwds = FloatTensor(ep_disc_rwds)
                ep_gms = FloatTensor(ep_gms)
                ep_lmbs = FloatTensor(ep_lmbs)

                ep_costs = (-1) * torch.log(self.d(ep_obs, ep_acts)) \
                    .squeeze().detach()
                ep_disc_costs = ep_gms * ep_costs

                ep_disc_rets = FloatTensor(
                    [sum(ep_disc_costs[i:]) for i in range(ep_step)]
                )
                ep_rets = ep_disc_rets / ep_gms

                rets.append(ep_rets)

                self.v.eval()
                curr_vals = self.v(ep_obs).detach()
                next_vals = torch.cat(
                    (self.v(ep_obs)[1:], FloatTensor([[0.]]))
                ).detach()
                ep_deltas = ep_costs.unsqueeze(-1) \
                            + self.args.discount_factor * next_vals \
                            - curr_vals

                ep_advs = FloatTensor([
                    ((ep_gms * ep_lmbs)[:ep_step - j].unsqueeze(-1) * ep_deltas[j:])
                    .sum()
                    for j in range(ep_step)
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
                hessian = get_flat_grads(torch.dot(grad_diff, v), self.v) \
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

                return (old_p * (torch.log(old_p) - torch.log(p))) \
                    .sum(-1) \
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

            disc_causal_entropy = ((-1) * gms * self.pi(obs).log_prob(acts)) \
                .mean()
            grad_disc_causal_entropy = get_flat_grads(
                disc_causal_entropy, self.pi
            )
            new_params += self.args.lambda_ * grad_disc_causal_entropy

            set_params(self.pi, new_params)

            if (i + 1) % 10 == 0:
                self.eval_score(i + 1)
            else:
                logger.info(
                    "epoc: {0} | Target Reward Mean: {1: .2f} | Reward Mean: {2: .2f}"
                    .format(i + 1, self.test_reward, np.mean(rwd_iter))
                )

        if self.args.dryrun is False:
            self.save()

        return rwd_iter_means

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


#################################
# Some utility functions for GAIL
def get_flat_grads(f, net):
    flat_grads = torch.cat([
        grad.view(-1)
        for grad in torch.autograd.grad(f, net.parameters(), create_graph=True)
    ])

    return flat_grads


def get_flat_params(net):
    return torch.cat([param.view(-1) for param in net.parameters()])


def set_params(net, new_flat_params):
    start_idx = 0
    for param in net.parameters():
        end_idx = start_idx + np.prod(list(param.shape))
        param.data = torch.reshape(
            new_flat_params[start_idx:end_idx], param.shape
        )

        start_idx = end_idx


def conjugate_gradient(Av_func, b, max_iter=10, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b - Av_func(x)
    p = r
    rsold = r.norm() ** 2

    for _ in range(max_iter):
        Ap = Av_func(p)
        alpha = rsold / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.norm() ** 2
        if torch.sqrt(rsnew) < residual_tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x


def rescale_and_linesearch(
        g, s, Hs, max_kl, L, kld, old_params, pi, max_iter=10,
        success_ratio=0.1
):
    set_params(pi, old_params)
    L_old = L().detach()

    beta = torch.sqrt((2 * max_kl) / torch.dot(s, Hs))

    for _ in range(max_iter):
        new_params = old_params + beta * s

        set_params(pi, new_params)
        kld_new = kld().detach()

        L_new = L().detach()

        actual_improv = L_new - L_old
        approx_improv = torch.dot(g, beta * s)
        ratio = actual_improv / approx_improv

        if ratio > success_ratio \
                and actual_improv > 0 \
                and kld_new < max_kl:
            return new_params

        beta *= 0.5

    print("The line search was failed!")
    return old_params

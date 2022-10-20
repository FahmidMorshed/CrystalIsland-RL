import dataclasses
import pickle
from copy import deepcopy
from sklearn import metrics
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import logging

from src import utils
from src.model.crystalisland import CrystalIsland
from src.model.nets import PolicyModel, Discriminator

logger = logging.getLogger(__name__)


class GailExecutor:
    def __init__(self, args: dataclasses, train_student_df: pd.DataFrame, env: CrystalIsland):
        self.args = args
        logger.info("args: {0}".format(self.args.to_json_string()))

        # create networks
        self.pi = PolicyModel(self.args).to(self.args.device)
        self.pi_old = PolicyModel(self.args).to(self.args.device)
        self.pi_old.load_state_dict(self.pi.state_dict())
        self.d = Discriminator(self.args).to(self.args.device)

        # optimizers and schedulers
        self.optimizer_pi = torch.optim.Adam([
            {"params": self.pi.actor.parameters(), "lr": self.args.lr_actor},
            {"params": self.pi.critic.parameters(), "lr": self.args.lr_critic}
        ])
        self.optimizer_d = torch.optim.Adam(self.d.parameters(), lr=self.args.lr_discriminator)
        self.lr_scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_d,
                                                                     gamma=self.args.scheduler_gamma, verbose=False)
        self.lr_scheduler_pi = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_pi,
                                                                      gamma=self.args.scheduler_gamma, verbose=False)

        # define loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

        # create state and action tensors
        self.env = env
        self.train_student_df = train_student_df
        self.expert_state_actions, self.expert_states, self.expert_actions = \
            self._gen_state_action_tensor(np.stack(self.train_student_df['state']),
                                          np.array(self.train_student_df['action']))
        self.rand_data = self.env.gen_random_data(10000)
        self.rand_state_actions, _, _ = self._gen_state_action_tensor(np.stack(self.rand_data['state']),
                                                                      np.array(self.rand_data['action']))

        self.states = []
        self.actions = []
        self.log_prob_actions = []
        self.rewards = []
        self.is_terminal = []

        self.d_exp_score = 99999.0
        self.d_nov_score = 99990.0
        self.d_rand_score = 99999.0

        self.d_loss = 9999.0
        self.pi_loss = 9999.0

    def reset_buffers(self):
        self.states = []
        self.actions = []
        self.log_prob_actions = []
        self.rewards = []
        self.is_terminal = []

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.args.device)
        with torch.no_grad():
            action, action_log_prob = self.pi_old.act(state)
        self.states.append(state.detach())
        self.actions.append(action.detach())
        self.log_prob_actions.append(action_log_prob.detach())

        action = action.detach().item()
        next_state, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        self.is_terminal.append(done)

        return next_state, reward, done

    def _gen_state_action_tensor(self, states, actions):
        if type(states) is np.ndarray:
            state_tensor = torch.tensor(states, dtype=torch.float32, device=self.args.device)
            action_tensor = torch.tensor(actions, dtype=torch.int64, device=self.args.device)
        else:
            state_tensor = torch.stack(states, dim=0).to(self.args.device)
            action_tensor = torch.stack(actions, dim=0).to(self.args.device)
        actions_one_hot = torch.eye(self.args.action_dim)[action_tensor.long()].to(self.args.device)
        state_action_tensor = torch.cat([state_tensor, actions_one_hot], dim=1)
        return state_action_tensor, state_tensor, action_tensor

    def update(self):
        prev_log_prob_actions = torch.stack(self.log_prob_actions, dim=0).to(self.args.device)
        agent_state_actions, prev_states, prev_actions = self._gen_state_action_tensor(self.states, self.actions)

        curr_loss = 0.0
        for ep in range(self.args.internal_epoch_d):
            expert_prob = self.d(self.expert_state_actions)
            agent_prob = self.d(agent_state_actions)
            term1 = self.bce_loss(agent_prob, torch.ones((agent_state_actions.shape[0], 1), device=self.args.device))
            term2 = self.bce_loss(expert_prob, torch.zeros((self.expert_state_actions.shape[0], 1),
                                                           device=self.args.device))

            loss = term1 + term2
            curr_loss += loss.item()
            self.optimizer_d.zero_grad()
            loss.backward()
            self.optimizer_d.step()

        self.d_loss = curr_loss / self.args.internal_epoch_d

        with torch.no_grad():
            d_rewards = -torch.log(self.d(agent_state_actions))

        rewards = []
        cumulative_discounted_reward = 0.0
        for d_reward, terminal in zip(reversed(d_rewards), reversed(self.is_terminal)):
            if terminal:
                cumulative_discounted_reward = 0
            cumulative_discounted_reward = d_reward + (self.args.discount_factor * cumulative_discounted_reward)
            rewards.insert(0, cumulative_discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.args.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        curr_loss = 0.0
        for ep in range(self.args.internal_epoch_pi):
            # TODO CAN DO SAMPLING!!!
            values, log_prob_actions, entropy = self.pi.evaluate(prev_states, prev_actions)
            values = values.squeeze()
            advantages = rewards - values.detach()
            imp_ratios = torch.exp(log_prob_actions - prev_log_prob_actions)
            clamped_imp_ratio = torch.clamp(imp_ratios, 1 - self.args.clip_eps, 1 + self.args.clip_eps)
            term1 = -torch.min(imp_ratios, clamped_imp_ratio) * advantages
            term2 = 0.5 * self.mse_loss(values, rewards)
            term3 = -0.01 * entropy
            loss = term1 + term2 + term3
            curr_loss += loss.mean().item()
            self.optimizer_pi.zero_grad()
            loss.mean().backward()
            self.optimizer_pi.step()

        self.pi_loss = curr_loss / self.args.internal_epoch_pi
        self.pi_old.load_state_dict(self.pi.state_dict())

        self.d_nov_score = self.d(agent_state_actions).mean().detach().item()
        self.d_exp_score = self.d(self.expert_state_actions).mean().detach().item()
        self.d_rand_score = self.d(self.rand_state_actions).mean().detach().item()

        self.reset_buffers()

        self.lr_scheduler_pi.step()
        self.lr_scheduler_d.step()
        self.args.clip_eps = self.args.clip_eps * self.args.scheduler_gamma

    def run(self):
        t = 1
        success_count = 0
        update_count = 0
        finish = False
        while t <= self.args.train_steps:
            state = self.env.reset()
            done = False
            ep_len = 0
            while ep_len < self.args.max_episode_len:
                state, reward, done = self.take_action(state)
                if self.args.run_type == 'train' and t % self.args.update_steps == 0:
                    self.update()
                    update_count += 1
                    logger.info(
                        "iter: {0} | update: {1} | d_loss: {2:.2f} | pi_loss: {3: .2f} | "
                        "d_exp: {4: .4f} | d_nov: {5: .4f} | d_rand: {6: .4f}".format(
                            t, update_count, self.d_loss, self.pi_loss, self.d_exp_score, self.d_nov_score,
                            self.d_rand_score))

                    # check if conversed
                    if abs(self.d_exp_score - 0.5) <= self.args.d_stop_threshold and \
                            abs(self.d_nov_score - 0.5) <= self.args.d_stop_threshold:
                        success_count += 1
                        if success_count >= self.args.d_stop_count:
                            logger.info("--model converged. saving checkpoint--")
                            self.save()
                            finish = True
                    else:
                        success_count = 0

                t += 1
                ep_len += 1
                if done:
                    break
            if not done:
                logger.debug("truncated at horizon")
            if finish:
                self.save()
                break

    def save(self):
        torch.save(self.pi_old.state_dict(), "../checkpoint/policy.ckpt")
        torch.save(self.d.state_dict(), "../checkpoint/discriminator.ckpt")

    def load(self):
        policy_model_path = "../checkpoint/policy.ckpt"
        self.pi_old.load_state_dict(torch.load(policy_model_path, map_location=lambda x, y: x))
        self.pi.load_state_dict(self.pi_old.state_dict())
        discriminator_model_path = "../checkpoint/discriminator.ckpt"
        self.d.load_state_dict(torch.load(discriminator_model_path, map_location=lambda x, y: x))

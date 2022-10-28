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
from src.model.crystalisland import CrystalIsland, _gen_narrative_state
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
            utils.state_action_tensor(np.stack(self.train_student_df['state']),
                                      np.array(self.train_student_df['action']), self.args.action_dim)
        self.rand_data = self.env.gen_random_data(10000)
        self.rand_state_actions, _, _ = utils.state_action_tensor(np.stack(self.rand_data['state']),
                                                                  np.array(self.rand_data['action']),
                                                                  self.args.action_dim)

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

    def take_action(self, state, action: torch.Tensor = None):
        state = torch.tensor(state, dtype=torch.float32, device=self.args.device)
        if action:
            values, action_log_prob, entropy = self.pi.evaluate(state, action)
        else:
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

    def update(self):
        prev_log_prob_actions = torch.stack(self.log_prob_actions, dim=0).to(self.args.device)
        agent_state_actions, prev_states, prev_actions = utils.state_action_tensor(self.states, self.actions,
                                                                                   self.args.action_dim)

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
            term2 = 0.1 * self.mse_loss(values, rewards)
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

    def train(self):
        t = 1
        update_count = 0
        finish = False
        while True:
            state = self.env.reset()
            done = False
            ep_len = 0
            while ep_len < self.args.max_episode_len:
                state, reward, done = self.take_action(state)
                if t % self.args.update_steps == 0:
                    self.update()
                    update_count += 1
                    logger.info(
                        "iter: {0} | update: {1} | d_loss: {2:.2f} | pi_loss: {3: .2f} | "
                        "d_exp: {4: .4f} | d_nov: {5: .4f} | d_rand: {6: .4f}".format(
                            t, update_count, self.d_loss, self.pi_loss, self.d_exp_score, self.d_nov_score,
                            self.d_rand_score))

                    # TODO This is not valid criteria. https://arxiv.org/pdf/1802.03446.pdf also check
                    #  https://stats.stackexchange.com/questions/482653/what-is-the-stop-criteria-of-generative
                    #  -adversarial-nets
                    # if abs(self.d_exp_score - 0.5) <= self.args.d_stop_threshold and \
                    #         abs(self.d_nov_score - 0.5) <= self.args.d_stop_threshold:
                    if update_count >= self.args.gail_train_steps:
                        logger.info("--finished training. saving checkpoint--")
                        self.save()
                        finish = True
                        break

                t += 1
                ep_len += 1
                if done:
                    break
            if not done:
                action = torch.tensor(5, dtype=torch.int64, device=self.args.device)
                state, reward, done = self.take_action(state, action=action)
                logger.debug("truncated at horizon. forced end game")
            if finish:
                if self.args.dryrun is False:
                    self.save()
                break

    def save(self):
        torch.save(self.pi_old.state_dict(), "../checkpoint/" + self.args.run_name + "_policy.ckpt")
        torch.save(self.d.state_dict(), "../checkpoint/" + self.args.run_name + "_discriminator.ckpt")

    def load(self):
        self.pi_old.load_state_dict(torch.load("../checkpoint/" + self.args.run_name + "_policy.ckpt",
                                               map_location=lambda x, y: x))
        self.pi.load_state_dict(self.pi_old.state_dict())
        self.d.load_state_dict(torch.load("../checkpoint/" + self.args.run_name + "_discriminator.ckpt",
                                          map_location=lambda x, y: x))

    def simulate(self, episode, validator):
        logger.info("creating simulated data")
        data = []
        data_narr = []
        total_skipped = 0
        for ep in range(episode):
            state = self.env.reset()
            step_narr = 0
            ep_step = 0
            done = False
            ep_data = []
            ep_data_narr = []
            while ep_step < self.args.max_episode_len - 1:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.args.device)
                with torch.no_grad():
                    action, action_log_prob = self.pi_old.act(state_tensor)
                action = action.detach().item()
                next_state, reward, done, info = self.env.step(action)

                ep_data.append({'student_id': str(ep), 'step': ep_step, 'state': state, 'action': action, 'reward': reward,
                             'done': done, 'info': info})

                state = next_state
                if len(info) != 0:
                    ep_data_narr.append({'student_id': str(ep), 'step': step_narr,
                                      'state': info['narrative_state'], 'action': info['narrative_action'],
                                      'reward': 0, 'done': done, 'info': info})
                    step_narr += 1

                ep_step += 1

                if done:
                    if len(ep_data_narr) > 0:
                        ep_data_narr[-1]['done'] = True
                    break

            # force end if episode length is longer
            if not done:
                action = 5  # game end
                next_state, reward, done, info = self.env.step(action)
                ep_data.append({'student_id': str(ep), 'step': ep_step, 'state': state, 'action': action, 'reward': reward,
                             'done': done, 'info': info})
                if len(ep_data_narr) > 0:
                    ep_data_narr[-1]['done'] = True
                ep_step += 1

            # filter out invalid episodes and assign reward:
            states = np.stack([row['state'] for row in ep_data])
            actions = np.array([row['action'] for row in ep_data])
            is_authentic, is_high = validator.validate_episode(states, actions)
            if is_authentic is False:
                total_skipped += 1
                logger.info("skipping episode {0} | total skipped {1}".format(ep, total_skipped))
                continue
            reward = 100 if is_high else -100
            ep_data[-1]['reward'] = reward
            ep_data_narr[-1]['reward'] = reward
            data += ep_data
            data_narr += ep_data_narr
            logger.info("{} out of episode {} is finished".format(ep, episode))

        df = pd.DataFrame(data, columns=['student_id', 'step', 'state', 'action', 'reward', 'done', 'info'])
        df_narr = pd.DataFrame(data_narr, columns=['student_id', 'step', 'state', 'action', 'reward', 'done', 'info'])

        if self.args.dryrun is False:
            df.to_pickle('../simulated_data/' + self.args.run_name + '_sim.pkl')
            df_narr.to_pickle('../simulated_data/' + self.args.run_name + '_sim_narr.pkl')
        return df, df_narr

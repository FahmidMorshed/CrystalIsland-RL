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
    def __init__(self, args: dataclasses):
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

        # DUAL D
        self.d2 = Discriminator(self.args).to(self.args.device)
        self.optimizer_d2 = torch.optim.Adam(self.d2.parameters(), lr=self.args.lr_discriminator)
        self.lr_scheduler_d2 = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_d2,
                                                                     gamma=self.args.scheduler_gamma, verbose=False)

        # EVALUATOR
        self.d_eval = Discriminator(self.args).to(self.args.device)
        self.optimizer_d_eval = torch.optim.Adam(self.d_eval.parameters(), lr=self.args.lr_discriminator)
        self.lr_scheduler_d_eval = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_d_eval,
                                                                      gamma=self.args.scheduler_gamma, verbose=False)

        # define loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

        # create state and action tensors
        self.train_student_df, self.test_student_df, self.train_comp_df, self.test_comp_df = utils.load_student_data(self.args)

        self.expert_state_actions, self.expert_states, self.expert_actions = \
            self._gen_state_action_tensor(np.stack(self.train_student_df['state']),
                                          np.array(self.train_student_df['action']))

        # create env
        self.s0 = np.stack(self.train_student_df.loc[self.train_student_df['step'] == 0, 'state'])
        self.env = CrystalIsland(args, s0=self.s0)

        self.states = []
        self.actions = []
        self.log_prob_actions = []
        self.rewards = []
        self.is_terminal = []

        self.d_exp_score = 99999.0
        self.d_nov_score = 99990.0

        self.d_loss = 9999.0
        self.pi_loss = 9999.0


        # for test set only
        self.test_state_actions, self.test_states, self.test_actions = \
            self._gen_state_action_tensor(np.stack(self.test_student_df['state']),
                                          np.array(self.test_student_df['action']))

        self.d_test_score = 99999.0
        self.d_rand_score = 99999.0
        self.rand_actions = []
        self.rand_states = []
        self.s0_test = np.stack(self.test_student_df.loc[self.test_student_df['step'] == 0, 'state'])
        self.env_test = CrystalIsland(args, s0=self.s0_test)

        # for complement training only
        self.d_comp_score = 9999.0
        self.d_comp_nov_score = 9999.0
        self.d_comp_test_score = 9999.0

        self.comp_state_actions, self.comp_states, self.comp_actions = \
            self._gen_state_action_tensor(np.stack(self.train_comp_df['state']),
                                          np.array(self.train_comp_df['action']))

        # for testing comp
        self.comp_test_state_actions, self.comp_test_states, self.comp_test_actions = \
            self._gen_state_action_tensor(np.stack(self.test_comp_df['state']),
                                          np.array(self.test_comp_df['action']))

    def reset_buffers(self):
        self.states = []
        self.actions = []
        self.log_prob_actions = []
        self.rewards = []
        self.is_terminal = []

        # for test score
        self.rand_actions = []
        self.rand_states = []

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

    def random_action_simulate(self, step):
        state = self.env_test.reset()
        for i in range(step):
            state = torch.tensor(state, dtype=torch.float32, device=self.args.device)
            self.rand_states.append(state.detach())

            action = np.random.choice(range(0, self.args.action_dim))
            self.rand_actions.append(torch.tensor(action))

            state, reward, done, info = self.env_test.step(action)

            if done:
                state = self.env_test.reset()


    # TODO might need to implement different version of this
    def eval_gen_action_for_experts(self):
        if self.args.run_type != "eval":
            logger.error("You must call eval_gen_action_for_experts in eval mode!")
            exit(1)

        with torch.no_grad():
            gail_actions, gail_action_log_probs = self.pi_old.act(self.expert_states)
            expert_actions, expert_action_log_probs, _ = self.pi_old.evaluate(self.expert_states, self.expert_actions)
        gail_action_log_probs = gail_action_log_probs.detach()
        expert_action_log_probs = expert_action_log_probs.detach()

        imp_ratios = torch.exp(gail_action_log_probs - expert_action_log_probs).mean().detach()
        print(imp_ratios)
        # save_to_file(actions, os.path.join("trajectory", self.args.run_name + "_actions_solve.csv"))


    def train_evaluator(self):
        sample_size = 50000
        for ep in range(10000):
            self.reset_buffers()
            self.random_action_simulate(sample_size)
            rand_state_actions, _, _ = self._gen_state_action_tensor(self.rand_states, self.rand_actions)

            idx = np.random.choice(range(self.expert_state_actions.size()[0]), sample_size, replace=False)
            expert_state_actions = self.expert_state_actions[idx]

            expert_prob = self.d_eval(expert_state_actions)
            rand_prob = self.d_eval(rand_state_actions)
            term1 = self.bce_loss(rand_prob, torch.ones((rand_state_actions.shape[0], 1), device=self.args.device))
            term2 = self.bce_loss(expert_prob, torch.zeros((expert_state_actions.shape[0], 1), device=self.args.device))

            loss = term1 + term2
            curr_loss = loss.item()
            self.optimizer_d_eval.zero_grad()
            loss.backward()
            self.optimizer_d_eval.step()

            logger.info("training evaluator | step: {0} | current loss: {1: .4f} ".format(ep, curr_loss))
            if curr_loss < 0.1:
                break

        logger.info("finished training evaluator")
        self.reset_buffers()

    def eval(self):
        # clear buffers
        self.reset_buffers()

        sample_size = 5000
        # generate random samples
        self.random_action_simulate(sample_size)
        t = 0
        while t < sample_size:
            state = self.env.reset()
            ep_len = 0
            while ep_len < self.args.max_episode_len and t < sample_size:
                state, reward, done = self.take_action(state)
                t += 1
                if done:
                    break

        # original sim
        agent_state_actions, _, _ = self._gen_state_action_tensor(self.states, self.actions)
        y_sim = self.d_eval(agent_state_actions).detach()
        y_sim = (y_sim > .5)
        y_sim_truth = np.random.choice([0, 1], sample_size)
        print("Simulated with random truths: ", metrics.accuracy_score(y_sim_truth, y_sim))

        # original test
        idx = np.random.choice(range(self.test_state_actions.size()[0]), sample_size, replace=False)
        test_state_actions = self.test_state_actions[idx]
        y_test = self.d_eval(test_state_actions).detach()
        y_test = (y_test > .5)
        y_test_truth = np.random.choice([0, 1], sample_size)
        print("Test with random truths: ", metrics.accuracy_score(y_test_truth, y_test))

        # random with test
        rand_state_actions, _, _ = self._gen_state_action_tensor(self.rand_states, self.rand_actions)
        y_rand = self.d_eval(rand_state_actions).detach()
        y_rand = (y_rand > .5)
        y_rand_truth = np.ones(sample_size)
        print("Random with actual truths: ", metrics.accuracy_score(y_rand_truth, y_rand))

        y_test = self.d_eval(test_state_actions).detach()
        y_test = (y_test > .5)
        y_test_truth = np.zeros(sample_size)
        y_rand = self.d_eval(rand_state_actions).detach()
        y_rand = (y_rand > .5)
        y_rand_truth = np.ones(sample_size)
        y = torch.cat([y_rand, y_test], dim=0)
        y_truth = np.concatenate([y_rand_truth, y_test_truth])
        print("Test and Random with actual truths: ", metrics.accuracy_score(y_truth, y))

        self.reset_buffers()

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

        # for test score
        rand_state_actions, _, _ = self._gen_state_action_tensor(self.rand_states, self.rand_actions)

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

        # DUAL D
        for ep in range(self.args.internal_epoch_d):
            agent_prob = self.d2(agent_state_actions)
            term1 = self.bce_loss(agent_prob, torch.zeros((agent_state_actions.shape[0], 1), device=self.args.device))

            comp_prob = self.d2(self.comp_state_actions)
            term2 = self.bce_loss(comp_prob, torch.ones((self.comp_state_actions.shape[0], 1),
                                                        device=self.args.device))

            rand_prob = self.d2(rand_state_actions)
            term3 = self.bce_loss(rand_prob, torch.ones((rand_state_actions.shape[0], 1),
                                                        device=self.args.device))

            loss = term1 + term2 + term3
            curr_loss += loss.item()
            self.optimizer_d2.zero_grad()
            loss.backward()
            self.optimizer_d2.step()

        with torch.no_grad():
            d_rewards = -torch.log(self.d(agent_state_actions)) + (0.1 * torch.log(self.d2(agent_state_actions)))

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
        self.d_test_score = self.d(self.test_state_actions).mean().detach().item()
        self.d_rand_score = self.d(rand_state_actions).mean().detach().item()

        self.d_comp_score = self.d2(self.comp_state_actions).mean().detach().item()
        self.d_comp_nov_score = self.d2(agent_state_actions).mean().detach().item()
        self.d_comp_test_score = self.d2(self.comp_test_state_actions).mean().detach().item()

        self.reset_buffers()
        self.lr_scheduler_pi.step()
        self.lr_scheduler_d.step()
        self.lr_scheduler_d2.step()
        # manual decay
        self.args.clip_eps = self.args.clip_eps * self.args.scheduler_gamma

    def run(self):
        self.train_evaluator()
        t = 1
        success_count = 0
        update_count = 0
        finish = False
        while t <= self.args.train_steps:
            state = self.env.reset()
            total_reward = 0
            done = False
            ep_len = 0
            while ep_len < self.args.max_episode_len:
                state, reward, done = self.take_action(state)
                total_reward += reward
                if self.args.run_type == 'train' and t % self.args.update_steps == 0:
                    self.random_action_simulate(self.args.update_steps)
                    self.update()
                    update_count += 1

                    logger.info(
                        "iter: {0} | update: {6} | reward: {1:.1f} | d_loss: {2:.2f} | pi_loss: {3: .2f} | "
                        "d_exp: {4: .4f} | d_nov: {5: .4f} | d_test: {7: .4f} | d_rand: {8: .4f} | d_comp: {9: .4f} | "
                        "d_comp_nov: {10: .4f} | d_comp_test: {11: .4f}".format(
                            t, total_reward, self.d_loss, self.pi_loss, self.d_exp_score, self.d_nov_score,
                            update_count, self.d_test_score, self.d_rand_score, self.d_comp_score,
                            self.d_comp_nov_score, self.d_comp_test_score))

                    # check if conversed
                    if abs(self.d_exp_score - 0.5) <= self.args.d_stop_threshold and \
                            abs(self.d_nov_score - 0.5) <= self.args.d_stop_threshold:
                        success_count += 1
                        if success_count >= self.args.d_stop_count:
                            logger.info("model converged. saving checkpoint")
                            self.save()
                            finish = True
                    else:
                        success_count = 0

                t += 1
                ep_len += 1
                if done:

                    if not self.args.run_type == 'train':
                        self.reset_buffers()
                    break

            if not done:
                logger.debug("truncated at horizon")
            if finish:
                self.save()
                break

        self.eval()

    def save(self):
        torch.save(self.pi_old.state_dict(), "../checkpoint/policy.ckpt")
        torch.save(self.d.state_dict(), "../checkpoint/discriminator.ckpt")
        torch.save(self.d2.state_dict(), "../checkpoint/discriminator2.ckpt")
        torch.save(self.d_eval.state_dict(), "../checkpoint/eval.ckpt")

    def load(self):
        policy_model_path = "../checkpoint/policy.ckpt"
        self.pi_old.load_state_dict(torch.load(policy_model_path, map_location=lambda x, y: x))
        self.pi.load_state_dict(self.pi_old.state_dict())
        discriminator_model_path = "../checkpoint/discriminator.ckpt"
        self.d.load_state_dict(torch.load(discriminator_model_path, map_location=lambda x, y: x))
        discriminator_model_path2 = "../checkpoint/discriminator2.ckpt"
        self.d2.load_state_dict(torch.load(discriminator_model_path2, map_location=lambda x, y: x))
        evaluator_model_path = "../checkpoint/eval.ckpt"
        self.d_eval.load_state_dict(torch.load(evaluator_model_path, map_location=lambda x, y: x))

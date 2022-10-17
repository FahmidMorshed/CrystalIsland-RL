import dataclasses
from copy import deepcopy

import torch
import torch.nn as nn
import logging

from src.model.nets import PolicyModel, Discriminator

logger = logging.getLogger(__name__)


class GailExecutor:
    def __init__(self, args: dataclasses):
        self.args = args

        logger.info("args: {0}".format(self.args.to_json_string()))
        set_all_seeds(self.args.seed) #TODO

        self.pi = PolicyModel(self.args).to(self.args.device)
        self.pi_old = PolicyModel(self.args).to(self.args.device)
        self.pi_old.load_state_dict(self.pi.state_dict())
        self.d = Discriminator(self.args).to(self.args.device)

        self.optimizer_pi = torch.optim.Adam([
            {"params": self.pi.actor.parameters(), "lr": self.args.lr_actor},
            {"params": self.pi.critic.parameters(), "lr": self.args.lr_critic}
        ])
        self.optimizer_d = torch.optim.Adam(self.d.parameters(), lr=self.args.lr_discriminator)

        self.lr_scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_d, gamma=0.96,
                                                                     verbose=True)
        self.lr_scheduler_pi = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_pi, gamma=0.96, verbose=True)

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

        # TODO
        expert_states = np.genfromtxt("trajectory/" + self.args.run_name + "_states.csv")
        self.expert_states = torch.tensor(expert_states, dtype=torch.float32, device=self.args.device)
        expert_actions = np.genfromtxt("trajectory/" + self.args.run_name + "_actions.csv", dtype=np.int32)
        self.expert_actions = torch.tensor(expert_actions, dtype=torch.int64, device=self.args.device)
        expert_actions = torch.eye(self.args.num_actions)[self.expert_actions].to(self.args.device)
        self.expert_state_actions = torch.cat([self.expert_states, expert_actions], dim=1)

        # TODO load state_action_map

        self.states = []
        self.actions = []
        self.log_prob_actions = []
        self.rewards = []
        self.is_terminal = []

        self.d_loss = 999999.0
        self.p_loss = 999999.0
        self.d_val_mean_exp = 99999.0
        self.d_val_mean_nov = 99990.0

    def take_step(self, state, action):
        next_state = deepcopy(state)
        next_state[self.action_state_map[action]] += 1
        done = False
        reward = 0
        if self.action_state_map[action] == 's_end_game':
            done = True
        return next_state, reward, done

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

    def update(self):
        prev_states = torch.stack(self.states, dim=0).to(self.args.device)
        prev_actions = torch.stack(self.actions, dim=0).to(self.args.device)
        prev_log_prob_actions = torch.stack(self.log_prob_actions, dim=0).to(self.args.device)
        prev_actions_one_hot = torch.eye(self.args.num_actions)[prev_actions.long()].to(self.args.device)
        agent_state_actions = torch.cat([prev_states, prev_actions_one_hot], dim=1)

        curr_loss = 0.0
        for ep in range(self.args.num_d_epochs):
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

        self.d_loss = curr_loss / self.args.num_d_epochs

        with torch.no_grad():
            d_rewards = -torch.log(self.d(agent_state_actions))

        rewards = []
        cumulative_discounted_reward = 0.
        for d_reward, terminal in zip(reversed(d_rewards), reversed(self.is_terminal)):
            if terminal:
                cumulative_discounted_reward = 0
            cumulative_discounted_reward = d_reward + (self.args.discount_factor * cumulative_discounted_reward)
            rewards.insert(0, cumulative_discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.args.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        curr_loss = 0.0
        for ep in range(self.args.num_epochs):
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
        self.p_loss = curr_loss / self.args.num_epochs

        self.pi_old.load_state_dict(self.pi.state_dict())

        self.d_val_mean_nov = self.d(agent_state_actions).mean().detach().item()
        self.d_val_mean_exp = self.d(self.expert_state_actions).mean().detach().item()

        self.reset_buffers()
        self.lr_scheduler_pi.step()
        self.lr_scheduler_d.step()
        self.args.clip_eps = self.args.clip_eps * 0.99

    def run(self):
        t = 1
        success_count = 0
        finish = False
        record = []
        while t <= self.args.train_steps:
            state = self.env.reset()
            total_reward = 0
            done = False
            ep_len = 0
            while ep_len < self.args.max_episode_len:
                state, reward, done = self.take_action(state)
                total_reward += reward
                if self.args.run_type == 'train' and t % self.args.update_steps == 0:
                    logger.info("updating model")
                    self.update()

                    # if total_reward >= self.args.reward_threshold:
                    if abs(self.d_val_mean_exp - self.d_val_mean_nov) <= 0.10:
                        success_count += 1
                        if success_count >= 5:
                            logger.info("model trained. saving checkpoint")
                            self.save(self.args.checkpoint_dir)
                            finish = True
                    else:
                        success_count = 0

                if self.args.run_type == 'train' and t % self.args.checkpoint_steps == 0:
                    logger.info("saving checkpoint")
                    self.save(self.args.checkpoint_dir)
                t += 1
                ep_len += 1
                if done:
                    logger.info(
                        "iter: {0} | reward: {1:.1f} | d_loss: {2:.2f} | p_loss: {3: .4f} | d_val_mean_exp: {4: .2f} "
                        "| d_val_mean_nov: {5: .2f}".format(
                            t, total_reward, self.d_loss, self.p_loss, self.d_val_mean_exp, self.d_val_mean_nov))
                    if not self.args.run_type == 'train':
                        self.reset_buffers()
                    break
            record.append((ep_len, total_reward))

            if not done:
                logger.info(
                    "truncated at horizon | iter: {0} | reward: {1:.1f} | d_loss: {2:.2f} | p_loss: {3: .4f} | "
                    "d_val_mean_exp: {4: .2f} "
                    "| d_val_mean_nov: {5: .2f}".format(
                        t, total_reward, self.d_loss, self.p_loss, self.d_val_mean_exp, self.d_val_mean_nov))
            if finish:
                break
        with open("{0}/record.pkl".format(self.args.checkpoint_dir), "wb") as handle:
            pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save(self, checkpoint_dir):
        torch.save(self.pi_old.state_dict(), "{0}/policy.ckpt".format(checkpoint_dir))
        torch.save(self.d.state_dict(), "{0}/discriminator.ckpt".format(checkpoint_dir))

    def load(self, checkpoint_dir):
        policy_model_path = "{0}/policy.ckpt".format(checkpoint_dir)
        self.pi_old.load_state_dict(torch.load(policy_model_path, map_location=lambda x, y: x))
        self.pi.load_state_dict(self.pi_old.state_dict())
        discriminator_model_path = "{0}/discriminator.ckpt".format(checkpoint_dir)
        self.d.load_state_dict(torch.load(discriminator_model_path, map_location=lambda x, y: x))


def gen_trajectories(args):
    env = gym.make(args.env_id)
    env.seed(args.seed)
    args.state_dim = env.observation_space.shape[0]
    args.num_actions = env.action_space.n

    # loading expert model
    expert = Expert(
        args.state_dim, args.num_actions)
    expert.pi.load_state_dict(
        torch.load("expert/policy.ckpt")
    )

    total_step = 0
    total_reward = 0

    logs = []
    states = []
    actions = []
    for ep in range(args.gen_total_ep):
        ep_step = 0
        ep_reward = 0
        ob = env.reset()
        env.seed(ep)

        while True:
            if args.render:
                env.render()

            act = expert.act(ob)
            if np.random.random() < args.gen_poison:
                act = 1

            next_ob, rwd, done, info = env.step(act)
            ep_reward += rwd

            step_log = {'total_step': total_step, 'episode': ep, 'episode_step': ep_step, 'state': ob,
                        'action': act,
                        'next_state': next_ob, 'reward': rwd, 'done': done, 'episode_reward': ep_reward}
            logs.append(step_log)

            states.append(ob)
            actions.append(act)

            ep_step += 1
            total_step += 1
            ob = next_ob

            if ep_step >= args.max_episode_len or done:
                total_reward += ep_reward
                print(f"Expert Reward for Episode {ep}: {ep_reward}. Total Steps in Episode: {ep_step}")
                break
    print(f"Mean Expert Reward: {total_reward / args.gen_total_ep}. Mean Steps in Episode: {total_step / args.gen_total_ep}")

    df = pd.DataFrame(logs)
    df.to_pickle(os.path.join("trajectory/detailed_run", args.run_name + ".pkl"))

    save_to_file(states, os.path.join("trajectory", args.run_name + "_states.csv"))
    save_to_file(actions, os.path.join("trajectory", args.run_name + "_actions.csv"))

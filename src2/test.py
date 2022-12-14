"""This is a simple example demonstrating how to clone the behavior of an expert.

Refer to the jupyter notebooks for more detailed examples of how to use the algorithms.
"""

import gym
import numpy as np
import pandas as pd
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data.types import Transitions
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper

import src.env.crystalisland
import src.utils
rng = np.random.default_rng(0)
df = pd.read_pickle("../processed_data/student_trajectories.pkl")
transitions = src.utils.get_transitions(df)
env = Monitor(src.env.crystalisland.CrystalIsland())

venv = make_vec_env(src.env.crystalisland.CrystalIsland, n_envs=8)
learner = PPO(env=venv, policy=MlpPolicy)
reward_net = BasicRewardNet(
    env.observation_space,
    env.action_space,
    normalize_input_layer=RunningNorm,
)
gail_trainer = GAIL(
    demonstrations=transitions,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=4,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
)

gail_trainer.train(100000)
rewards, _ = evaluate_policy(learner, venv, 100, return_episode_rewards=True)
print("Rewards:", rewards)


#
# rng = np.random.default_rng(0)
#
# bc_trainer = bc.BC(
#     observation_space=env.observation_space,
#     action_space=env.action_space,
#     demonstrations=transitions,
#     rng=rng,
# )
#
# reward, _ = evaluate_policy(
#     bc_trainer.policy,  # type: ignore[arg-type]
#     env,
#     n_eval_episodes=3,
#     render=True,
# )
# print(f"Reward before training: {reward}")
#
# print("Training a policy using Behavior Cloning")
# bc_trainer.train(n_epochs=2)
#
# reward, _ = evaluate_policy(
#     bc_trainer.policy,  # type: ignore[arg-type]
#     env,
#     n_eval_episodes=3,
#     render=True,
# )
# print(f"Reward after training: {reward}")




def dummy():
    args = utils.parse_config(ModelArguments, external_config)
    args.seed = 7
    utils.set_all_seeds(args.seed)
    args.run_name += "_" + str(args.seed)
    train, test = utils.load_data(args.student_data_loc, test_size=.2)

    rng = np.random.default_rng(0)
    transitions = utils.get_transitions(train)

    venv = make_vec_env(get_env, n_envs=8)
    print("virtual env done")
    learner = PPO(env=venv, policy=MlpPolicy)
    reward_net = BasicRewardNet(
        venv.observation_space,
        venv.action_space,
        normalize_input_layer=RunningNorm,
    )
    gail_trainer = GAIL(
        demonstrations=transitions,
        demo_batch_size=1024,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=4,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
    )

    gail_trainer.train(100000)
    rewards, _ = evaluate_policy(learner, venv, 100, return_episode_rewards=True)
    print("Rewards:", rewards)

def get_env():
    args = utils.parse_config(ModelArguments, external_config)
    args.seed = 7
    utils.set_all_seeds(args.seed)
    args.run_name += "_" + str(args.seed)
    train, test = utils.load_data(args.student_data_loc, test_size=.2)

    anomaly_detector = utils.get_anomaly_detector(train, test)
    action_probs = utils.get_action_probs()
    clf = utils.solve_predictor(train, args.seed, print_eval=False)
    env = CrystalIsland(solution_predictor=clf, action_probs=action_probs, anomaly_detector=anomaly_detector)
    return env
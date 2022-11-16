import dataclasses
import logging
import os.path
from collections import deque
from copy import deepcopy

import torch
import numpy as np
import pandas as pd

from src import utils
from src.model.rl import BCQ, RANDOM
from src.model.crystalisland import CrystalIsland
from src.model.rl import FQE
from src.model.gail import GailExecutor
from src.model.rewardmodel import OutcomePredictor
from src.model.validator import SimpleValidator
from src.model_args import ModelArguments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

external_config = {
    'dryrun': False,
    # 'simulate_episodes': 1000,
    'run_name': 'nov14_1500',
    # 'train_steps': 1000,
    # 'update_frequency': 100,
    # 'log_frequency': 100,
    # 'gail_train_steps': 100,
}


def full_run():
    args = utils.parse_config(ModelArguments, external_config)
    results = []
    for seed in range(10):
        logger.info("+" * 30)
        logger.info("===== starting run with seed {0} =====".format(seed))

        args.seed = seed
        args.run_name = "seed_" + str(seed)
        utils.set_all_seeds(args.seed)

        train_narr, test_narr, s0_narr = utils.load_data(args.narrative_data_loc, test_size=.5)

        sim_narr_loc = '../simulated_data/' + args.run_name + '_sim_narr.pkl'
        if os.path.isfile(sim_narr_loc):
            sim_narr = pd.read_pickle(sim_narr_loc)
        else:
            # generate simulated students
            train_student = train_narr['episode'].unique()
            test_student = test_narr['episode'].unique()

            train, test, s0 = utils.load_data(args.student_data_loc, train_student=train_student,
                                              test_student=test_student)
            validator = SimpleValidator(args, train, test, s0, name="org")
            validator.train()

            env = CrystalIsland(args, s0)
            gail = GailExecutor(args, train, env, "org")
            gail.train()
            _, sim_narr = gail.simulate(args.simulate_episodes, validator)

        logger.info("-- training fqe for evaluation --")
        fqe = FQE(args, test_narr)
        fqe.train()

        logger.info("-- training narrative planner with org data --")
        bcq_org = BCQ(args, train_narr, fqe, "org")
        bcq_org.train_behavior_cloning()
        bcq_org.train()
        ecr, dm, isamp, wisamp, dr = bcq_org.print_logs(0, 0)
        results.append({"seed": seed, "name": "ori", "direct_method": dm, "importance_sampling": isamp,
                        "weighted_importance_sampling": wisamp, "doubly_robust": dr})

        logger.info("-- training narrative planner with sim data --")
        bcq_sim = BCQ(args, sim_narr, fqe, "sim")
        bcq_sim.train_behavior_cloning()
        bcq_sim.train()
        ecr, dm, isamp, wisamp, dr = bcq_sim.print_logs(0, 0)
        results.append({"seed": seed, "name": "sim", "direct_method": dm, "importance_sampling": isamp,
                        "weighted_importance_sampling": wisamp, "doubly_robust": dr})

        random_policy = RANDOM(args, train_narr, fqe)
        ecr, dm, isamp, wisamp, dr = random_policy.print_logs()
        results.append({"seed": seed, "name": "rand_ori", "direct_method": dm, "importance_sampling": isamp,
                        "weighted_importance_sampling": wisamp, "doubly_robust": dr})

        random_policy = RANDOM(args, sim_narr, fqe)
        ecr, dm, isamp, wisamp, dr = random_policy.print_logs()
        results.append({"seed": seed, "name": "rand_sim", "direct_method": dm, "importance_sampling": isamp,
                        "weighted_importance_sampling": wisamp, "doubly_robust": dr})



def save_results(results: list):
    r = pd.DataFrame(results, columns=["seed", "name", "direct_method", "importance_sampling",
                                       "weighted_importance_sampling", "doubly_robust"])
    save_results(r)
    r_sum = r.groupby(['name']).describe()
    r_sum = r_sum[['direct_method', 'importance_sampling', 'weighted_importance_sampling', 'doubly_robust']]
    r_sum = r_sum.drop(['count', 'min', 'max', '25%', '75%'], axis=1, level=1)
    r_sum = r_sum.round(2)
    r_sum = r_sum.reindex(["sim", "ori", "rand_sim", "rand_ori"])
    r_sum.to_csv("../temp/results_nov_14_sum.csv")
    r.to_csv("../temp/results_nov_14.csv")

def dummy():
    args = utils.parse_config(ModelArguments, external_config)
    utils.set_all_seeds(args.seed)

    train, test, s0 = utils.load_data(args.student_data_loc, test_size=.2)

    reward_model = OutcomePredictor(args, train, test)
    reward_model.train()


if __name__ == "__main__":
    dummy()


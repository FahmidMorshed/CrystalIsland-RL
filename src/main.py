import dataclasses
import logging
import os.path
from copy import deepcopy

import torch
import numpy as np
import pandas as pd

from src import utils
from src.model.bcq import BCQ
from src.model.crystalisland import CrystalIsland
from src.model.fqe import FQE
from src.model.gail import GailExecutor
from src.model.validator import SimpleValidator
from src.model_args import ModelArguments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

external_config = {
    'bcq_train_steps': int(1e5),
    'gail_train_steps': 1000,
    'validator_train_steps': 1000,
    'dryrun': False,
    'simulate_episodes': 1000,
    'run_name': 'test',
    'load_validator': False,
    'load_gail': False,
    'load_sim': True,
    'fqe_train_steps': int(1e5),
    'behavior_cloning_train_steps': int(1e5),
    'load_fqe': True,
    'load_bc': False,
}


def full_run():
    args = utils.parse_config(ModelArguments, external_config)
    for seed in range(10):
        logger.info("+" * 30)
        logger.info("===== starting run with seed {0} =====".format(seed))

        args.seed = seed
        args.run_name = "seed_" + str(seed)
        utils.set_all_seeds(args.seed)

        train_narr, test_narr, s0_narr = utils.load_data(args.narrative_data_loc, test_size=.5)

        logger.info("-- training fqe for evaluation --")
        fqe = FQE(args, test_narr, 'fqe_test')
        fqe.train()

        logger.info("-- training narrative planner with original data --")
        bcq = BCQ(args, train_narr, test_narr, fqe, 'org')
        bcq.train_behavior_cloning()
        bcq.train()
        bcq.print_logs(1, 0)
        bcq.print_logs(1, 0)
        bcq.print_logs(1, 0)

        sim_narr_loc = '../simulated_data/'+args.run_name + '_sim_narr.pkl'
        if os.path.isfile(sim_narr_loc):
            sim_narr = pd.read_pickle(sim_narr_loc)
        else:
            # generate simulated students
            train_student = train_narr['student_id'].unique()
            test_student = test_narr['student_id'].unique()
            train, test, s0 = utils.load_data(args.student_data_loc, train_student=train_student,
                                              test_student=test_student)
            validator = SimpleValidator(args, train, test, s0, name="org")
            validator.train()

            env = CrystalIsland(args, s0)
            gail = GailExecutor(args, train, env, "org")
            gail.train()
            _, sim_narr = gail.simulate(args.simulate_episodes, validator)

        logger.info("-- training narrative planner with sim data --")
        bcq_sim = BCQ(args, sim_narr, test_narr, fqe, 'sim')
        bcq_sim.train_behavior_cloning()
        bcq_sim.train()
        bcq_sim.print_logs(1, 0)
        bcq_sim.print_logs(1, 0)
        bcq_sim.print_logs(1, 0)

        # combine simulated and training data and shuffle
        comb_narr = pd.concat([train_narr, sim_narr], axis=0)
        student_ids = comb_narr['student_id'].unique()
        comb_narr = comb_narr.set_index("student_id").loc[student_ids].reset_index()

        logger.info("-- training narrative planner with combined data --")
        bcq_comb = BCQ(args, comb_narr, test_narr, fqe, 'comb')
        bcq_comb.train_behavior_cloning()
        bcq_comb.train()
        bcq_comb.print_logs(1, 0)
        bcq_comb.print_logs(1, 0)
        bcq_comb.print_logs(1, 0)


def main():
    if external_config['debug']:
        external_config['dryrun'] = True

    args = utils.parse_config(ModelArguments, external_config)
    utils.set_all_seeds(args.seed)

if __name__ == "__main__":
    full_run()

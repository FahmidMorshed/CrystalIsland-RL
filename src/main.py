import dataclasses
import logging
from copy import deepcopy

import torch
import numpy as np
import pandas as pd

from src import utils
from src.model.bcq import BCQ
from src.model.crystalisland import CrystalIsland
from src.model.gail import GailExecutor
from src.model.validator import Validator
from src.model_args import ModelArguments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


external_config = {
    'bcq_train_steps': 1000000,
    'gail_train_steps': 1000,
    'validator_train_steps': 1000,
    'dryrun': False,
    'simulate_episodes': 1000,
    'run_name': 'test',
    'load_validator': False,
    'load_gail': False,
    'load_sim': True,
}

def main():
    args = utils.parse_config(ModelArguments, external_config)
    utils.set_all_seeds(args.seed)

    if args.load_sim is False:
        train, test, s0 = utils.load_data(args.student_data_loc)
        validator = Validator(args, train, test, s0)
        if args.load_validator is False:
            # train the validator
            logger.info("-- training validator --")
            validator.train()
        else:
            logger.info("-- load validator --")
            validator.load()

        env = CrystalIsland(args, s0)
        gail = GailExecutor(args, train, env)
        if args.load_gail is False:
            # train gail and simulate data
            logger.info("-- training gail --")
            gail.train()
        else:
            logger.info("-- training gail --")
            gail.load()

        _, sim_narr = gail.simulate(args.simulate_episodes, validator)
    else:
        sim_narr = pd.read_pickle('../simulated_data/' + args.run_name + '_sim_narr.pkl')

    # training bcq with original data
    # logger.info("-- training bcq with original data --")
    # train_narr, test_narr, s0_narr = utils.load_data(args.narrative_data_loc, test_size=.2)
    # bcq = BCQ(args, train_narr, test_narr, s0_narr)
    # bcq.train()
    #
    # training bcq with simulated data
    # logger.info("-- training bcq with sim data --")
    # bcq_sim = BCQ(args, sim_narr, test_narr, s0_narr)
    # bcq_sim.train()

    # random
    train_narr, test_narr, s0_narr = utils.load_data(args.narrative_data_loc, test_size=.2)
    sim = pd.read_pickle('../simulated_data/' + args.run_name + '_sim.pkl')
    
    print(len(sim))





if __name__ == "__main__":
    main()

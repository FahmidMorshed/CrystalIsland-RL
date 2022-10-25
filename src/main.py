import dataclasses
import logging

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


def main():
    args = ModelArguments()
    utils.set_all_seeds(args.seed)

    train, test, s0 = utils.load_data(args.student_data_loc)
    a = np.stack(test['state'])
    # validator = Validator(args, train, test, s0)
    # validator.train()
    #
    # train_high, train_low, test_high, test_low = utils.split_by_nlg(train, test)
    # env = CrystalIsland(args, s0)
    # gail = GailExecutor(args, train_high, env)
    # gail.train()
    # df, df_narr = gail.simulate(100)
    # result = validator.validate_df(df)
    # result.to_csv('../simulated_data/' + args.run_name + '_result.csv')

    # testing random things
    train, test, s0 = utils.load_data(args.narrative_data_loc)
    bcq = BCQ(args, train)
    bcq.train()

if __name__ == "__main__":
    main()

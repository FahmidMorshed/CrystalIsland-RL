import dataclasses
import logging

import torch
import numpy as np
import pandas as pd

from src import utils
from src.model.crystalisland import CrystalIsland
from src.model.gail import GailExecutor
from src.model.validator import Validator
from src.model_args import ModelArguments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    args = ModelArguments()
    utils.set_all_seeds(args.seed)
    train_high, train_low, test_high, test_low, s0 = utils.load_student_data_by_nlg(args)
    env = CrystalIsland(args, s0)
    gail = GailExecutor(args, train_low, env)
    gail.run(total_updates=20, dryrun=True)
    # df, df_narr = gail.simulate(5, save=True, filename='test')
    # train, test, s0 = utils.load_student_data(args)



if __name__ == "__main__":
    main()

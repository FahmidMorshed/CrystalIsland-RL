import dataclasses
import logging

import torch
import numpy as np
import pandas as pd

from src import utils
from src.model.crystalisland import CrystalIsland
from src.model.gail import GailExecutor
from src.model_args import ModelArguments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    args = ModelArguments()
    utils.set_all_seeds(args.seed)
    train_low, train_high, test_low, test_high, s0 = utils.load_student_data(args)
    env = CrystalIsland(args, s0)
    gail = GailExecutor(args, train_low, env)
    gail.run()

    # print(env.gen_random_data(50))


if __name__ == "__main__":
    main()

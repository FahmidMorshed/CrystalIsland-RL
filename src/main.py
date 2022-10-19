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
    gail = GailExecutor(args)
    gail.load()
    gail.eval()
    # conservative gail tryout: we are trying to produce behaviors that is close to pi_1 (say low nlg students)
    # and far from pi_2 (high nlg students).


if __name__ == "__main__":
    main()

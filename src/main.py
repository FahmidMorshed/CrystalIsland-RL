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
    # env = CrystalIsland(args, s0)
    # gail = GailExecutor(args, train_low, env)
    # gail.run()
    train, test, s0 = utils.load_student_data(args)
    validator = Validator(args)
    validator.load()

    d = train.loc[train['student_id']=='100-0086']
    states = np.stack(d['state'])
    actions = np.array(d['action'])

    val_df = validator.validate_df(test)
    print()
    a = test.groupby('student_id').last()['nlg'].reset_index()
    a.loc[a['nlg'] == 100, 'nlg'] = True
    a.loc[a['nlg'] == -100, 'nlg'] = False
    val_df = val_df.merge(a, on='student_id')
    print(len(val_df))
    print("NLG Wrong:", len(val_df.loc[(val_df['is_high'] != val_df['nlg'])])/len(val_df))
    print("Auth Wrong:", len(val_df.loc[(val_df['is_authentic']==False)])/len(val_df))



if __name__ == "__main__":
    main()

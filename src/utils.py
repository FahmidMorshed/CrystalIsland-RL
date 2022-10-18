import dataclasses
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


def load_student_data(args: dataclasses):
    df = pd.read_pickle(args.student_data_loc)
    complementary_df = None
    if args.split_by == 'high':
        high_nlg_students = df.loc[df['nlg'] == 100]['student_id'].unique()
        complement_df = df.loc[~df['student_id'].isin(high_nlg_students)].reset_index(drop=True)
        df = df.loc[df['student_id'].isin(high_nlg_students)].reset_index(drop=True)

    elif args.split_by == 'low':
        low_nlg_students = df.loc[df['nlg'] == -100]['student_id'].unique()
        complement_df = df.loc[~df['student_id'].isin(low_nlg_students)].reset_index(drop=True)
        df = df.loc[df['student_id'].isin(low_nlg_students)].reset_index(drop=True)

    train_student, test_student = train_test_split(df['student_id'].unique(), test_size=0.2)

    train_df = df.loc[df['student_id'].isin(train_student)].reset_index(drop=True)
    test_df = df.loc[df['student_id'].isin(test_student)].reset_index(drop=True)

    # shuffle by student_id
    train_df = train_df.set_index("student_id").loc[train_student].reset_index()
    test_df = test_df.set_index("student_id").loc[test_student].reset_index()

    return train_df, test_df, complement_df


def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

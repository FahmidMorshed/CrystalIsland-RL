import dataclasses
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


def load_student_data_by_nlg(args: dataclasses):
    df = pd.read_pickle(args.student_data_loc)

    d = df.loc[df['done']]
    student_ids = d['student_id'].tolist()
    nlgs = d['nlg'].tolist()

    train_student, test_student = train_test_split(student_ids, test_size=0.2, stratify=nlgs)

    train_df = df.loc[df['student_id'].isin(train_student)].reset_index(drop=True)
    test_df = df.loc[df['student_id'].isin(test_student)].reset_index(drop=True)

    train_high_student = df.loc[(df['student_id'].isin(train_student)) & (df['done']) & (df['nlg'] == 100)][
        'student_id']
    train_low_student = df.loc[(df['student_id'].isin(train_student)) & (df['done']) & (df['nlg'] == -100)][
        'student_id']
    train_df_high = df.loc[df['student_id'].isin(train_high_student)].reset_index(drop=True)
    train_df_low = df.loc[df['student_id'].isin(train_low_student)].reset_index(drop=True)

    # shuffle
    train_df_high = train_df_high.set_index("student_id").loc[train_high_student].reset_index()
    train_df_low = train_df_low.set_index("student_id").loc[train_low_student].reset_index()

    test_high_student = df.loc[(df['student_id'].isin(test_student)) & (df['done']) & (df['nlg'] == 100)][
        'student_id']
    test_low_student = df.loc[(df['student_id'].isin(test_student)) & (df['done']) & (df['nlg'] == -100)][
        'student_id']
    test_df_high = df.loc[df['student_id'].isin(test_high_student)].reset_index(drop=True)
    test_df_low = df.loc[df['student_id'].isin(test_low_student)].reset_index(drop=True)
    # shuffle
    test_df_high = test_df_high.set_index("student_id").loc[test_high_student].reset_index()
    test_df_low = test_df_low.set_index("student_id").loc[test_low_student].reset_index()

    s0 = np.stack(df.loc[df['step'] == 0, 'state'])
    return train_df_high, train_df_low, test_df_high, test_df_low, s0


def load_student_data(args: dataclasses):
    df = pd.read_pickle(args.student_data_loc)
    s0 = np.stack(df.loc[df['step'] == 0, 'state'])

    d = df.loc[df['done']]
    student_ids = d['student_id'].tolist()
    nlgs = d['nlg'].tolist()

    train_student, test_student = train_test_split(student_ids, test_size=0.2, stratify=nlgs)

    train_df = df.loc[df['student_id'].isin(train_student)].reset_index(drop=True)
    test_df = df.loc[df['student_id'].isin(test_student)].reset_index(drop=True)

    # shuffle
    train_df = train_df.set_index("student_id").loc[train_student].reset_index()
    test_df = test_df.set_index("student_id").loc[test_student].reset_index()

    return train_df, test_df, s0


def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def state_action_tensor(states, actions,  action_dim, device='cpu',):
    if type(states) is np.ndarray:
        state_tensor = torch.tensor(states, dtype=torch.float32, device=device)
        action_tensor = torch.tensor(actions, dtype=torch.int64, device=device)
    else:
        state_tensor = torch.stack(states, dim=0).to(device)
        action_tensor = torch.stack(actions, dim=0).to(device)
    actions_one_hot = torch.eye(action_dim)[action_tensor.long()].to(device)
    sa_tensor = torch.cat([state_tensor, actions_one_hot], dim=1)
    return sa_tensor, state_tensor, action_tensor

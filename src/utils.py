import dataclasses
import logging
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_student_data_by_nlg(args: dataclasses):
    df = pd.read_pickle(args.student_data_loc)

    d = df.loc[df['done']]
    student_ids = d['student_id'].tolist()
    nlgs = d['reward'].tolist()

    train_student, test_student = train_test_split(student_ids, test_size=0.2, stratify=nlgs)

    train_df = df.loc[df['student_id'].isin(train_student)].reset_index(drop=True)
    test_df = df.loc[df['student_id'].isin(test_student)].reset_index(drop=True)

    train_high_student = df.loc[(df['student_id'].isin(train_student)) & (df['done']) & (df['reward'] == 100)][
        'student_id']
    train_low_student = df.loc[(df['student_id'].isin(train_student)) & (df['done']) & (df['reward'] == -100)][
        'student_id']
    train_df_high = df.loc[df['student_id'].isin(train_high_student)].reset_index(drop=True)
    train_df_low = df.loc[df['student_id'].isin(train_low_student)].reset_index(drop=True)

    # shuffle
    train_df_high = train_df_high.set_index("student_id").loc[train_high_student].reset_index()
    train_df_low = train_df_low.set_index("student_id").loc[train_low_student].reset_index()

    test_high_student = df.loc[(df['student_id'].isin(test_student)) & (df['done']) & (df['reward'] == 100)][
        'student_id']
    test_low_student = df.loc[(df['student_id'].isin(test_student)) & (df['done']) & (df['reward'] == -100)][
        'student_id']
    test_df_high = df.loc[df['student_id'].isin(test_high_student)].reset_index(drop=True)
    test_df_low = df.loc[df['student_id'].isin(test_low_student)].reset_index(drop=True)
    # shuffle
    test_df_high = test_df_high.set_index("student_id").loc[test_high_student].reset_index()
    test_df_low = test_df_low.set_index("student_id").loc[test_low_student].reset_index()

    s0 = np.stack(df.loc[df['step'] == 0, 'state'])
    return train_df_high, train_df_low, test_df_high, test_df_low, s0


def load_data(df_location, test_size=0.2, train_student=None, test_student=None):
    logger.info("-- loading data from {0} with test size {1} --".format(df_location, test_size))
    df = pd.read_pickle(df_location)
    a = (df.groupby(['action']).count() / len(df))[['step']]
    a.rename(columns={'step': 'action_prob'}, inplace=True)
    df = df.merge(a, on=['action'], how='left')

    s0 = np.stack(df.groupby('student_id').first()['state'])

    if train_student is None and test_student is None:
        d = df.loc[df['done']]
        student_ids = d['student_id'].tolist()
        nlgs = d['reward'].tolist()

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


def state_action_tensor(states, actions, action_dim, device='cpu'):
    if type(states) is np.ndarray:
        state_tensor = torch.tensor(states, dtype=torch.float32, device=device)
        action_tensor = torch.tensor(actions, dtype=torch.int64, device=device)
    else:
        state_tensor = torch.stack(states, dim=0).to(device)
        action_tensor = torch.stack(actions, dim=0).to(device)
    actions_one_hot = torch.eye(action_dim)[action_tensor.long()].to(device)
    sa_tensor = torch.cat([state_tensor, actions_one_hot], dim=1)
    return sa_tensor, state_tensor, action_tensor


def split_by_reward(df_train, df_test):
    # for train data
    train_high_student = df_train.loc[(df_train['done']) & (df_train['reward'] == 100)]['student_id'].unique()
    train_low_student = df_train.loc[(df_train['done']) & (df_train['reward'] == -100)]['student_id'].unique()

    train_df_high = df_train.loc[(df_train['student_id'].isin(train_high_student))]
    train_df_low = df_train.loc[(df_train['student_id'].isin(train_low_student))]

    # shuffle
    train_df_high = train_df_high.set_index("student_id").loc[train_high_student].reset_index()
    train_df_low = train_df_low.set_index("student_id").loc[train_low_student].reset_index()

    # for test data
    test_high_student = df_test.loc[(df_test['done']) & (df_test['reward'] == 100)]['student_id'].unique()
    test_low_student = df_test.loc[(df_test['done']) & (df_test['reward'] == -100)]['student_id'].unique()

    test_df_high = df_test.loc[(df_test['student_id'].isin(test_high_student))]
    test_df_low = df_test.loc[(df_test['student_id'].isin(test_low_student))]

    # shuffle
    test_df_high = test_df_high.set_index("student_id").loc[test_high_student].reset_index()
    test_df_low = test_df_low.set_index("student_id").loc[test_low_student].reset_index()

    return train_df_high, train_df_low, test_df_high, test_df_low


def parse_config(args_class, external_config):
    keys = {f.name for f in dataclasses.fields(args_class)}
    inputs = {k: v for k, v in external_config.items() if k in keys}
    return args_class(**inputs)


def gen_buffer_tensor(df, device="cpu") -> dict:
    """
    creates tensor for state, action, reward, next_state, done and action-musk
    :return: returns a dictionary of five tensor sets
    """
    states = np.stack(df['state'])
    actions = np.array(df['action'])
    rewards = np.array(df['reward'])
    dones = np.array(df['done'])

    next_states = np.stack(df['state'][1:])
    next_states = np.vstack([next_states, np.zeros(next_states.shape[1])])
    idx = np.where(dones == True)
    next_states[idx] = np.zeros(next_states.shape[1])

    state_tensor = torch.tensor(states, dtype=torch.float32, device=device)
    action_tensor = torch.tensor(actions, dtype=torch.int64, device=device)
    reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_state_tensor = torch.tensor(next_states, dtype=torch.float32, device=device)
    done_tensor = torch.tensor(dones, dtype=torch.float32, device=device)

    musk_tensor = musk_tensor_np_next_state_actions(next_state_tensor)

    ret_dict = {'state': state_tensor, 'action': action_tensor, 'reward': reward_tensor,
                'next_state': next_state_tensor, 'done': done_tensor, 'musk': musk_tensor}
    return ret_dict


def musk_tensor_np_next_state_actions(next_states: torch.Tensor, device="cpu"):
    action_triggers = next_states[:, -4:].tolist()  # last four features are trigger marker
    musk = []
    for row in action_triggers:
        allowed_action = [0.] * 10
        if row == [0., 0., 1., 0.]:  # knowledge quiz
            allowed_action[5] = 1.
            allowed_action[6] = 1.
        elif row == [0., 1., 0., 0.]:  # teresa
            allowed_action[2] = 1.
            allowed_action[3] = 1.
            allowed_action[4] = 1.
        elif row == [1., 0., 0., 0.]:  # bryce
            allowed_action[0] = 1.
            allowed_action[1] = 1.
        elif row == [0., 0., 0., 1.]:  # diagnosis
            allowed_action[7] = 1
            allowed_action[8] = 1
            allowed_action[9] = 1

        musk.append(allowed_action)
    return torch.tensor(musk, dtype=torch.float32, device=device)


def sample_buffer_tensor(buffer_tensor: dict, sample_size: int) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    """
    sample transactions of given batch size in tensor format
    :return: five tensors that contains samples of state, action, reward, next_state, done
    """
    total_rows = buffer_tensor['state'].size()[0]
    if sample_size == -1:
        idx = np.array(range(total_rows))
    else:
        idx = np.random.choice(range(total_rows), sample_size)

    state, action, reward, next_state, done, next_state_action_musk = (buffer_tensor['state'][idx],
                                               buffer_tensor['action'][idx],
                                               buffer_tensor['reward'][idx],
                                               buffer_tensor['next_state'][idx],
                                               buffer_tensor['done'][idx],
                                               buffer_tensor['musk'][idx])

    return state, action, reward, next_state, done, next_state_action_musk
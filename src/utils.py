import dataclasses
import logging
import random
from collections import deque

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neural_network import MLPClassifier

import src.env.constants as envconst
from src.model import dummy_policy

logger = logging.getLogger(__name__)

def get_act_prob_df(df):
    df['state_tup'] = df.apply(lambda x: tuple(x['state']), axis=1)
    d = df.groupby(['state_tup', 'action']).count()['step'].reset_index()
    dd = d.groupby(['state_tup']).sum()['step'].reset_index()
    dd.rename(columns={'step': 'total'}, inplace=True)
    d.rename(columns={'step': 'count'}, inplace=True)
    d = d.merge(dd, on='state_tup', how='left')
    d['prob'] = d['count'] / d['total']
    d.drop(columns=['count', 'total'], inplace=True)

    dd = d.groupby('state_tup').apply(lambda x: dict(zip(x['action'], x['prob']))).reset_index()
    all_probs = []
    for i, row in dd.iterrows():
        probs = []
        for act in range(len(envconst.action_map)):
            probs.append(row[0].get(act, 0.000001))
        all_probs.append({'state_tup': row['state_tup'], 'act_prob': np.array(probs)})

    dd = pd.DataFrame(all_probs, columns=['state_tup', 'act_prob'])

    df = df.merge(dd, on=['state_tup'], how='left')
    df.drop(columns=['state_tup'], inplace=True)
    return df

def reward_predictor(df_org, seed=0, print_eval=False):
    df = df_org.loc[(df_org['action'] == envconst.action_map['a_workshsubmit'])].copy()
    df['temp'] = df.apply(lambda x: x['next_state'][envconst.state_map['s_end']], axis=1)
    df = df.loc[df['temp'] == 0]

    X = np.stack(df['state'])
    df['y'] = df.apply(lambda x: x['next_state'][envconst.state_map['s_solved']], axis=1)
    y = np.array(df['y'])

    if print_eval:
        skf = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
        f1 = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf = MLPClassifier(random_state=seed, max_iter=1000, shuffle=True)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            print(classification_report(y_test, y_pred))
            f1.append(f1_score(y_test, y_pred, average='weighted'))

        print("F1 Weighted Mean:", np.mean(f1))

    clf = MLPClassifier(random_state=seed, max_iter=1000, shuffle=True)
    clf.fit(X, y)
    return clf

def load_data_by_reward(df_location, reward=100, test_size=0.2):
    logger.info("-- loading data from {0} with test size {1} --".format(df_location, test_size))
    df = pd.read_pickle(df_location)

    df = get_act_prob_df(df)

    d = df.loc[(df['reward'] == reward)]
    student_ids = d['episode'].tolist()

    train_student, test_student = train_test_split(student_ids, test_size=test_size)

    train_df = df.loc[df['episode'].isin(train_student)].reset_index(drop=True)
    test_df = df.loc[df['episode'].isin(test_student)].reset_index(drop=True)

    # shuffle
    train_df = train_df.set_index("episode").loc[train_student].reset_index()
    test_df = test_df.set_index("episode").loc[test_student].reset_index()

    return train_df, test_df

def load_data(df_location, test_size=0.2, train_student=None, test_student=None):
    logger.info("-- loading data from {0} with test size {1} --".format(df_location, test_size))
    df = pd.read_pickle(df_location)
    df = get_act_prob_df(df)

    if train_student is None and test_student is None:
        d = df.loc[df['done']]
        student_ids = d['episode'].tolist()
        split_by = d['reward'].tolist()

        train_student, test_student = train_test_split(student_ids, test_size=test_size, stratify=split_by)

    train_df = df.loc[df['episode'].isin(train_student)].reset_index(drop=True)
    test_df = df.loc[df['episode'].isin(test_student)].reset_index(drop=True)

    # shuffle
    train_df = train_df.set_index("episode").loc[train_student].reset_index()
    test_df = test_df.set_index("episode").loc[test_student].reset_index()

    return train_df, test_df

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
    train_high_student = df_train.loc[(df_train['done']) & (df_train['reward'] == 100)]['episode'].unique()
    train_low_student = df_train.loc[(df_train['done']) & (df_train['reward'] == -100)]['episode'].unique()

    train_df_high = df_train.loc[(df_train['episode'].isin(train_high_student))]
    train_df_low = df_train.loc[(df_train['episode'].isin(train_low_student))]

    # shuffle
    train_df_high = train_df_high.set_index("episode").loc[train_high_student].reset_index()
    train_df_low = train_df_low.set_index("episode").loc[train_low_student].reset_index()

    # for test data
    test_high_student = df_test.loc[(df_test['done']) & (df_test['reward'] == 100)]['episode'].unique()
    test_low_student = df_test.loc[(df_test['done']) & (df_test['reward'] == -100)]['episode'].unique()

    test_df_high = df_test.loc[(df_test['episode'].isin(test_high_student))]
    test_df_low = df_test.loc[(df_test['episode'].isin(test_low_student))]

    # shuffle
    test_df_high = test_df_high.set_index("episode").loc[test_high_student].reset_index()
    test_df_low = test_df_low.set_index("episode").loc[test_low_student].reset_index()

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


def converged(curr_loss: float, prev_loss: float, prev_diffs: deque):
    diff = abs(curr_loss - prev_loss)
    prev_diffs.append(diff)
    diff_mean = np.mean(prev_diffs)
    return diff_mean <= 0.001


def diff_state(state, next_state):
    fchanges = {}
    fids = np.where(state != next_state)[0]
    fvals = next_state[fids]
    for fid, val in zip(fids, fvals):
        fname = envconst.state_map_rev[fid]
        fchanges[fname] = val
    return fchanges


def get_action_probs():
    df = pd.read_pickle("../processed_data/raw_logs.pkl")
    probs = {}
    d = df.loc[(df['action'] == 'PICKUP') & (~df['more_detail'].isin(
        ['cur-action-pickup-crate-1', 'cur-action-pickup-crate-3', 'cur-action-pickup-null']))].copy()
    d['name'] = d.apply(lambda x: x['more_detail'].split('-')[-1], axis=1)
    d['name'] = d.apply(
        lambda x: "s_obj_" + x["name"][:3] if x['name'] not in ["10", "11", "4"] else "s_obj_jar" + x["name"], axis=1)
    probs["a_obj"] = (d.groupby('name').count()['action'] / len(d)).to_dict()

    d = df.loc[(df['action'] == 'BOOKREAD')].copy()
    d['name'] = d.apply(lambda x: "s_book_" + x["detail"][:3], axis=1)
    probs["a_book"] = (d.groupby('name').count()['action'] / len(d)).to_dict()

    d = df.loc[(df['action'] == 'LOOKSTART')].copy()
    d['name'] = d.apply(
        lambda x: "s_post_" + x['detail'].split('-')[1][:3] + "_" + x['detail'].split('-')[2][:3] if len(
            x['detail'].split('-')) >= 3 else "s_post_" + x['detail'].split('-')[1][:3], axis=1)
    probs["a_post"] = (d.groupby('name').count()['action'] / len(d)).to_dict()
    return probs
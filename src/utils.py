import dataclasses
import logging
import random
from collections import deque
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from imitation.data.types import Transitions
from pyod.models.abod import ABOD
from pyod.models.auto_encoder import AutoEncoder
# from pyod.models.auto_encoder_torch import AutoEncoder
from pyod.models.cblof import CBLOF
from pyod.models.copod import COPOD
from pyod.models.iforest import IForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import src.env.constants as envconst
from src.model import policy

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


def get_anomaly_detector(df, seed=0, print_eval=False):
    X, y = actions_by_ep(df)

    if print_eval:
        skf = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
        train_1 = []
        test_1 = []
        train_2 = []
        test_2 = []
        train_3 = []
        test_3 = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            detector = CBLOF(contamination=0.05, random_state=seed)
            detector.fit(X_train)
            y_train = detector.predict(X_train)
            y_test = detector.predict(X_test)
            train_1.append(sum(y_train) / len(y_train) * 100.0)
            test_1.append(sum(y_test) / len(y_test) * 100.0)

            detector = AutoEncoder(contamination=0.05, verbose=0, epochs=500)
            detector.fit(X_train)
            y_train = detector.predict(X_train)
            y_test = detector.predict(X_test)
            train_2.append(sum(y_train) / len(y_train) * 100.0)
            test_2.append(sum(y_test) / len(y_test) * 100.0)

            detector = IForest(contamination=0.05)
            detector.fit(X_train)
            y_train = detector.predict(X_train)
            y_test = detector.predict(X_test)
            train_3.append(sum(y_train) / len(y_train) * 100.0)
            test_3.append(sum(y_test) / len(y_test) * 100.0)

        logger.info("CBLOF | Train | Anomaly: {0: .2f}%".format(np.mean(train_1)))
        logger.info("CBLOF | Test | Anomaly: {0: .2f}%".format(np.mean(test_1)))
        logger.info("ABOD | Train | Anomaly: {0: .2f}%".format(np.mean(train_2)))
        logger.info("ABOD | Test | Anomaly: {0: .2f}%".format(np.mean(test_2)))
        logger.info("IForest | Train | Anomaly: {0: .2f}%".format(np.mean(train_3)))
        logger.info("IForest | Test | Anomaly: {0: .2f}%".format(np.mean(test_3)))

    detector = CBLOF(contamination=0.05, random_state=seed)
    detector.fit(X)

    return detector

def outcome_predictor(df_location, seed=0, print_eval=False):
    df_org = pd.read_pickle(df_location)
    df = df_org.loc[(df_org['action'] == envconst.action_map['a_end'])].copy()

    X = np.stack(df['state'])
    df['y'] = df.apply(lambda x: 1 if x['reward'] == 100.0 else 0, axis=1)
    y = np.array(df['y'])

    if print_eval:
        skf = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
        metrics = []
        metrics_svm = []
        metrics_dt = []
        metrics_rf = []
        metrics_base = []
        metrics_major = []
        metrics_minor = []
        acc = []
        acc_svm = []
        acc_dt = []
        acc_rf = []
        acc_base = []
        acc_major = []
        acc_minor = []
        f1w = []
        f1w_svm = []
        f1w_dt = []
        f1w_rf = []
        f1w_base = []
        f1w_major = []
        f1w_minor = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf = MLPClassifier(random_state=seed, hidden_layer_sizes=(128, 128, 128), max_iter=10000, shuffle=True, activation='relu')
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            metrics.append(precision_recall_fscore_support(y_test, y_pred, zero_division=0))
            acc.append(accuracy_score(y_test, y_pred))
            f1w.append(f1_score(y_test, y_pred, average='weighted'))

            # other baselines
            clf = SVC(random_state=seed)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            metrics_svm.append(precision_recall_fscore_support(y_test, y_pred, zero_division=0))
            acc_svm.append(accuracy_score(y_test, y_pred))
            f1w_svm.append(f1_score(y_test, y_pred, average='weighted'))

            clf = DecisionTreeClassifier(random_state=seed)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            metrics_dt.append(precision_recall_fscore_support(y_test, y_pred, zero_division=0))
            acc_dt.append(accuracy_score(y_test, y_pred))
            f1w_dt.append(f1_score(y_test, y_pred, average='weighted'))

            clf = RandomForestClassifier(random_state=seed)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            metrics_rf.append(precision_recall_fscore_support(y_test, y_pred, zero_division=0))
            acc_rf.append(accuracy_score(y_test, y_pred))
            f1w_rf.append(f1_score(y_test, y_pred, average='weighted'))

            y_base = np.random.choice([0, 1], p=[.84, .16], size=len(y_test))
            metrics_base.append(precision_recall_fscore_support(y_test, y_base, zero_division=0))
            acc_base.append(accuracy_score(y_test, y_base))
            f1w_base.append(f1_score(y_test, y_base, average='weighted'))

            y_major = np.array([0] * len(y_test))
            metrics_major.append(precision_recall_fscore_support(y_test, y_major, zero_division=0))
            acc_major.append(accuracy_score(y_test, y_major))
            f1w_major.append(f1_score(y_test, y_major, average='weighted'))

            y_minor = np.array([1] * len(y_test))
            metrics_minor.append(precision_recall_fscore_support(y_test, y_minor, zero_division=0))
            acc_minor.append(accuracy_score(y_test, y_minor))
            f1w_minor.append(f1_score(y_test, y_minor, average='weighted'))

        prec, rec, f1 = [round(v1 * 100, 2) for (v0, v1) in np.mean(np.stack(metrics), axis=0)][:3]
        accuracy = round(np.mean(acc) * 100, 2)
        f1w = round(np.mean(f1w) * 100, 2)
        print(f"NN | Prec: {prec} | Recall {rec} | F1 {f1} | Acc {accuracy} | f1w {f1w}")
        prec, rec, f1 = [round(v1 * 100, 2) for (v0, v1) in np.mean(np.stack(metrics_svm), axis=0)][:3]
        accuracy = round(np.mean(acc_svm) * 100, 2)
        f1w = round(np.mean(f1w_svm) * 100, 2)
        print(f"SVM | Prec: {prec} | Recall {rec} | F1 {f1} | Acc {accuracy} | f1w {f1w}")
        prec, rec, f1 = [round(v1 * 100, 2) for (v0, v1) in np.mean(np.stack(metrics_dt), axis=0)][:3]
        accuracy = round(np.mean(acc_dt) * 100, 2)
        f1w = round(np.mean(f1w_dt) * 100, 2)
        print(f"DT | Prec: {prec} | Recall {rec} | F1 {f1} | Acc {accuracy} | f1w {f1w}")
        prec, rec, f1 = [round(v1 * 100, 2) for (v0, v1) in np.mean(np.stack(metrics_rf), axis=0)][:3]
        accuracy = round(np.mean(acc_rf) * 100, 2)
        f1w = round(np.mean(f1w_rf) * 100, 2)
        print(f"RF | Prec: {prec} | Recall {rec} | F1 {f1} | Acc {accuracy} | f1w {f1w}")

        prec, rec, f1 = [round(v1 * 100, 2) for (v0, v1) in np.mean(np.stack(metrics_base), axis=0)][:3]
        accuracy = round(np.mean(acc_base) * 100, 2)
        f1w = round(np.mean(f1w_base) * 100, 2)
        print(f"BASE | Prec: {prec} | Recall {rec} | F1 {f1} | Acc {accuracy} | f1w {f1w}")
        prec, rec, f1 = [round(v1 * 100, 2) for (v0, v1) in np.mean(np.stack(metrics_major), axis=0)][:3]
        accuracy = round(np.mean(acc_major) * 100, 2)
        f1w = round(np.mean(f1w_major) * 100, 2)
        print(f"MAJOR | Prec: {prec} | Recall {rec} | F1 {f1} | Acc {accuracy} | f1w {f1w}")
        prec, rec, f1 = [round(v1 * 100, 2) for (v0, v1) in np.mean(np.stack(metrics_minor), axis=0)][:3]
        accuracy = round(np.mean(acc_minor) * 100, 2)
        f1w = round(np.mean(f1w_minor) * 100, 2)
        print(f"MINOR | Prec: {prec} | Recall {rec} | F1 {f1} | Acc {accuracy} | f1w {f1w}")

    clf = MLPClassifier(random_state=seed, hidden_layer_sizes=(100,), max_iter=10000, shuffle=True,
                        activation='logistic')
    clf.fit(X, y)
    return clf

def load_data_by_outcome(df_location, outcome=100, test_size=0.2):
    logger.info("-- loading data from {0} with test size {1} --".format(df_location, test_size))
    df = pd.read_pickle(df_location)

    df = get_act_prob_df(df)

    d = df.loc[(df['reward'] == 0)]
    not_student_ids = set(d['episode'].tolist())
    student_ids = df.loc[~df['episode'].isin(not_student_ids)]['episode'].unique()

    train_student, test_student = train_test_split(student_ids, test_size=test_size)

    train_df = df.loc[df['episode'].isin(train_student)].reset_index(drop=True)
    test_df = df.loc[df['episode'].isin(test_student)].reset_index(drop=True)

    # shuffle
    train_df = train_df.set_index("episode").loc[train_student].reset_index()
    test_df = test_df.set_index("episode").loc[test_student].reset_index()

    return train_df, test_df

def load_data_by_reward(df_location, high=True, test_size=0.2):
    logger.info("-- loading data from {0} with test size {1} --".format(df_location, test_size))
    df = pd.read_pickle(df_location)

    df = get_act_prob_df(df)

    d = df.groupby('episode').sum()
    student_ids = d.loc[d['reward'] >= -150]['episode'].unique() if high else d.loc[d['reward'] < -150]['episode'].unique()

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


def pad_states(df, scaler=None, max_len=230):
    all_states = []
    for student, d in df.groupby('episode'):
        states = d['state']
        states = np.stack(states)
        states = states if scaler is None else scaler.transform(states)
        curr_len = len(states)
        if curr_len < max_len:
            pad_len = max_len - curr_len
            states = np.pad(states, pad_width=[(0, pad_len), (0, 0)], mode='constant', constant_values=0.)
        elif curr_len >= max_len:
            states = states[:max_len, :]

        all_states.append(states)

    states = np.stack(all_states)
    return states

def actions_by_ep(df):
    all_actions = []
    all_rewards = []
    for student, d in df.groupby('episode'):
        actions = d['action']
        actions = np.array(actions)
        all_rewards.append(d.iloc[-1]['reward'])

        if len(actions)!=envconst.max_ep_len:
            pad = envconst.max_ep_len - len(actions)
            pad = np.array([envconst.action_map['a_end']] * pad)
            actions = np.concatenate((actions, pad))

        all_actions.append(actions)

    actions = np.stack(all_actions)
    rewards = np.array(all_rewards)
    return actions, rewards

def states_by_ep(df):
    all_states = []
    for student, d in df.groupby('episode'):
        states = d['state']
        states = np.stack(states)

        if len(states) != envconst.max_ep_len:
            pad = envconst.max_ep_len - len(states)
            last_state = deepcopy(states[-1])
            pad = np.array([last_state]*pad)
            states = np.concatenate((states, pad))
        states = states.flatten()
        all_states.append(states)

    states = np.stack(all_states)
    return states



def get_episode_df(policy, total_eps):
    actions = []
    steps = []
    rewards = []
    step = 0
    state = policy.env.reset()
    action_counts = []
    curr_reward = 0
    for ep in range(total_eps):
        action = policy.get_action(state)
        actions.append(action)
        state, reward, done, info = policy.env.step(action)

        curr_reward += reward
        if done:
            steps.append(step)
            act_count = Counter(actions)
            action_counts.append(act_count)
            rewards.append(curr_reward)
            curr_reward = 0
            actions = []
            state = policy.env.reset()
            step = 0

    avg_action_counts = pd.DataFrame(action_counts).mean().reset_index().sort_values(by='index').set_index('index').round(1).to_dict()[0]
    return np.mean(steps), np.mean(rewards), avg_action_counts


def simulate_env(policy, total_episode):
    # logger.info("-- creating simulated data --")
    data = []
    steps = 0
    for ep in range(total_episode):
        state = policy.env.reset()
        ep_step = 0
        done = False
        while not done:
            action = policy.get_action(state)
            if ep_step == envconst.max_ep_len-1:
                action = envconst.action_map['a_end']
            next_state, reward, done, info = policy.env.step(action)

            data.append({'episode': str(ep), 'step': ep_step, 'state': state, 'action': action, 'reward': reward,
                         'next_state': next_state, 'done': done, 'info': info})
            state = deepcopy(next_state)
            ep_step += 1
            steps += 1
    df = pd.DataFrame(data, columns=['episode', 'step', 'state', 'action', 'reward', 'next_state', 'done', 'info'])

    return df

def get_transitions(df: pd.DataFrame):
    states = np.stack(df['state'])
    actions = np.array(df['action'])
    next_states = np.stack(df['next_state'])
    dones = np.array(df['done'])
    infos = np.array(df['info'])
    transitions = Transitions(obs=states, acts=actions, infos=infos, next_obs=next_states, dones=dones)
    return transitions


def Xy_shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def state_string(state):
    non_zero_feature_idx = np.where(state != 0)[0]

    gender = "female"
    if state[envconst.state_map['s_static_gender']] == 0:
        gender = "male"
    pretest = "high"
    if state[envconst.state_map['s_static_pretest']] == 0:
        pretest = "low"
    skill = "high"
    if state[envconst.state_map['s_static_gameskill']] == 0:
        skill = "low"

    target_item = "egg"
    if state[envconst.state_map['s_target_item']] == 1:
        target_item = "milk"
    elif state[envconst.state_map['s_target_item']] == 2:
        target_item = "sandwich"
    target_disease = "influenza"
    if state[envconst.state_map['s_target_disease']] == 1:
        target_disease = "salmonellosis"

    out = f"I am a {gender}. I have {pretest} pretest and {skill} game skill. " \
          f"In the game, I am investigating an outbreak of {target_disease} spreading through {target_item}. "

    spoken_to = []
    picked_up = []
    book_read = []
    post_read = []
    tested = []
    testleft = ""
    testpos = ""
    label = ""
    lesson = ""
    slide = ""
    worksheet = ""
    notetake = ""
    noteview = ""
    computer = ""
    submit = ""
    aes = []
    found = ""
    end = ""
    for fid in non_zero_feature_idx:
        fname = envconst.state_map_rev[fid]
        val = state[fid]
        if "s_talk_" in fname:
            name = fname.split("s_talk_")[1]
            spoken_to.append(name)
        elif "s_obj_" in fname:
            name = fname.split("s_obj_")[1]
            picked_up.append(name)
        elif "s_book_" in fname:
            name = fname.split("s_book_")[1]
            book_read.append(name)
        elif "s_post_" in fname:
            name = fname.split("s_post_")[1]
            post_read.append(name)
        elif "s_objtest_" in fname:
            name = fname.split("s_objtest_")[1]
            tested.append(name)
        elif "s_testleft" == fname:
            testleft = "I have " + str(val) + " test(s) left. "
        elif "s_testpos" == fname:
            testpos = "I have found the source of the disease. "
        elif "s_label" == fname:
            label = "I have worked on " + str(val) + " labels. "
        elif "s_label_lesson" == fname:
            lesson = "I have worked on the lesson. "
        elif "s_label_slide" == fname:
            slide = "I have worked on the slide. "
        elif "s_worksheet" == fname:
            worksheet = "I have worked on the worksheet " + str(val) + " time(s). "
        elif "s_notetake" == fname:
            notetake = "I have taken " + str(val) + " note(s). "
        elif "s_noteview" == fname:
            noteview = "I have viewed my notes " + str(val) + " time(s). "
        elif "s_computer" == fname:
            computer = "I have used the computer " + str(val) + " time(s). "
        elif "s_workshsubmit" == fname:
            submit = "I have submitted my worksheet " + str(val) + " time(s). "
        elif "s_aes_" in fname:
            aes.append(fname + " " + str(val) + " time(s)")
        elif "s_solved" == fname:
            found = "I have the correct solution."
        elif "s_end" == fname:
            end = "I closed the game."
    if len(spoken_to) != 0:
        out += "I have spoken to " + ', '.join(spoken_to) + ". "
    if len(picked_up) != 0:
        out += "I have picked up " + ', '.join(picked_up) + ". "
    if len(book_read) != 0:
        out += "I have read books on " + ', '.join(book_read) + ". "
    if len(post_read) != 0:
        out += "I have read posters on " + ', '.join(post_read) + ". "
    if len(tested) != 0:
        out += "I have tested " + ', '.join(tested) + " for the source of the disease. "
    out += testleft + testpos + label + lesson + slide + worksheet + notetake + noteview + computer + submit + found + end
    if len(aes) != 0:
        out += "I have received " + ', '.join(aes) + ". "
    return out

def action_string(action):
    out = "I want to "
    aname = envconst.action_map_rev[action]
    if "a_talk_" in aname:
        name = aname.split("a_talk_")[1]
        out += "speak to " + name + "."
    elif "a_obj" == aname:
        out += "pickup an object."
    elif "a_objtest" == aname:
        out += "test an object."
    elif "a_book" == aname:
        out += "read a book."
    elif "a_post" == aname:
        out += "read a poster."
    elif "a_notetake" == aname:
        out += "take a note."
    elif "a_noteview" == aname:
        out += "view my note(s)."
    elif "a_computer" == aname:
        out += "use the computer."
    elif "a_worksheet" == aname:
        out += "work on the worksheet."
    elif "a_label" == aname:
        out += "work on the label."
    elif "a_testleft" == aname:
        out += "get new test kit."
    elif "a_workshsubmit" == aname:
        out += "submit my worksheet."
    elif "a_end" == aname:
        out += "close the game."

    return out

def action_string_rev(out):
    for (aname, action) in envconst.action_map.items():
        if out == action_string(action):
            return action
    print("NO ACTION FOUND FOR ", out)
    return np.random.choice(list(envconst.action_map.values()))

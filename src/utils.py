import dataclasses
import logging
import random
from collections import deque

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_student_data_by_nlg(args: dataclasses):
    df = pd.read_pickle(args.student_data_loc)

    d = df.loc[df['done']]
    student_ids = d['episode'].tolist()
    nlgs = d['reward'].tolist()

    train_student, test_student = train_test_split(student_ids, test_size=0.2, stratify=nlgs)

    train_df = df.loc[df['episode'].isin(train_student)].reset_index(drop=True)
    test_df = df.loc[df['episode'].isin(test_student)].reset_index(drop=True)

    train_high_student = df.loc[(df['episode'].isin(train_student)) & (df['done']) & (df['reward'] == 100)][
        'episode']
    train_low_student = df.loc[(df['episode'].isin(train_student)) & (df['done']) & (df['reward'] == -100)][
        'episode']
    train_df_high = df.loc[df['episode'].isin(train_high_student)].reset_index(drop=True)
    train_df_low = df.loc[df['episode'].isin(train_low_student)].reset_index(drop=True)

    # shuffle
    train_df_high = train_df_high.set_index("episode").loc[train_high_student].reset_index()
    train_df_low = train_df_low.set_index("episode").loc[train_low_student].reset_index()

    test_high_student = df.loc[(df['episode'].isin(test_student)) & (df['done']) & (df['reward'] == 100)][
        'episode']
    test_low_student = df.loc[(df['episode'].isin(test_student)) & (df['done']) & (df['reward'] == -100)][
        'episode']
    test_df_high = df.loc[df['episode'].isin(test_high_student)].reset_index(drop=True)
    test_df_low = df.loc[df['episode'].isin(test_low_student)].reset_index(drop=True)
    # shuffle
    test_df_high = test_df_high.set_index("episode").loc[test_high_student].reset_index()
    test_df_low = test_df_low.set_index("episode").loc[test_low_student].reset_index()

    s0 = np.stack(df.loc[df['step'] == 0, 'state'])
    return train_df_high, train_df_low, test_df_high, test_df_low, s0


def load_data(df_location, test_size=0.2, train_student=None, test_student=None):
    logger.info("-- loading data from {0} with test size {1} --".format(df_location, test_size))
    df = pd.read_pickle(df_location)

    d = df.groupby('episode').last()
    eps = d.loc[d['done'] == False].index
    df = df.loc[~df['episode'].isin(eps)]

    if train_student is None and test_student is None:
        d = df.loc[df['done']]
        student_ids = d['episode'].tolist()
        nlgs = d['reward'].tolist()

        train_student, test_student = train_test_split(student_ids, test_size=test_size, stratify=nlgs)

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


######################################################
# this is part of the actual gail implementation

def get_flat_grads(f, net):
    flat_grads = torch.cat([
        grad.view(-1)
        for grad in torch.autograd.grad(f, net.parameters(), create_graph=True)
    ])

    return flat_grads


def get_flat_params(net):
    return torch.cat([param.view(-1) for param in net.parameters()])


def set_params(net, new_flat_params):
    start_idx = 0
    for param in net.parameters():
        end_idx = start_idx + np.prod(list(param.shape))
        param.data = torch.reshape(
            new_flat_params[start_idx:end_idx], param.shape
        )

        start_idx = end_idx


def conjugate_gradient(Av_func, b, max_iter=10, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b - Av_func(x)
    p = r
    rsold = r.norm() ** 2

    for _ in range(max_iter):
        Ap = Av_func(p)
        alpha = rsold / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.norm() ** 2
        if torch.sqrt(rsnew) < residual_tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x


def rescale_and_linesearch(
    g, s, Hs, max_kl, L, kld, old_params, pi, max_iter=10,
    success_ratio=0.1
):
    set_params(pi, old_params)
    L_old = L().detach()

    beta = torch.sqrt((2 * max_kl) / torch.dot(s, Hs))

    for _ in range(max_iter):
        new_params = old_params + beta * s

        set_params(pi, new_params)
        kld_new = kld().detach()

        L_new = L().detach()

        actual_improv = L_new - L_old
        approx_improv = torch.dot(g, beta * s)
        ratio = actual_improv / approx_improv

        if ratio > success_ratio \
            and actual_improv > 0 \
                and kld_new < max_kl:
            return new_params

        beta *= 0.5

    print("The line search was failed!")
    return old_params
import logging
from collections import Counter

import numpy as np
import pandas as pd
import torch
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.cblof import CBLOF
from pyod.models.iforest import IForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch import FloatTensor

from src import utils

logger = logging.getLogger(__name__)


def avg_rewards_actions(policy):
    logger.info("-- evaluation | reward and action --")
    df = utils.simulate_env(policy, total_episode=1000)
    rewards = df.groupby('episode')['reward'].sum().tolist()
    d = df.groupby('episode')['action'].apply(list)
    action_counts = [Counter(acts) for ep, acts in d.items()]
    avg_action_counts = pd.DataFrame(action_counts).mean().reset_index().sort_values(by='index').set_index('index').round(1).to_dict()[0]
    return np.mean(rewards), avg_action_counts


def perplexity(policy):
    logger.info("-- evaluation | perplexity --")
    all_states = np.stack(policy.test_df['state'])
    all_actions = np.array(policy.test_df['action'])
    all_act_log_probs = policy.get_all_act_log_probs(all_states, all_actions)
    return np.exp(-np.mean(all_act_log_probs))


def divergence(policy):
    logger.info("-- evaluation | divergence --")
    p = FloatTensor(policy.get_all_probs(np.stack(policy.test_df['state'])))
    q = FloatTensor(np.stack(policy.test_df['act_prob']))
    kld_pq = (p * (torch.log(p) - torch.log(q))).sum(-1).mean().item()
    kld_qp = (q * (torch.log(q) - torch.log(p))).sum(-1).mean().item()

    m = .5*(p + q)
    jsd = (.5 * (p * (torch.log(p) - torch.log(m))).sum(-1).mean().item()) + \
          (.5 * (q * (torch.log(q) - torch.log(p))).sum(-1).mean().item())

    return kld_pq, kld_qp, jsd


def anomaly(policy, print_eval=True):
    logger.info("-- evaluation | anomaly --")
    seed = policy.args.seed
    contamination = 0.1
    total_ep = int(len(policy.train_df['episode'].unique()) * contamination)

    df_sim = utils.simulate_env(policy, total_episode=total_ep)

    X_true, _ = utils.actions_by_ep(policy.train_df)
    X_sim, _ = utils.actions_by_ep(df_sim)
    y_true = np.array([0] * len(X_true))  # 0 no anomaly | actual
    y_sim = np.array([1] * len(X_sim))  # 1 is anomaly | simulated

    X = np.concatenate((X_sim, X_true), axis=0)
    y = np.concatenate((y_sim, y_true), axis=0)

    X, y = utils.Xy_shuffle(X, y)

    if print_eval:
        # scores
        metrics = []
        metrics_if = []
        acc = []
        acc_if = []
        f1w = []
        f1w_if = []
        ano = []
        ano_if = []
        metrics_train = []
        metrics_if_train = []
        acc_train = []
        acc_if_train = []
        f1w_train = []
        f1w_if_train = []
        ano_train = []
        ano_if_train = []
        skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # CBLOF
            detector = CBLOF(contamination=contamination, random_state=seed)
            detector.fit(X_train)
            y_pred_train = detector.predict(X_train)
            y_pred_test = detector.predict(X_test)
            # test
            metrics.append(precision_recall_fscore_support(y_test, y_pred_test, zero_division=0))
            acc.append(accuracy_score(y_test, y_pred_test))
            f1w.append(f1_score(y_test, y_pred_test, average='weighted'))
            ano.append(float(sum(y_pred_test))/float(len(y_pred_test)))
            # train
            metrics_train.append(precision_recall_fscore_support(y_train, y_pred_train, zero_division=0))
            acc_train.append(accuracy_score(y_train, y_pred_train))
            f1w_train.append(f1_score(y_train, y_pred_train, average='weighted'))
            ano_train.append(float(sum(y_pred_train))/float(len(y_pred_train)))

            # IForest
            detector = IForest(contamination=contamination, random_state=seed)
            detector.fit(X_train)
            y_pred_train = detector.predict(X_train)
            y_pred_test = detector.predict(X_test)
            # test
            metrics_if.append(precision_recall_fscore_support(y_test, y_pred_test, zero_division=0))
            acc_if.append(accuracy_score(y_test, y_pred_test))
            f1w_if.append(f1_score(y_test, y_pred_test, average='weighted'))
            ano_if.append(float(sum(y_pred_test)) / float(len(y_pred_test)))
            # train
            metrics_if_train.append(precision_recall_fscore_support(y_train, y_pred_train, zero_division=0))
            acc_if_train.append(accuracy_score(y_train, y_pred_train))
            f1w_if_train.append(f1_score(y_train, y_pred_train, average='weighted'))
            ano_if_train.append(float(sum(y_pred_train)) / float(len(y_pred_train)))

        prec_t, rec_t, f1_t = [round(v1 * 100, 2) for (v0, v1) in np.mean(np.stack(metrics_train), axis=0)][:3]
        accuracy_t = round(np.mean(acc_train) * 100, 2)
        f1w_t = round(np.mean(f1w_train) * 100, 2)
        ano_t = round(np.mean(ano_train) * 100, 2)
        prec, rec, f1 = [round(v1 * 100, 2) for (v0, v1) in np.mean(np.stack(metrics), axis=0)][:3]
        accuracy = round(np.mean(acc) * 100, 2)
        f1w = round(np.mean(f1w) * 100, 2)
        ano = round(np.mean(ano) * 100, 2)
        print(f"CBLOF | Prec: {prec} | Recall {rec} | F1 {f1} | Acc {accuracy} | "
              f"f1w {f1w} | Anomaly {ano}")

        prec_t, rec_t, f1_t = [round(v1 * 100, 2) for (v0, v1) in np.mean(np.stack(metrics_if_train), axis=0)][:3]
        accuracy_t = round(np.mean(acc_if_train) * 100, 2)
        f1w_t = round(np.mean(f1w_if_train) * 100, 2)
        ano_t = round(np.mean(ano_if_train) * 100, 2)
        prec, rec, f1 = [round(v1 * 100, 2) for (v0, v1) in np.mean(np.stack(metrics_if), axis=0)][:3]
        accuracy = round(np.mean(acc_if) * 100, 2)
        f1w = round(np.mean(f1w_if) * 100, 2)
        ano = round(np.mean(ano_if) * 100, 2)
        print(f"IF | Prec: {prec} | Recall {rec} | F1 {f1} | Acc {accuracy} | "
              f"f1w {f1w} | Anomaly {ano}")

    detector = CBLOF(contamination=contamination, random_state=seed)
    detector.fit(X)

    # test
    total_ep = int(len(policy.test_df['episode'].unique()) * contamination)
    df_sim_test = utils.simulate_env(policy, total_episode=total_ep)

    X_true_test, _ = utils.actions_by_ep(policy.test_df)
    X_sim_test, _ = utils.actions_by_ep(df_sim_test)
    y_true_test = np.array([0] * len(X_true_test))  # 0 is no anomaly | actual
    y_sim_test = np.array([1] * len(X_sim_test))  # 1 is anomaly | simulated

    X_test = np.concatenate((X_sim_test, X_true_test), axis=0)
    y_test = np.concatenate((y_sim_test, y_true_test), axis=0)

    X_test, y_test = utils.Xy_shuffle(X_test, y_test)

    y_pred = detector.predict(X_test)
    f1w = f1_score(y_test, y_pred, pos_label=1)
    return detector, f1w * 100


def classify(policy, print_eval=True):
    # TODO MOST DATA IS RANDOM/SINK states after padding, might confuse the clf, push the results in odd position
    # TODO also, if padded state is 1s, and our policy is not ending, then it is easy to see that at 220 pos, we expect 1
    logger.info("-- evaluation | classify --")
    seed = policy.args.seed
    total_ep = len(policy.train_df['episode'].unique())
    df_sim = utils.simulate_env(policy, total_episode=total_ep)

    df_sim['temp'] = df_sim.apply(lambda x: x['step'] < any(x['state']) and x['step'] > 2, axis=1)
    if len(df_sim.loc[df_sim['temp']==True])>0:
        print("WRONG")
    # X_sim = np.stack(df_sim['state'])
    # X_true = np.stack(policy.train_df['state'])
    X_sim = utils.states_by_ep(df_sim)
    X_true = utils.states_by_ep(policy.train_df)

    s_example = X_sim[np.random.randint(X_sim.shape[0], size=10), :]
    t_example = X_true[np.random.randint(X_true.shape[0], size=10), :]

    y_sim = np.array([1] * len(X_sim))  # 1 is simulated
    y_true = np.array([0] * len(X_true))  # 0 is actual

    X = np.concatenate((X_sim, X_true), axis=0)
    y = np.concatenate((y_sim, y_true), axis=0)

    X, y = utils.Xy_shuffle(X, y)

    if print_eval:
        # scores
        metrics_dt = []
        metrics_rf = []
        acc_dt = []
        acc_rf = []
        f1w_dt = []
        f1w_rf = []
        skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf = DecisionTreeClassifier(random_state=seed)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            metrics_dt.append(precision_recall_fscore_support(y_test, y_pred, zero_division=0))
            acc_dt.append(accuracy_score(y_test, y_pred))
            f1w_dt.append(f1_score(y_test, y_pred, average='weighted'))
            fi_dt = np.argmax(clf.feature_importances_)

            clf = RandomForestClassifier(random_state=seed)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            metrics_rf.append(precision_recall_fscore_support(y_test, y_pred, zero_division=0))
            acc_rf.append(accuracy_score(y_test, y_pred))
            f1w_rf.append(f1_score(y_test, y_pred, average='weighted'))
            fi_rf = np.argmax(clf.feature_importances_)

            fi_dt_name = fi_dt % len(policy.env.envconst.state_map)
            fi_rf_name = fi_rf % len(policy.env.envconst.state_map)
            fi_dt_step = fi_dt // len(policy.env.envconst.state_map)
            fi_rf_step = fi_rf // len(policy.env.envconst.state_map)
            for s, t in zip(s_example, t_example):
                s_val = s[fi_dt]
                t_val = t[fi_dt]
                print("DT", "FName:", policy.env.envconst.state_map_rev[fi_dt_name], "Step:", fi_dt_step, t_val, s_val)

                s_val = s[fi_rf]
                t_val = t[fi_rf]
                print("RF", "FName:", policy.env.envconst.state_map_rev[fi_rf_name], "Step:", fi_rf_step, t_val, s_val)

        prec, rec, f1 = [round(v1 * 100, 2) for (v0, v1) in np.mean(np.stack(metrics_dt), axis=0)][:3]
        accuracy = round(np.mean(acc_dt) * 100, 2)
        f1w = round(np.mean(f1w_dt) * 100, 2)
        print(f"DT | Prec: {prec} | Recall {rec} | F1 {f1} | Acc {accuracy} | f1w {f1w}")
        prec, rec, f1 = [round(v1 * 100, 2) for (v0, v1) in np.mean(np.stack(metrics_rf), axis=0)][:3]
        accuracy = round(np.mean(acc_rf) * 100, 2)
        f1w = round(np.mean(f1w_rf) * 100, 2)
        print(f"RF | Prec: {prec} | Recall {rec} | F1 {f1} | Acc {accuracy} | f1w {f1w}")

    clf = DecisionTreeClassifier(random_state=seed)
    clf.fit(X, y)

    # test score
    df_sim_test = utils.simulate_env(policy, total_episode=len(policy.test_df['episode'].unique()))
    # X_sim_test = np.stack(df_sim_test['state'])
    # X_true_test = np.stack(policy.test_df['state'])
    X_sim_test = utils.states_by_ep(df_sim_test)
    X_true_test = utils.states_by_ep(policy.test_df)
    y_sim_test = np.array([1] * len(X_sim_test))  # 1 is simulated
    y_true_test = np.array([0] * len(X_true_test))  # 0 is actual

    X_test = np.concatenate((X_sim_test, X_true_test), axis=0)
    y_test = np.concatenate((y_sim_test, y_true_test), axis=0)

    X_test, y_test = utils.Xy_shuffle(X_test, y_test)

    y_pred = clf.predict(X_test)
    f1w = f1_score(y_test, y_pred, average='weighted')
    return clf, f1w * 100
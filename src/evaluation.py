from sklearn.ensemble import RandomForestClassifier


def eval_sim(org_train, org_test, sim_train, sim_test, seed=0):
    """
    :param org_train: pd DataFrame for training classifier, all positive
    :param org_test: pd DataFrame for test classifier, all positive
    :param sim_train: pd DataFrame for training classifier, all negative, same length as org_train
    :param sim_test: pd DataFrame for training classifier, all negative, same length as org_test
    :return: accuracy, precision, f1 of test data
    """

    clf = RandomForestClassifier(random_state=seed)
    X_train_pos = org_train.apply(lambda x: x['state'] + [x['action'], ], axis=1)
    X_train_neg = sim_train.apply(lambda x: x['state'] + [x['action'], ], axis=1)
    print()
    return


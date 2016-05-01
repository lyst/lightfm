import numpy as np

import scipy.sparse as sp

from sklearn.metrics import roc_auc_score

from lightfm import LightFM, evaluation


def _precision_at_k(model, ground_truth, k, user_features=None, item_features=None):
    # Alternative test implementation

    ground_truth = ground_truth.tocsr()

    no_users, no_items = ground_truth.shape

    pid_array = np.arange(no_items, dtype=np.int32)

    precisions = []

    for user_id, row in enumerate(ground_truth):
        uid_array = np.empty(no_items, dtype=np.int32)
        uid_array.fill(user_id)
        predictions = model.predict(uid_array, pid_array,
                                    user_features=user_features,
                                    item_features=item_features,
                                    num_threads=4)

        top_k = set(np.argsort(-predictions)[:k])
        true_pids = set(row.indices[row.data == 1])

        if true_pids:
            precisions.append(len(top_k & true_pids) / float(k))

    return sum(precisions) / len(precisions)


def _auc(model, ground_truth, user_features=None, item_features=None):

    ground_truth = ground_truth.tocsr()

    no_users, no_items = ground_truth.shape

    pid_array = np.arange(no_items, dtype=np.int32)

    scores = []

    for user_id, row in enumerate(ground_truth):
        uid_array = np.empty(no_items, dtype=np.int32)
        uid_array.fill(user_id)
        predictions = model.predict(uid_array, pid_array,
                                    user_features=user_features,
                                    item_features=item_features,
                                    num_threads=4)

        true_pids = row.indices[row.data == 1]

        grnd = np.zeros(no_items, dtype=np.int32)
        grnd[true_pids] = 1

        if len(true_pids):
            scores.append(roc_auc_score(grnd, predictions))

    return scores


def test_precision_at_k():

    no_users, no_items = (10, 100)

    train = sp.rand(no_users, no_items, format='coo')
    train.data = np.ones_like(train.data)

    model = LightFM(loss='bpr')
    model.fit_partial(train)

    k = 10

    precision = evaluation.precision_at_k(model,
                                          train,
                                          k=k)
    expected_mean_precision = _precision_at_k(model,
                                              train,
                                              k)

    assert np.allclose(precision.mean(), expected_mean_precision)
    assert len(precision) == (train.getnnz(axis=1) > 0).sum()
    assert len(evaluation.precision_at_k(model,
                                         train,
                                         preserve_rows=True)) == train.shape[0]


def test_auc_score():

    no_users, no_items = (10, 100)

    train = sp.rand(no_users, no_items, format='coo')
    train.data = np.ones_like(train.data)

    model = LightFM(loss='bpr')
    model.fit_partial(train)

    auc = evaluation.auc_score(model,
                               train,
                               num_threads=2)
    expected_auc = np.array(_auc(model,
                                 train))

    assert auc.shape == expected_auc.shape
    assert np.abs(auc.mean() - expected_auc.mean()) < 0.01
    assert len(auc) == (train.getnnz(axis=1) > 0).sum()
    assert len(evaluation.auc_score(model,
                                    train,
                                    preserve_rows=True)) == train.shape[0]

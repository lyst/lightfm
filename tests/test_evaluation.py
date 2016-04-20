import numpy as np

import scipy.sparse as sp

from tests.utils import full_auc

from lightfm import LightFM


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


def test_precision_at_k():

    no_users, no_items = (10, 100)

    train = sp.rand(no_users, no_items, format='coo')
    train.data = np.ones_like(train.data)

    model = LightFM(loss='bpr')
    model.fit_partial(train)

    k = 10
    ranks = model.predict_rank(train, num_threads=2)
    ranks.data[ranks.data < k] = 1.0
    ranks.data[ranks.data >= k] = 0.0

    precision = np.squeeze(np.array(ranks.sum(axis=1))) / k
    precision = precision[train.getnnz(axis=1) > 0]
    mean_precision = precision.mean()

    train_precision = _precision_at_k(model,
                                      train,
                                      k)

    assert np.allclose(train_precision, mean_precision)

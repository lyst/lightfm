import numpy as np

import scipy.sparse as sp

from sklearn.metrics import roc_auc_score

from lightfm import LightFM, evaluation


def _generate_data(num_users, num_items, density=0.1, test_fraction=0.2):
    # Generate a dataset where every user has interactions
    # in both the train and the test set.

    train = sp.lil_matrix((num_users, num_items), dtype=np.float32)
    test = sp.lil_matrix((num_users, num_items), dtype=np.float32)

    for user_id in range(num_users):
        positives = np.random.choice(num_items,
                                     size=int(density * num_items),
                                     replace=False)

        for item_id in positives[:int(test_fraction * len(positives))]:
            test[user_id, item_id] = 1.0

        for item_id in positives[int(test_fraction * len(positives)):]:
            train[user_id, item_id] = 1.0

    return train.tocoo(), test.tocoo()


def _precision_at_k(model, ground_truth, k, train=None, user_features=None, item_features=None):
    # Alternative test implementation

    ground_truth = ground_truth.tocsr()

    no_users, no_items = ground_truth.shape

    pid_array = np.arange(no_items, dtype=np.int32)

    precisions = []

    uid_array = np.empty(no_items, dtype=np.int32)

    if train is not None:
        train = train.tocsr()

    for user_id, row in enumerate(ground_truth):
        uid_array.fill(user_id)

        predictions = model.predict(uid_array, pid_array,
                                    user_features=user_features,
                                    item_features=item_features,
                                    num_threads=4)
        if train is not None:
            train_items = train[user_id].indices
            top_k = set([x for x in np.argsort(-predictions)
                         if x not in train_items][:k])
        else:
            top_k = set(np.argsort(-predictions)[:k])

        true_pids = set(row.indices[row.data == 1])

        if true_pids:
            precisions.append(len(top_k & true_pids) / float(k))

    return sum(precisions) / len(precisions)


def _recall_at_k(model, ground_truth, k, train=None, user_features=None,
                 item_features=None):
    # Alternative test implementation

    ground_truth = ground_truth.tocsr()

    no_users, no_items = ground_truth.shape

    pid_array = np.arange(no_items, dtype=np.int32)

    recalls = []

    uid_array = np.empty(no_items, dtype=np.int32)

    if train is not None:
        train = train.tocsr()

    for user_id, row in enumerate(ground_truth):
        uid_array.fill(user_id)

        predictions = model.predict(uid_array, pid_array,
                                    user_features=user_features,
                                    item_features=item_features,
                                    num_threads=4)
        if train is not None:
            train_items = train[user_id].indices
            top_k = set([x for x in np.argsort(-predictions)
                         if x not in train_items][:k])
        else:
            top_k = set(np.argsort(-predictions)[:k])

        true_pids = set(row.indices[row.data == 1])

        if true_pids:
            recalls.append(len(top_k & true_pids) / float(len(true_pids)))

    return sum(recalls) / len(recalls)


def _auc(model, ground_truth, train=None, user_features=None, item_features=None):

    ground_truth = ground_truth.tocsr()

    no_users, no_items = ground_truth.shape

    pid_array = np.arange(no_items, dtype=np.int32)

    scores = []

    if train is not None:
        train = train.tocsr()

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

        if not len(true_pids):
            continue

        if train is not None:
            train_indices = train[user_id].indices
            not_in_train = np.array([x not in train_indices for x in range(no_items)])
            scores.append(roc_auc_score(grnd[not_in_train], predictions[not_in_train]))
        else:
            scores.append(roc_auc_score(grnd, predictions))

    return scores


def test_precision_at_k():

    no_users, no_items = (10, 100)

    train, test = _generate_data(no_users, no_items)

    model = LightFM(loss='bpr')

    # We want a high precision to catch the k=1 case
    model.fit_partial(test)

    for k in (10, 5, 1):

        # Without omitting train interactions
        precision = evaluation.precision_at_k(model,
                                              test,
                                              k=k)
        expected_mean_precision = _precision_at_k(model,
                                                  test,
                                                  k)

        assert np.allclose(precision.mean(), expected_mean_precision)
        assert len(precision) == (test.getnnz(axis=1) > 0).sum()
        assert len(evaluation.precision_at_k(model,
                                             train,
                                             preserve_rows=True)) == test.shape[0]

        # With omitting train interactions
        precision = evaluation.precision_at_k(model,
                                              test,
                                              k=k,
                                              train_interactions=train)
        expected_mean_precision = _precision_at_k(model,
                                                  test,
                                                  k,
                                                  train=train)

        assert np.allclose(precision.mean(), expected_mean_precision)


def test_precision_at_k_with_ties():

    no_users, no_items = (10, 100)

    train, test = _generate_data(no_users, no_items)

    model = LightFM(loss='bpr')
    model.fit_partial(train)

    # Make all predictions zero
    model.user_embeddings = np.zeros_like(model.user_embeddings)
    model.item_embeddings = np.zeros_like(model.item_embeddings)
    model.user_biases = np.zeros_like(model.user_biases)
    model.item_biases = np.zeros_like(model.item_biases)

    k = 10

    precision = evaluation.precision_at_k(model,
                                          test,
                                          k=k)

    # Pessimistic precision with all ties
    assert precision.mean() == 0.0


def test_recall_at_k():

    no_users, no_items = (10, 100)

    train, test = _generate_data(no_users, no_items)

    model = LightFM(loss='bpr')
    model.fit_partial(test)

    for k in (10, 5, 1):

        # Without omitting train interactions
        recall = evaluation.recall_at_k(model,
                                        test,
                                        k=k)
        expected_mean_recall = _recall_at_k(model,
                                            test,
                                            k)

        assert np.allclose(recall.mean(), expected_mean_recall)
        assert len(recall) == (test.getnnz(axis=1) > 0).sum()
        assert len(evaluation.recall_at_k(model,
                                          train,
                                          preserve_rows=True)) == test.shape[0]

        # With omitting train interactions
        recall = evaluation.recall_at_k(model,
                                        test,
                                        k=k,
                                        train_interactions=train)
        expected_mean_recall = _recall_at_k(model,
                                            test,
                                            k,
                                            train=train)

        assert np.allclose(recall.mean(), expected_mean_recall)


def test_auc_score():

    no_users, no_items = (10, 100)

    train, test = _generate_data(no_users, no_items)

    model = LightFM(loss='bpr')
    model.fit_partial(train)

    auc = evaluation.auc_score(model,
                               test,
                               num_threads=2)
    expected_auc = np.array(_auc(model,
                                 test))

    assert auc.shape == expected_auc.shape
    assert np.abs(auc.mean() - expected_auc.mean()) < 0.01
    assert len(auc) == (test.getnnz(axis=1) > 0).sum()
    assert len(evaluation.auc_score(model,
                                    train,
                                    preserve_rows=True)) == test.shape[0]

    # With omitting train interactions
    auc = evaluation.auc_score(model,
                               test,
                               train_interactions=train,
                               num_threads=2)
    expected_auc = np.array(_auc(model,
                                 test,
                                 train))
    assert np.abs(auc.mean() - expected_auc.mean()) < 0.01

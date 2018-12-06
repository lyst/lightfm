# coding=utf-8
"""
Module containing evaluation functions suitable for judging the performance of
a fitted LightFM model.
"""

import numpy as np

from ._lightfm_fast import CSRMatrix, calculate_auc_from_rank

__all__ = ["precision_at_k", "recall_at_k", "auc_score", "reciprocal_rank"]


def precision_at_k(
    model,
    test_interactions,
    train_interactions=None,
    k=10,
    user_features=None,
    item_features=None,
    preserve_rows=False,
    num_threads=1,
    check_intersections=True,
):
    """
    Measure the precision at k metric for a model: the fraction of known
    positives in the first k positions of the ranked list of results.
    A perfect score is 1.0.

    Parameters
    ----------

    model: LightFM instance
         the fitted model to be evaluated
    test_interactions: np.float32 csr_matrix of shape [n_users, n_items]
         Non-zero entries representing known positives in the evaluation set.
    train_interactions: np.float32 csr_matrix of shape [n_users, n_items], optional
         Non-zero entries representing known positives in the train set. These
         will be omitted from the score calculations to avoid re-recommending
         known positives.
    k: integer, optional
         The k parameter.
    user_features: np.float32 csr_matrix of shape [n_users, n_user_features], optional
         Each row contains that user's weights over features.
    item_features: np.float32 csr_matrix of shape [n_items, n_item_features], optional
         Each row contains that item's weights over features.
    preserve_rows: boolean, optional
         When False (default), the number of rows in the output will be equal
         to the number of users with interactions in the evaluation set.
         When True, the number of rows in the output will be equal to the
         number of users.
    num_threads: int, optional
         Number of parallel computation threads to use. Should
         not be higher than the number of physical cores.
    check_intersections: bool, optional, True by default,
        Only relevant when train_interactions are supplied.
        A flag that signals whether the test and train matrices should be checked
        for intersections to prevent optimistic ranks / wrong evaluation / bad data split.

    Returns
    -------

    np.array of shape [n_users with interactions or n_users,]
         Numpy array containing precision@k scores for each user. If there are
         no interactions for a given user the returned precision will be 0.
    """

    if num_threads < 1:
        raise ValueError("Number of threads must be 1 or larger.")

    ranks = model.predict_rank(
        test_interactions,
        train_interactions=train_interactions,
        user_features=user_features,
        item_features=item_features,
        num_threads=num_threads,
        check_intersections=check_intersections,
    )

    ranks.data = np.less(ranks.data, k, ranks.data)

    precision = np.squeeze(np.array(ranks.sum(axis=1))) / k

    if not preserve_rows:
        precision = precision[test_interactions.getnnz(axis=1) > 0]

    return precision


def recall_at_k(
    model,
    test_interactions,
    train_interactions=None,
    k=10,
    user_features=None,
    item_features=None,
    preserve_rows=False,
    num_threads=1,
    check_intersections=True,
):
    """
    Measure the recall at k metric for a model: the number of positive items in
    the first k positions of the ranked list of results divided by the number
    of positive items in the test period. A perfect score is 1.0.

    Parameters
    ----------

    model: LightFM instance
         the fitted model to be evaluated
    test_interactions: np.float32 csr_matrix of shape [n_users, n_items]
         Non-zero entries representing known positives in the evaluation set.
    train_interactions: np.float32 csr_matrix of shape [n_users, n_items], optional
         Non-zero entries representing known positives in the train set. These
         will be omitted from the score calculations to avoid re-recommending
         known positives.
    k: integer, optional
         The k parameter.
    user_features: np.float32 csr_matrix of shape [n_users, n_user_features], optional
         Each row contains that user's weights over features.
    item_features: np.float32 csr_matrix of shape [n_items, n_item_features], optional
         Each row contains that item's weights over features.
    preserve_rows: boolean, optional
         When False (default), the number of rows in the output will be equal
         to the number of users with interactions in the evaluation set.
         When True, the number of rows in the output will be equal to the
         number of users.
    num_threads: int, optional
         Number of parallel computation threads to use. Should
         not be higher than the number of physical cores.
    check_intersections: bool, optional, True by default,
        Only relevant when train_interactions are supplied.
        A flag that signals whether the test and train matrices should be checked
        for intersections to prevent optimistic ranks / wrong evaluation / bad data split.

    Returns
    -------

    np.array of shape [n_users with interactions or n_users,]
         Numpy array containing recall@k scores for each user. If there are no
         interactions for a given user having items in the test period, the
         returned recall will be 0.
    """

    if num_threads < 1:
        raise ValueError("Number of threads must be 1 or larger.")

    ranks = model.predict_rank(
        test_interactions,
        train_interactions=train_interactions,
        user_features=user_features,
        item_features=item_features,
        num_threads=num_threads,
        check_intersections=check_intersections,
    )

    ranks.data = np.less(ranks.data, k, ranks.data)

    retrieved = np.squeeze(test_interactions.getnnz(axis=1))
    hit = np.squeeze(np.array(ranks.sum(axis=1)))

    if not preserve_rows:
        hit = hit[test_interactions.getnnz(axis=1) > 0]
        retrieved = retrieved[test_interactions.getnnz(axis=1) > 0]

    return hit / retrieved


def auc_score(
    model,
    test_interactions,
    train_interactions=None,
    user_features=None,
    item_features=None,
    preserve_rows=False,
    num_threads=1,
    check_intersections=True,
):
    """
    Measure the ROC AUC metric for a model: the probability that a randomly
    chosen positive example has a higher score than a randomly chosen negative
    example.
    A perfect score is 1.0.

    Parameters
    ----------

    model: LightFM instance
         the fitted model to be evaluated
    test_interactions: np.float32 csr_matrix of shape [n_users, n_items]
         Non-zero entries representing known positives in the evaluation set.
    train_interactions: np.float32 csr_matrix of shape [n_users, n_items], optional
         Non-zero entries representing known positives in the train set. These
         will be omitted from the score calculations to avoid re-recommending
         known positives.
    user_features: np.float32 csr_matrix of shape [n_users, n_user_features], optional
         Each row contains that user's weights over features.
    item_features: np.float32 csr_matrix of shape [n_items, n_item_features], optional
         Each row contains that item's weights over features.
    preserve_rows: boolean, optional
         When False (default), the number of rows in the output will be equal
         to the number of users with interactions in the evaluation set.
         When True, the number of rows in the output will be equal to the
         number of users.
    num_threads: int, optional
         Number of parallel computation threads to use. Should
         not be higher than the number of physical cores.
    check_intersections: bool, optional, True by default,
        Only relevant when train_interactions are supplied.
        A flag that signals whether the test and train matrices should be checked
        for intersections to prevent optimistic ranks / wrong evaluation / bad data split.

    Returns
    -------

    np.array of shape [n_users with interactions or n_users,]
         Numpy array containing AUC scores for each user. If there are no
         interactions for a given user the returned AUC will be 0.5.
    """

    if num_threads < 1:
        raise ValueError("Number of threads must be 1 or larger.")

    ranks = model.predict_rank(
        test_interactions,
        train_interactions=train_interactions,
        user_features=user_features,
        item_features=item_features,
        num_threads=num_threads,
        check_intersections=check_intersections,
    )

    assert np.all(ranks.data >= 0)

    auc = np.zeros(ranks.shape[0], dtype=np.float32)

    if train_interactions is not None:
        num_train_positives = np.squeeze(
            np.array(train_interactions.getnnz(axis=1)).astype(np.int32)
        )
    else:
        num_train_positives = np.zeros(test_interactions.shape[0], dtype=np.int32)

    # The second argument is modified in-place, but
    # here we don't care about the inconsistency
    # introduced into the ranks matrix.
    calculate_auc_from_rank(
        CSRMatrix(ranks), num_train_positives, ranks.data, auc, num_threads
    )

    if not preserve_rows:
        auc = auc[test_interactions.getnnz(axis=1) > 0]

    return auc


def reciprocal_rank(
    model,
    test_interactions,
    train_interactions=None,
    user_features=None,
    item_features=None,
    preserve_rows=False,
    num_threads=1,
    check_intersections=True,
):
    """
    Measure the reciprocal rank metric for a model: 1 / the rank of the highest
    ranked positive example. A perfect score is 1.0.

    Parameters
    ----------

    model: LightFM instance
         the fitted model to be evaluated
    test_interactions: np.float32 csr_matrix of shape [n_users, n_items]
         Non-zero entries representing known positives in the evaluation set.
    train_interactions: np.float32 csr_matrix of shape [n_users, n_items], optional
         Non-zero entries representing known positives in the train set. These
         will be omitted from the score calculations to avoid re-recommending
         known positives.
    user_features: np.float32 csr_matrix of shape [n_users, n_user_features], optional
         Each row contains that user's weights over features.
    item_features: np.float32 csr_matrix of shape [n_items, n_item_features], optional
         Each row contains that item's weights over features.
    preserve_rows: boolean, optional
         When False (default), the number of rows in the output will be equal
         to the number of users with interactions in the evaluation set.
         When True, the number of rows in the output will be equal to the
         number of users.
    num_threads: int, optional
         Number of parallel computation threads to use. Should
         not be higher than the number of physical cores.
    check_intersections: bool, optional, True by default,
        Only relevant when train_interactions are supplied.
        A flag that signals whether the test and train matrices should be checked
        for intersections to prevent optimistic ranks / wrong evaluation / bad data split.

    Returns
    -------

    np.array of shape [n_users with interactions or n_users,]
         Numpy array containing reciprocal rank scores for each user.
         If there are no interactions for a given user the returned value will
         be 0.0.
    """

    if num_threads < 1:
        raise ValueError("Number of threads must be 1 or larger.")

    ranks = model.predict_rank(
        test_interactions,
        train_interactions=train_interactions,
        user_features=user_features,
        item_features=item_features,
        num_threads=num_threads,
        check_intersections=check_intersections,
    )

    ranks.data = 1.0 / (ranks.data + 1.0)

    ranks = np.squeeze(np.array(ranks.max(axis=1).todense()))

    if not preserve_rows:
        ranks = ranks[test_interactions.getnnz(axis=1) > 0]

    return ranks

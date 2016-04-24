"""
Module containing evaluation function suitable for judging the performance of a fitted
LightFM model.
"""

import numpy as np

from ._lightfm_fast import (CSRMatrix,
                            calculate_auc_from_rank)


__all__ = ['precision_at_k',
           'auc_score',
           'reciprocal_rank']


def precision_at_k(model, interactions, k=10, user_features=None, item_features=None, num_threads=1):
    """
    Measure the precision at k metric for a model: the fraction of known positives in the first k
    positions of the ranked list of results.
    A perfect score is 1.0.

    Parameters
    ----------

    model: LightFM instance
         the model to be evaluated
    interactions: np.float32 csr_matrix of shape [n_users, n_items]
         Non-zero entries representing known positives.
    k: integer, optional
         The k parameter.
    user_features: np.float32 csr_matrix of shape [n_users, n_user_features], optional
         Each row contains that user's weights over features.
    item_features: np.float32 csr_matrix of shape [n_items, n_item_features], optional
         Each row contains that item's weights over features.
    num_threads: int, optional
         Number of parallel computation threads to use. Should
         not be higher than the number of physical cores.

    Returns
    -------

    np.array of shape [n_users,]
         Numpy array containing precision@k scores for each user. If there are no interactions for a given
         user the returned precision will be 0.
    """

    ranks = model.predict_rank(interactions,
                               user_features=user_features,
                               item_features=item_features,
                               num_threads=num_threads)

    ranks.data[ranks.data < k] = 1.0
    ranks.data[ranks.data >= k] = 0.0

    precision = np.squeeze(np.array(ranks.sum(axis=1))) / k

    return precision


def auc_score(model, interactions, user_features=None, item_features=None, num_threads=1):
    """
    Measure the ROC AUC metric for a model: the probability that a randomly chosen positive
    example has a higher score than a randomly chosen negative example.
    A perfect score is 1.0.

    Parameters
    ----------

    model: LightFM instance
         the model to be evaluated
    interactions: np.float32 csr_matrix of shape [n_users, n_items]
         Non-zero entries representing known positives.
    user_features: np.float32 csr_matrix of shape [n_users, n_user_features], optional
         Each row contains that user's weights over features.
    item_features: np.float32 csr_matrix of shape [n_items, n_item_features], optional
         Each row contains that item's weights over features.
    num_threads: int, optional
         Number of parallel computation threads to use. Should
         not be higher than the number of physical cores.

    Returns
    -------

    np.array of shape [n_users,]
         Numpy array containing AUC scores for each user. If there are no interactions for a given
         user the returned AUC will be 0.5.
    """

    ranks = model.predict_rank(interactions,
                               user_features=user_features,
                               item_features=item_features,
                               num_threads=num_threads)
    assert np.all(ranks.data >= 0)

    auc = np.zeros(ranks.shape[0], dtype=np.float32)

    # The second argument is modified in-place, but
    # here we don't care about the inconsistency
    # introduced into the ranks matrix.
    calculate_auc_from_rank(CSRMatrix(ranks),
                            ranks.data,
                            auc,
                            num_threads)

    return auc


def reciprocal_rank(model, interactions, user_features=None, item_features=None, num_threads=1):
    """
    Measure the reciprocal rank metric for a model: 1 / the rank of the highest ranked positive example.
    A perfect score is 1.0.

    Parameters
    ----------

    model: LightFM instance
         the model to be evaluated
    interactions: np.float32 csr_matrix of shape [n_users, n_items]
         Non-zero entries representing known positives.
    user_features: np.float32 csr_matrix of shape [n_users, n_user_features], optional
         Each row contains that user's weights over features.
    item_features: np.float32 csr_matrix of shape [n_items, n_item_features], optional
         Each row contains that item's weights over features.
    num_threads: int, optional
         Number of parallel computation threads to use. Should
         not be higher than the number of physical cores.

    Returns
    -------

    np.array of shape [n_users,]
         Numpy array containing reciprocal rank scores for each user. If there are no interactions for a given
         user the returned value will be 0.0.
    """

    ranks = model.predict_rank(interactions,
                               user_features=user_features,
                               item_features=item_features,
                               num_threads=num_threads)

    ranks.data = 1.0 / (ranks.data + 1.0)

    return np.squeeze(np.array(ranks.max(axis=1).todense()))

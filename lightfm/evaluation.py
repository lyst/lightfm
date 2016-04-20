import numpy as np

from .lightfm_fast import (CSRMatrix,
                           calculate_auc_from_rank)


def precision_at_k(model, interactions, k=10, user_features=None, item_features=None, num_threads=1):
    """
    Measure the precision at k metric for a model: the fraction of known positives in the first k
    positions of the ranked list of results.

    Arguments:
    - LightFM instance model: the model to be evaluated
    - csr_matrix interactions: np.float32 matrix of shape [n_users, n_items]
                               with non-zero entries representing known positives.

    Optional arguments:
    - integer k: the k parameter. Default: 10.
    - csr_matrix user_features: array of shape [n_users, n_user_features].
                                Each row contains that user's weights
                                over features.
    - csr_matrix item_features: array of shape [n_items, n_item_features].
                                Each row contains that item's weights
                                over features.
    - int num_threads: number of parallel computation threads to use. Should
                       not be higher than the number of physical cores.
                       Default: 1

    Returns:
    - np.array of shape [n_users,] containing precision@k scores for each user.

    If there are no interactions for a given user the returned precision will be 0.
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

    ranks = model.predict_rank(interactions,
                               user_features=user_features,
                               item_features=item_features,
                               num_threads=num_threads)

    auc = np.zeros(ranks.shape[0], dtype=np.float32)

    calculate_auc_from_rank(CSRMatrix(ranks),
                            auc,
                            num_threads)

    return auc


def reciprocal_rank(model, interactions, user_features=None, item_features=None, num_threads=1):

    ranks = model.predict_rank(interactions,
                               user_features=user_features,
                               item_features=item_features,
                               num_threads=num_threads)

    ranks.data = 1.0 / (ranks.data + 1.0)

    return np.squeeze(np.array(ranks.max(axis=1).todense()))

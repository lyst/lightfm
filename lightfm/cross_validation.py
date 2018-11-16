# coding=utf-8
"""
Dataset splitting functions.
"""

import numpy as np
import scipy.sparse as sp


def _shuffle(uids, iids, data, random_state):

    shuffle_indices = np.arange(len(uids))
    random_state.shuffle(shuffle_indices)

    return (uids[shuffle_indices], iids[shuffle_indices], data[shuffle_indices])


def random_train_test_split(interactions, test_percentage=0.2, random_state=None):
    """
    Randomly split interactions between training and testing.

    This function takes an interaction set and splits it into
    two disjoint sets, a training set and a test set. Note that
    no effort is made to make sure that all items and users with
    interactions in the test set also have interactions in the
    training set; this may lead to a partial cold-start problem
    in the test set.
    To split a sample_weight matrix along the same lines, pass it
    into this function with the same random_state seed as was used
    for splitting the interactions.

    Parameters
    ----------

    interactions: a scipy sparse matrix containing interactions
        The interactions to split.
    test_percentage: float, optional
        The fraction of interactions to place in the test set.
    random_state: int or numpy.random.RandomState, optional
        Random seed used to initialize the numpy.random.RandomState number generator.
        Accepts an instance of numpy.random.RandomState for backwards compatibility.

    Returns
    -------

    (train, test): (scipy.sparse.COOMatrix,
                    scipy.sparse.COOMatrix)
         A tuple of (train data, test data)
    """

    if not sp.issparse(interactions):
        raise ValueError("Interactions must be a scipy.sparse matrix.")

    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(seed=random_state)

    interactions = interactions.tocoo()

    shape = interactions.shape
    uids, iids, data = (interactions.row, interactions.col, interactions.data)

    uids, iids, data = _shuffle(uids, iids, data, random_state)

    cutoff = int((1.0 - test_percentage) * len(uids))

    train_idx = slice(None, cutoff)
    test_idx = slice(cutoff, None)

    train = sp.coo_matrix(
        (data[train_idx], (uids[train_idx], iids[train_idx])),
        shape=shape,
        dtype=interactions.dtype,
    )
    test = sp.coo_matrix(
        (data[test_idx], (uids[test_idx], iids[test_idx])),
        shape=shape,
        dtype=interactions.dtype,
    )

    return train, test

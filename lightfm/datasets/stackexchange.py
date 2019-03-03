import os

import numpy as np

import scipy.sparse as sp

from lightfm.datasets import _common


def fetch_stackexchange(
    dataset,
    test_set_fraction=0.2,
    min_training_interactions=1,
    data_home=None,
    indicator_features=True,
    tag_features=False,
    download_if_missing=True,
):
    """
    Fetch a dataset from the `StackExchange network <http://stackexchange.com/>`_.

    The datasets contain users answering questions: an interaction is defined as a user
    answering a given question.

    The following datasets from the StackExchange network are available:

    - CrossValidated: From stats.stackexchange.com. Approximately 9000 users, 72000 questions,
      and 70000 answers.
    - StackOverflow: From stackoverflow.stackexchange.com. Approximately 1.3M users, 11M questions,
      and 18M answers.

    Parameters
    ----------

    dataset: string, one of ('crossvalidated', 'stackoverflow')
        The part of the StackExchange network for which to fetch the dataset.
    test_set_fraction: float, optional
        The fraction of the dataset used for testing. Splitting into the train and test set is done
        in a time-based fashion: all interactions before a certain time are in the train set and
        all interactions after that time are in the test set.
    min_training_interactions: int, optional
        Only include users with this amount of interactions in the training set.
    data_home: path, optional
        Path to the directory in which the downloaded data should be placed.
        Defaults to ``~/lightfm_data/``.
    indicator_features: bool, optional
        Use an [n_users, n_users] identity matrix for item features. When True with genre_features,
        indicator and genre features are concatenated into a single feature matrix of shape
        [n_users, n_users + n_genres].
    download_if_missing: bool, optional
        Download the data if not present. Raises an IOError if False and data is missing.

    Notes
    -----

    The return value is a dictionary containing the following keys:

    Returns
    -------

    train: sp.coo_matrix of shape [n_users, n_items]
         Contains training set interactions.
    test: sp.coo_matrix of shape [n_users, n_items]
         Contains testing set interactions.
    item_features: sp.csr_matrix of shape [n_items, n_item_features]
         Contains item features.
    item_feature_labels: np.array of strings of shape [n_item_features,]
         Labels of item features.
    """

    if not (indicator_features or tag_features):
        raise ValueError(
            "At least one of item_indicator_features " "or tag_features must be True"
        )

    if dataset not in ("crossvalidated", "stackoverflow"):
        raise ValueError("Unknown dataset")

    if not (0.0 < test_set_fraction < 1.0):
        raise ValueError("Test set fraction must be between 0 and 1")

    urls = {
        "crossvalidated": (
            "https://github.com/maciejkula/lightfm_datasets/releases/"
            "download/v0.1.0/stackexchange_crossvalidated.npz"
        ),
        "stackoverflow": (
            "https://github.com/maciejkula/lightfm_datasets/releases/"
            "download/v0.1.0/stackexchange_stackoverflow.npz"
        ),
    }

    path = _common.get_data(
        data_home,
        urls[dataset],
        os.path.join("stackexchange", dataset),
        "data.npz",
        download_if_missing,
    )

    data = np.load(path)

    interactions = sp.coo_matrix(
        (
            data["interactions_data"],
            (data["interactions_row"], data["interactions_col"]),
        ),
        shape=data["interactions_shape"].flatten(),
    )
    interactions.sum_duplicates()

    tag_features_mat = sp.coo_matrix(
        (data["features_data"], (data["features_row"], data["features_col"])),
        shape=data["features_shape"].flatten(),
    )
    tag_labels = data["labels"]

    test_cutoff_index = int(len(interactions.data) * (1.0 - test_set_fraction))
    test_cutoff_timestamp = np.sort(interactions.data)[test_cutoff_index]
    in_train = interactions.data < test_cutoff_timestamp
    in_test = np.logical_not(in_train)

    train = sp.coo_matrix(
        (
            np.ones(in_train.sum(), dtype=np.float32),
            (interactions.row[in_train], interactions.col[in_train]),
        ),
        shape=interactions.shape,
    )
    test = sp.coo_matrix(
        (
            np.ones(in_test.sum(), dtype=np.float32),
            (interactions.row[in_test], interactions.col[in_test]),
        ),
        shape=interactions.shape,
    )

    if min_training_interactions > 0:
        include = np.squeeze(np.array(train.getnnz(axis=1))) > min_training_interactions

        train = train.tocsr()[include].tocoo()
        test = test.tocsr()[include].tocoo()

    if indicator_features and not tag_features:
        features = sp.identity(train.shape[1], format="csr", dtype=np.float32)
        labels = np.array(["question_id:{}".format(x) for x in range(train.shape[1])])
    elif not indicator_features and tag_features:
        features = tag_features_mat.tocsr()
        labels = tag_labels
    else:
        id_features = sp.identity(train.shape[1], format="csr", dtype=np.float32)
        features = sp.hstack([id_features, tag_features_mat]).tocsr()
        labels = np.concatenate(
            [
                np.array(["question_id:{}".format(x) for x in range(train.shape[1])]),
                tag_labels,
            ]
        )

    return {
        "train": train,
        "test": test,
        "item_features": features,
        "item_feature_labels": labels,
    }

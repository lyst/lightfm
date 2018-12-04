import pytest

import numpy as np
import os

from sklearn.metrics import roc_auc_score

from lightfm import LightFM
from lightfm.datasets import fetch_movielens


def _binarize(dataset):

    positives = dataset.data >= 4.0
    dataset.data[positives] = 1.0
    dataset.data[np.logical_not(positives)] = -1.0

    return dataset


def _cleanup():
    os.remove(TEST_FILE_PATH)


TEST_FILE_PATH = "./tests/test.npz"
movielens = fetch_movielens()
train, test = _binarize(movielens["train"]), _binarize(movielens["test"])


def test_all_params_persisted():
    model = LightFM(loss="warp")
    model.fit(movielens["train"], epochs=1, num_threads=4)
    model.save(TEST_FILE_PATH)

    # Load and confirm all model params are present.
    saved_model_params = list(np.load(TEST_FILE_PATH).keys())
    for x in dir(model):
        ob = getattr(model, x)
        # We don't need to persist model functions, or magic variables of the model.
        if not callable(ob) and not x.startswith("__"):
            assert x in saved_model_params

    _cleanup()


def test_model_populated():
    model = LightFM(loss="warp")
    model.fit(movielens["train"], epochs=1, num_threads=4)
    model.save(TEST_FILE_PATH)

    # Load a model onto an uninstanciated object
    loaded_model = LightFM.load(TEST_FILE_PATH)

    assert loaded_model.item_embeddings.any()
    assert loaded_model.user_embeddings.any()

    _cleanup()


def test_model_performance():
    # Train and persist a model
    model = LightFM(random_state=10)
    model.fit_partial(train, epochs=10, num_threads=4)
    model.save(TEST_FILE_PATH)

    train_predictions = model.predict(train.row, train.col)
    test_predictions = model.predict(test.row, test.col)

    trn_pred = roc_auc_score(train.data, train_predictions)
    tst_pred = roc_auc_score(test.data, test_predictions)

    # Performance is same as before when loaded from disk
    loaded_model = LightFM.load(TEST_FILE_PATH)

    train_predictions = loaded_model.predict(train.row, train.col)
    test_predictions = loaded_model.predict(test.row, test.col)

    assert roc_auc_score(train.data, train_predictions) == trn_pred
    assert roc_auc_score(test.data, test_predictions) == tst_pred

    _cleanup()

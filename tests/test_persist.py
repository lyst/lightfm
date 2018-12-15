import pytest

import numpy as np
import os

from sklearn.metrics import roc_auc_score

from lightfm.lightfm import LightFM, model_weights
from lightfm.datasets import fetch_movielens


def _binarize(dataset):
    positives = dataset.data >= 4.0
    dataset.data[positives] = 1.0
    dataset.data[np.logical_not(positives)] = -1.0

    return dataset


TEST_FILE_PATH = "./tests/test.npz"
movielens = fetch_movielens()
train, test = _binarize(movielens["train"]), _binarize(movielens["test"])


class TestPersist:
    @pytest.fixture
    def model(self):
        # Train and persist a model
        model = LightFM(random_state=10)
        model.fit(movielens["train"], epochs=5, num_threads=4)
        model.save(TEST_FILE_PATH)
        return model

    @classmethod
    def teardown_class(cls):
        os.remove(TEST_FILE_PATH)

    def test_all_params_persisted(self, model):
        # Load and confirm all model params are present.
        saved_model_params = list(np.load(TEST_FILE_PATH).keys())
        for x in dir(model):
            ob = getattr(model, x)
            # We don't need to persist model functions, or magic variables of the model.
            if not callable(ob) and not x.startswith("__"):
                assert x in saved_model_params

    def test_all_loaded_weights_numpy_arrays(self, model):
        # Load a model onto an uninstanciated object
        loaded_model = LightFM.load(TEST_FILE_PATH)

        for weight_name in model_weights:
            assert callable(getattr(loaded_model, weight_name).any)

    def test_model_performance(self, model):
        train_predictions = model.predict(train.row, train.col)
        test_predictions = model.predict(test.row, test.col)

        trn_pred = roc_auc_score(train.data, train_predictions)
        tst_pred = roc_auc_score(test.data, test_predictions)

        # Performance is same as before when loaded from disk
        loaded_model = LightFM.load(TEST_FILE_PATH)

        train_predictions = loaded_model.predict(train.row, train.col)
        test_predictions = loaded_model.predict(test.row, test.col)

        # Use approximately equal because floating point math may make our summation slightly different.
        assert roc_auc_score(train.data, train_predictions) == pytest.approx(
            trn_pred, 0.0001
        )
        assert roc_auc_score(test.data, test_predictions) == pytest.approx(
            tst_pred, 0.0001
        )

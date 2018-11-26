import pytest

import numpy as np
import os

from lightfm import LightFM
from lightfm.datasets import fetch_movielens

TEST_FILE_PATH = "./tests/test.npz"


def test_all_params_persisted():
    # Train and persist a model
    data = fetch_movielens(min_rating=5.0)
    model = LightFM(loss="warp")
    model.fit(data["train"], epochs=5, num_threads=4)
    model.save(TEST_FILE_PATH)

    # Load and confirm all model params are present.
    saved_model_params = list(np.load(TEST_FILE_PATH).keys())
    for x in dir(model):
        ob = getattr(model, x)
        if not callable(ob) and not x.startswith("__"):
            assert x in saved_model_params

    # Clean up
    os.remove(TEST_FILE_PATH)


def test_model_populated():
    # Train and persist a model
    data = fetch_movielens(min_rating=5.0)
    model = LightFM(loss="warp")
    model.fit(data["train"], epochs=5, num_threads=4)
    model.save(TEST_FILE_PATH)

    # Load a model onto an uninstanciated object
    model = LightFM(loss="warp")

    assert model.item_embeddings == None
    assert model.user_embeddings == None

    model.load(TEST_FILE_PATH)

    assert model.item_embeddings.any()
    assert model.user_embeddings.any()

    # Clean up
    os.remove(TEST_FILE_PATH)

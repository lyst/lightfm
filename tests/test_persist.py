import numpy as np
import pytest

from lightfm import LightFM
from lightfm.datasets import fetch_movielens

def test_all_params_persisted():
    # Train and persist a model
    data = fetch_movielens(min_rating=5.0)
    model = LightFM(loss='warp')
    model.fit(data['train'], epochs=5, num_threads=4)
    model.save('./test.npz')

    # Load and confirm all model params are present.
    saved_model_params = list(np.load('./test.npz').keys())
    for x in dir(model):
        ob = getattr(model, x)
        if not callable(ob) and not x.startswith('__'):
            assert x in saved_model_params

def test_model_populated():
    # Train and persist a model
    data = fetch_movielens(min_rating=5.0)
    model = LightFM(loss='warp')
    model.fit(data['train'], epochs=5, num_threads=4)
    model.save('./test.npz')

    # Load a model onto an uninstanciated object
    model = LightFM(loss='warp')

    assert model.item_embeddings == None
    assert model.user_embeddings == None

    model.load('./test.npz')

    assert model.item_embeddings.any()
    assert model.user_embeddings.any()

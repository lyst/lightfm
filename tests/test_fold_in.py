import numpy as np
import pytest

from lightfm import LightFM


def _setup_model(
        item_feature_names=list('abc'), user_feature_names=list('uvxy')):
    """Helper method to setup a model"""
    no_components = 2
    if item_feature_names is None:
        no_item_features = 10
    else:
        no_item_features = len(item_feature_names)

    if user_feature_names is None:
        no_user_features = 7
    else:
        no_user_features = len(user_feature_names)

    model = LightFM(
        no_components=no_components,
        item_feature_names=item_feature_names,
        user_feature_names=user_feature_names)

    # initialize the model to get some numbers into the embeddings and biases
    model._initialize(
        no_components=no_components,
        no_item_features=no_item_features,
        no_user_features=no_user_features)

    # biases are initialized as zeros, so some some values to be able to assert
    # that the copy correctly
    model.item_biases = np.array([.1, .3, .7])

    return model


def test_model_should_carry_learned_values_to_new_model():
    old_model = _setup_model()
    new_model = old_model.resize(
        item_feature_names=list('dcab'), user_feature_names=list('yzuvx'))

    # the item feature named 'c' has
    # index 1 in new_model, and index 2 in old_model
    assert (new_model.item_embeddings[1] == old_model.item_embeddings[2]).all()

    # the user feature named 'y' has
    # index 0 in new_model, and index 3 in old_model
    assert (new_model.user_embeddings[0] == old_model.user_embeddings[3]).all()


def test_hyperparameters_should_copy_from_old_to_new_model():
    old_model = _setup_model()
    new_model = old_model.resize(
        item_feature_names=old_model.item_feature_names,
        user_feature_names=old_model.user_feature_names)
    params = old_model.get_params()
    for param, value in params.items():
        assert getattr(old_model, param) == getattr(new_model, param)


def test_should_raise_if_copying_from_model_without_feature_names():
    old_model = _setup_model(
        item_feature_names=None,
        user_feature_names=None
    )
    with pytest.raises(ValueError):
        old_model.resize(
            item_feature_names=list('abc'),
            user_feature_names=list('xyz'))

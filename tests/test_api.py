import numpy as np

import pytest

import scipy.sparse as sp

from lightfm import LightFM


def test_empty_matrix():

    no_users, no_items = (10, 100)

    train = sp.coo_matrix((no_users,
                           no_items),
                          dtype=np.int32)

    model = LightFM()
    model.fit_partial(train)


def test_matrix_types():

    mattypes = (sp.coo_matrix,
                sp.lil_matrix,
                sp.csr_matrix,
                sp.csc_matrix)

    dtypes = (np.int32,
              np.int64,
              np.float32,
              np.float64)

    no_users, no_items = (10, 100)
    no_features = 20

    for mattype in mattypes:
        for dtype in dtypes:
            train = mattype((no_users,
                             no_items),
                            dtype=dtype)

            user_features = mattype((no_users,
                                     no_features),
                                    dtype=dtype)
            item_features = mattype((no_items,
                                     no_features),
                                    dtype=dtype)

            model = LightFM()
            model.fit_partial(train,
                              user_features=user_features,
                              item_features=item_features)

            model.predict(np.random.randint(0, no_users, 10).astype(np.int32),
                          np.random.randint(0, no_items, 10).astype(np.int32),
                          user_features=user_features,
                          item_features=item_features)


def test_input_dtypes():

    dtypes = (np.int32,
              np.int64,
              np.float32,
              np.float64)

    no_users, no_items = (10, 100)
    no_features = 20

    for dtype in dtypes:
        train = sp.coo_matrix((no_users,
                               no_items),
                              dtype=dtype)

        user_features = sp.coo_matrix((no_users,
                                       no_features),
                                      dtype=dtype)
        item_features = sp.coo_matrix((no_items,
                                       no_features),
                                      dtype=dtype)

        model = LightFM()
        model.fit_partial(train,
                          user_features=user_features,
                          item_features=item_features)

        model.predict(np.random.randint(0, no_users, 10).astype(np.int32),
                      np.random.randint(0, no_items, 10).astype(np.int32),
                      user_features=user_features,
                      item_features=item_features)


def test_not_enough_features_fails():

    no_users, no_items = (10, 100)
    no_features = 20

    train = sp.coo_matrix((no_users,
                           no_items),
                          dtype=np.int32)

    user_features = sp.csr_matrix((no_users-1,
                                   no_features),
                                  dtype=np.int32)
    item_features = sp.csr_matrix((no_items-1,
                                   no_features),
                                  dtype=np.int32)
    model = LightFM()
    with pytest.raises(Exception):
        model.fit_partial(train,
                          user_features=user_features,
                          item_features=item_features)


def test_feature_inference_fails():

    # On predict if we try to use feature inference and supply
    # higher ids than the number of features that were supplied to fit
    # we should complain

    no_users, no_items = (10, 100)
    no_features = 20

    train = sp.coo_matrix((no_users,
                           no_items),
                          dtype=np.int32)

    user_features = sp.csr_matrix((no_users,
                                   no_features),
                                  dtype=np.int32)
    item_features = sp.csr_matrix((no_items,
                                   no_features),
                                  dtype=np.int32)
    model = LightFM()
    model.fit_partial(train,
                      user_features=user_features,
                      item_features=item_features)

    with pytest.raises(AssertionError):
        model.predict(np.array([no_features], dtype=np.int32),
                      np.array([no_features], dtype=np.int32))


def test_return_self():

    no_users, no_items = (10, 100)

    train = sp.coo_matrix((no_users,
                           no_items),
                          dtype=np.int32)

    model = LightFM()
    assert model.fit_partial(train) is model
    assert model.fit(train) is model


def test_param_sanity():

    with pytest.raises(AssertionError):
        LightFM(no_components=-1)

    with pytest.raises(AssertionError):
        LightFM(user_alpha=-1.0)

    with pytest.raises(AssertionError):
        LightFM(item_alpha=-1.0)

    with pytest.raises(ValueError):
        LightFM(max_sampled=-1.0)


def test_sample_weight():

    model = LightFM()

    train = sp.coo_matrix(np.array([[0, 1],
                                    [0, 1]]))

    with pytest.raises(ValueError):
        # Wrong number of weights
        sample_weight = sp.coo_matrix(np.zeros((2, 2)))

        model.fit(train,
                  sample_weight=sample_weight)

    with pytest.raises(ValueError):
        # Wrong shape
        sample_weight = sp.coo_matrix(np.zeros(2))
        model.fit(train,
                  sample_weight=np.zeros(3))

    with pytest.raises(ValueError):
        # Wrong order of entries
        sample_weight = sp.coo_matrix((train.data,
                                       (train.row[::-1],
                                        train.col[::-1])))
        model.fit(train,
                  sample_weight=np.zeros(3))

    sample_weight = sp.coo_matrix((train.data,
                                   (train.row,
                                    train.col)))
    model.fit(train, sample_weight=sample_weight)

    model = LightFM(loss='warp-kos')

    with pytest.raises(NotImplementedError):
        model.fit(train,
                  sample_weight=np.ones(1))

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

            model.predict_rank(train,
                               user_features=user_features,
                               item_features=item_features)


def test_coo_with_duplicate_entries():
    # Calling .tocsr on a COO matrix with duplicate entries
    # changes its data arrays in-place, leading to out-of-bounds
    # array accesses in the WARP code.
    # Reported in https://github.com/lyst/lightfm/issues/117.

    rows, cols = (1000, 100)
    mat = sp.random(rows, cols)
    mat.data[:] = 1

    # Duplicate entries in the COO matrix
    mat.data = np.concatenate((mat.data, mat.data[:1000]))
    mat.row = np.concatenate((mat.row, mat.row[:1000]))
    mat.col = np.concatenate((mat.col, mat.col[:1000]))

    for loss in ('warp', 'bpr', 'warp-kos'):
        model = LightFM(loss='warp')
        model.fit(mat)


def test_predict():

    no_users, no_items = (10, 100)

    train = sp.coo_matrix((no_users,
                           no_items),
                          dtype=np.int32)

    model = LightFM()
    model.fit_partial(train)

    for uid in range(no_users):
        scores_arr = model.predict(np.repeat(uid, no_items),
                                   np.arange(no_items))
        scores_int = model.predict(uid,
                                   np.arange(no_items))
        assert np.allclose(scores_arr, scores_int)


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

    user_features = sp.csr_matrix((no_users - 1,
                                   no_features),
                                  dtype=np.int32)
    item_features = sp.csr_matrix((no_items - 1,
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


def test_predict_ranks():

    no_users, no_items = (10, 100)

    train = sp.coo_matrix((no_users,
                           no_items),
                          dtype=np.float32)
    train = sp.rand(no_users, no_items, format='csr', random_state=42)

    model = LightFM()
    model.fit_partial(train)

    # Compute ranks for all items
    rank_input = sp.csr_matrix(np.ones((no_users, no_items)))
    ranks = model.predict_rank(rank_input, num_threads=2).todense()

    assert np.all(ranks.min(axis=1) == 0)
    assert np.all(ranks.max(axis=1) == no_items - 1)

    for row in range(no_users):
        assert np.all(np.sort(ranks[row]) == np.arange(no_items))

    # Train set exclusions. All ranks should be zero
    # if train interactions is dense.
    ranks = model.predict_rank(rank_input,
                               train_interactions=rank_input).todense()
    assert np.all(ranks == 0)

    # Max rank should be num_items - 1 - number of positives
    # in train in that row
    ranks = model.predict_rank(rank_input,
                               train_interactions=train).todense()
    assert np.all(np.squeeze(np.array(ranks.max(axis=1))) ==
                  no_items - 1 - np.squeeze(np.array(train.getnnz(axis=1))))

    # Make sure ranks are computed pessimistically when
    # there are ties (that is, equal predictions for every
    # item will assign maximum rank to each).
    model.user_embeddings = np.zeros_like(model.user_embeddings)
    model.item_embeddings = np.zeros_like(model.item_embeddings)
    model.user_biases = np.zeros_like(model.user_biases)
    model.item_biases = np.zeros_like(model.item_biases)

    ranks = model.predict_rank(rank_input, num_threads=2).todense()

    assert np.all(ranks.min(axis=1) == 99)
    assert np.all(ranks.max(axis=1) == 99)

    # Wrong input dimensions
    with pytest.raises(ValueError):
        model.predict_rank(sp.csr_matrix((5, 5)), num_threads=2)


def test_exception_on_divergence():

    no_users, no_items = (1000, 1000)

    train = sp.coo_matrix((no_users,
                           no_items),
                          dtype=np.float32)
    train = sp.rand(no_users, no_items, format='csr', random_state=42)

    model = LightFM(learning_rate=10000000.0,
                    loss='warp')

    with pytest.raises(ValueError):
        model.fit(train, epochs=10)


def test_sklearn_api():
    model = LightFM()
    params = model.get_params()
    model2 = LightFM(**params)
    params2 = model2.get_params()
    assert params == params2
    model.set_params(**params)
    params['invalid_param'] = 666
    with pytest.raises(ValueError):
        model.set_params(**params)

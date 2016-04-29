import pytest

import scipy.sparse as sp

from lightfm.datasets import fetch_movielens, fetch_stackexchange


def test_basic_fetching_movielens():

    data = fetch_movielens()

    assert isinstance(data['train'], sp.coo_matrix)
    assert isinstance(data['test'], sp.coo_matrix)

    assert data['train'].shape == data['test'].shape
    assert data['train'].shape == (943, 1682)
    assert (data['train'].getnnz() + data['test'].getnnz()) == 100000

    assert data['item_features'].shape == (1682, 1682)
    assert len(data['item_feature_labels']) == 1682
    assert data['item_feature_labels'] is data['item_labels']

    data = fetch_movielens(genre_features=True)

    assert data['item_features'].shape == (1682, len(data['item_feature_labels']))
    assert data['item_feature_labels'] is not data['item_labels']

    with pytest.raises(ValueError):
        data = fetch_movielens(indicator_features=False, genre_features=False)


def test_basic_fetching_stackexchange():

    test_fractions = (0.2, 0.5, 0.6)

    for test_fraction in test_fractions:
        data = fetch_stackexchange('crossvalidated', test_set_fraction=test_fraction)

        train = data['train']
        test = data['test']

        assert isinstance(train, sp.coo_matrix)
        assert isinstance(test, sp.coo_matrix)

        assert train.shape == test.shape

        frac = float(test.getnnz()) / (train.getnnz() + test.getnnz())
        assert abs(frac - test_fraction) < 0.01

    data = fetch_stackexchange('crossvalidated', indicator_features=True, tag_features=False)
    assert isinstance(data['item_features'], sp.csr_matrix)
    assert data['item_features'].shape[0] == data['item_features'].shape[1] == data['train'].shape[1]

    data = fetch_stackexchange('crossvalidated', indicator_features=False, tag_features=True)
    assert isinstance(data['item_features'], sp.csr_matrix)
    assert data['item_features'].shape[0] > data['item_features'].shape[1]

    data = fetch_stackexchange('crossvalidated', indicator_features=True, tag_features=True)
    assert isinstance(data['item_features'], sp.csr_matrix)
    assert data['item_features'].shape[0] < data['item_features'].shape[1]

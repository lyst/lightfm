import pytest

import scipy.sparse as sp

from lightfm.datasets import fetch_movielens


def test_basic_fetching():

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

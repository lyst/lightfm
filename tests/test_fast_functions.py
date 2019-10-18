import numpy as np

import scipy.sparse as sp


from lightfm import _lightfm_fast


def test_in_positives():

    mat = sp.csr_matrix(np.array([[0, 1], [1, 0]])).astype(np.float32)

    assert not _lightfm_fast.__test_in_positives(0, 0, _lightfm_fast.CSRMatrix(mat))
    assert _lightfm_fast.__test_in_positives(0, 1, _lightfm_fast.CSRMatrix(mat))

    assert _lightfm_fast.__test_in_positives(1, 0, _lightfm_fast.CSRMatrix(mat))
    assert not _lightfm_fast.__test_in_positives(1, 1, _lightfm_fast.CSRMatrix(mat))


def test_item_groups():
    data = np.repeat(1.0, 5)
    rows = [0, 1, 3, 2, 1]
    cols = [0, 1, 2, 3, 4]
    random_state = np.random.RandomState()
    mat = sp.csr_matrix((data, (rows, cols)), shape=(4, 5)).astype(np.float32)
    res = _lightfm_fast.test_item_group_map(2, _lightfm_fast.CSRMatrix(mat), random_state)
    assert res == 3

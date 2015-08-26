import numpy as np
import scipy.sparse as sp


from lightfm import lightfm_fast


def test_in_positives():

    mat = sp.csr_matrix(np.array([[0, 1],
                                  [1, 0]])).astype(np.int32)

    assert not lightfm_fast.__test_in_positives(0, 0, lightfm_fast.CSRMatrix(mat))
    assert lightfm_fast.__test_in_positives(0, 1, lightfm_fast.CSRMatrix(mat))

    assert lightfm_fast.__test_in_positives(1, 0, lightfm_fast.CSRMatrix(mat))
    assert not lightfm_fast.__test_in_positives(1, 1, lightfm_fast.CSRMatrix(mat))

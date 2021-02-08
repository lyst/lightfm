import pytest
import numpy as np
import scipy.sparse as sp

from lightfm import LightFM

# set our default random seed
SEED = 42


@pytest.fixture(scope="session")
def rng():
    """Initialise a shared random number generator for all tests."""

    return np.random.RandomState(SEED)


@pytest.fixture(scope="session")
def array_int32(rng, size=10):
    """Initialise an array of type np.int32 of size `size`."""

    return rng.randint(0, 100, size=size, dtype=np.int32)


@pytest.fixture(
    scope="session",
    ids=["tuple", "list", "ndarray"],
    params=[tuple, list, np.array]
)
def user_ids(array_int32, request):
    """Initialise input user_ids valid for calls to the LightFM.predict method.

    Notes
    -----
    On parameterized pytest fixtures: This fixture will iterate over all passed
    `params`. This avoids having to apply a `pytest.mark.parameterize` decorator to
    every test that needs the same `user_ids`.

    You can find out more about parameterized fixtures in the pytest docs:
    https://docs.pytest.org/en/stable/parametrize.html

    """

    _type = request.param
    yield _type(array_int32)


@pytest.fixture(
    scope="session",
    ids=["tuple", "list", "ndarray"],
    params=[tuple, list, np.array]
)
def item_ids(array_int32, request):
    """Initialise input item_ids valid for calls to the LightFM.predict method.

    Notes
    -----
    See `user_ids` fixture for a note on parameterized fixtures.

    """
    _type = request.param
    yield _type(array_int32)


@pytest.fixture(scope="session")
def train_matrix(rng, n_users=1000, n_items=1000):
    """Create a random sparse CSR matrix of shape (n_users, n_items) for training."""

    return sp.rand(n_users, n_items, format="csr", random_state=rng)


@pytest.fixture(scope="session")
def lfm(train_matrix):
    """Create a _trained_ LightFM model instance."""

    model = LightFM()
    model.fit(train_matrix)

    return model

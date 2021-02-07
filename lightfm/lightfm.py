# coding=utf-8
from __future__ import print_function

import numpy as np

import scipy.sparse as sp

from ._lightfm_fast import (
    CSRMatrix,
    FastLightFM,
    fit_bpr,
    fit_logistic,
    fit_warp,
    fit_warp_kos,
    predict_lightfm,
    predict_ranks,
)

__all__ = ["LightFM"]

CYTHON_DTYPE = np.float32


class LightFM(object):
    """
    A hybrid latent representation recommender model.

    The model learns embeddings (latent representations in a high-dimensional
    space) for users and items in a way that encodes user preferences over items.
    When multiplied together, these representations produce scores for every item
    for a given user; items scored highly are more likely to be interesting to
    the user.

    The user and item representations are expressed in terms of representations
    of their features: an embedding is estimated for every feature, and these
    features are then summed together to arrive at representations for users and
    items. For example, if the movie 'Wizard of Oz' is described by the following
    features: 'musical fantasy', 'Judy Garland', and 'Wizard of Oz', then its
    embedding will be given by taking the features' embeddings and adding them
    together. The same applies to user features.

    The embeddings are learned through `stochastic gradient
    descent <http://cs231n.github.io/optimization-1/>`_ methods.

    Four loss functions are available:

    - logistic: useful when both positive (1) and negative (-1) interactions
      are present.
    - BPR: Bayesian Personalised Ranking [1]_ pairwise loss. Maximises the
      prediction difference between a positive example and a randomly
      chosen negative example. Useful when only positive interactions
      are present and optimising ROC AUC is desired.
    - WARP: Weighted Approximate-Rank Pairwise [2]_ loss. Maximises
      the rank of positive examples by repeatedly sampling negative
      examples until rank violating one is found. Useful when only
      positive interactions are present and optimising the top of
      the recommendation list (precision@k) is desired.
    - k-OS WARP: k-th order statistic loss [3]_. A modification of WARP that
      uses the k-th positive example for any given user as a basis for pairwise
      updates.

    Two learning rate schedules are available:

    - adagrad: [4]_
    - adadelta: [5]_

    Parameters
    ----------

    no_components: int, optional
        the dimensionality of the feature latent embeddings.
    k: int, optional
        for k-OS training, the k-th positive example will be selected from the
        n positive examples sampled for every user.
    n: int, optional
        for k-OS training, maximum number of positives sampled for each update.
    learning_schedule: string, optional
        one of ('adagrad', 'adadelta').
    loss: string, optional
        one of  ('logistic', 'bpr', 'warp', 'warp-kos'): the loss function.
    learning_rate: float, optional
        initial learning rate for the adagrad learning schedule.
    rho: float, optional
        moving average coefficient for the adadelta learning schedule.
    epsilon: float, optional
        conditioning parameter for the adadelta learning schedule.
    item_alpha: float, optional
        L2 penalty on item features. Tip: setting this number too high can slow
        down training. One good way to check is if the final weights in the
        embeddings turned out to be mostly zero. The same idea applies to
        the user_alpha parameter.
    user_alpha: float, optional
        L2 penalty on user features.
    max_sampled: int, optional
        maximum number of negative samples used during WARP fitting.
        It requires a lot of sampling to find negative triplets for users that
        are already well represented by the model; this can lead to very long
        training times and overfitting. Setting this to a higher number will
        generally lead to longer training times, but may in some cases improve
        accuracy.
    random_state: int seed, RandomState instance, or None
        The seed of the pseudo random number generator to use when shuffling
        the data and initializing the parameters.

    Attributes
    ----------

    item_embeddings: np.float32 array of shape [n_item_features, n_components]
         Contains the estimated latent vectors for item features. The [i, j]-th
         entry gives the value of the j-th component for the i-th item feature.
         In the simplest case where the item feature matrix is an identity
         matrix, the i-th row will represent the i-th item latent vector.
    user_embeddings: np.float32 array of shape [n_user_features, n_components]
         Contains the estimated latent vectors for user features. The [i, j]-th
         entry gives the value of the j-th component for the i-th user feature.
         In the simplest case where the user feature matrix is an identity
         matrix, the i-th row will represent the i-th user latent vector.
    item_biases: np.float32 array of shape [n_item_features,]
         Contains the biases for item_features.
    user_biases: np.float32 array of shape [n_user_features,]
         Contains the biases for user_features.

    Notes
    -----

    Users' and items' latent representations are expressed in terms of their
    features' representations. If no feature matrices are provided to the
    :func:`lightfm.LightFM.fit` or :func:`lightfm.LightFM.predict` methods, they are
    implicitly assumed to be identity matrices: that is, each user and item
    are characterised by one feature that is unique to that user (or item).
    In this case, LightFM reduces to a traditional collaborative filtering
    matrix factorization method.

    For including features, there are two strategies:

    1. Characterizing each user/item *only* by its features.

    2. Characterizing each user/item by its features *and* an identity matrix
       that captures interactions between users and items directly.

    1. When using only features, the feature matrix should be of shape
    ``(num_<users/items> x num_features)``. To build these feature matrices,
    it is recommended to use the build methods from the class
    :class:`lightfm.data.Dataset` and setting the ``<user/item>_identity_features``
    to ``False``.
    An embedding will then be estimated for every feature: that is, there will be
    ``num_features`` embeddings.
    To obtain the representation for user i, the model will look up the i-th
    row of the feature matrix to find the features with non-zero weights in
    that row; the embeddings for these features will then be added together
    to arrive at the user representation. For example, if user 10 has weight 1
    in the 5th column of the user feature matrix, and weight 3 in the 20th
    column, that user's representation will be found by adding together
    the embedding for the 5th and the 20th features (multiplying the latter
    by 3). The same goes for items.

    Note: This strategy may result in a less expressive model because no per-user
    features are estimated, the model may underfit. To combat this, follow
    strategy 2. and include per-user (per-item) features (that is, an identity matrix)
    as part of the feature matrix.

    2. To use features alongside user-item interactions, the feature matrix should
    include an identity matrix. The resulting feature matrix should be of shape
    ``(num_<users/items> x (num_<users/items> + num_features))``. This strategy is
    the default when using the :class:`lightfm.data.Dataset` class. The
    behavior is controlled by the ``<user/item>_identity_features=True`` default arguments.


    References
    ----------

    .. [1] Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from
           implicit feedback."
           Proceedings of the Twenty-Fifth Conference on Uncertainty in
           Artificial Intelligence. AUAI Press, 2009.
    .. [2] Weston, Jason, Samy Bengio, and Nicolas Usunier. "Wsabie: Scaling up
           to large vocabulary image annotation." IJCAI. Vol. 11. 2011.
    .. [3] Weston, Jason, Hector Yee, and Ron J. Weiss. "Learning to rank
           recommendations with the k-order statistic loss."
           Proceedings of the 7th ACM conference on Recommender systems. ACM,
           2013.
    .. [4] Duchi, John, Elad Hazan, and Yoram Singer. "Adaptive subgradient
           methods for online learning and stochastic optimization."
           The Journal of Machine Learning Research 12 (2011): 2121-2159.
    .. [5] Zeiler, Matthew D. "ADADELTA: An adaptive learning rate method."
           arXiv preprint arXiv:1212.5701 (2012).
    """

    def __init__(
        self,
        no_components=10,
        k=5,
        n=10,
        learning_schedule="adagrad",
        loss="logistic",
        learning_rate=0.05,
        rho=0.95,
        epsilon=1e-6,
        item_alpha=0.0,
        user_alpha=0.0,
        max_sampled=10,
        random_state=None,
    ):

        assert item_alpha >= 0.0
        assert user_alpha >= 0.0
        assert no_components > 0
        assert k > 0
        assert n > 0
        assert 0 < rho < 1
        assert epsilon >= 0
        assert learning_schedule in ("adagrad", "adadelta")
        assert loss in ("logistic", "warp", "bpr", "warp-kos")

        if max_sampled < 1:
            raise ValueError("max_sampled must be a positive integer")

        self.loss = loss
        self.learning_schedule = learning_schedule

        self.no_components = no_components
        self.learning_rate = learning_rate

        self.k = int(k)
        self.n = int(n)

        self.rho = rho
        self.epsilon = epsilon
        self.max_sampled = max_sampled

        self.item_alpha = item_alpha
        self.user_alpha = user_alpha

        if random_state is None:
            self.random_state = np.random.RandomState()
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(random_state)

        self._reset_state()

    def _reset_state(self):

        self.item_embeddings = None
        self.item_embedding_gradients = None
        self.item_embedding_momentum = None
        self.item_biases = None
        self.item_bias_gradients = None
        self.item_bias_momentum = None

        self.user_embeddings = None
        self.user_embedding_gradients = None
        self.user_embedding_momentum = None
        self.user_biases = None
        self.user_bias_gradients = None
        self.user_bias_momentum = None

    def _check_initialized(self):

        for var in (
            self.item_embeddings,
            self.item_embedding_gradients,
            self.item_embedding_momentum,
            self.item_biases,
            self.item_bias_gradients,
            self.item_bias_momentum,
            self.user_embeddings,
            self.user_embedding_gradients,
            self.user_embedding_momentum,
            self.user_biases,
            self.user_bias_gradients,
            self.user_bias_momentum,
        ):

            if var is None:
                raise ValueError(
                    "You must fit the model before " "trying to obtain predictions."
                )

    def _initialize(self, no_components, no_item_features, no_user_features):
        """
        Initialise internal latent representations.
        """

        # Initialise item features.
        self.item_embeddings = (
            (self.random_state.rand(no_item_features, no_components) - 0.5)
            / no_components
        ).astype(np.float32)
        self.item_embedding_gradients = np.zeros_like(self.item_embeddings)
        self.item_embedding_momentum = np.zeros_like(self.item_embeddings)
        self.item_biases = np.zeros(no_item_features, dtype=np.float32)
        self.item_bias_gradients = np.zeros_like(self.item_biases)
        self.item_bias_momentum = np.zeros_like(self.item_biases)

        # Initialise user features.
        self.user_embeddings = (
            (self.random_state.rand(no_user_features, no_components) - 0.5)
            / no_components
        ).astype(np.float32)
        self.user_embedding_gradients = np.zeros_like(self.user_embeddings)
        self.user_embedding_momentum = np.zeros_like(self.user_embeddings)
        self.user_biases = np.zeros(no_user_features, dtype=np.float32)
        self.user_bias_gradients = np.zeros_like(self.user_biases)
        self.user_bias_momentum = np.zeros_like(self.user_biases)

        if self.learning_schedule == "adagrad":
            self.item_embedding_gradients += 1
            self.item_bias_gradients += 1
            self.user_embedding_gradients += 1
            self.user_bias_gradients += 1

    def _construct_feature_matrices(
        self, n_users, n_items, user_features, item_features
    ):

        if user_features is None:
            user_features = sp.identity(n_users, dtype=CYTHON_DTYPE, format="csr")
        else:
            user_features = user_features.tocsr()

        if item_features is None:
            item_features = sp.identity(n_items, dtype=CYTHON_DTYPE, format="csr")
        else:
            item_features = item_features.tocsr()

        if n_users > user_features.shape[0]:
            raise Exception(
                "Number of user feature rows does not equal the number of users"
            )

        if n_items > item_features.shape[0]:
            raise Exception(
                "Number of item feature rows does not equal the number of items"
            )

        # If we already have embeddings, verify that
        # we have them for all the supplied features
        if self.user_embeddings is not None:
            if not self.user_embeddings.shape[0] >= user_features.shape[1]:
                raise ValueError(
                    "The user feature matrix specifies more "
                    "features than there are estimated "
                    "feature embeddings: {} vs {}.".format(
                        self.user_embeddings.shape[0], user_features.shape[1]
                    )
                )

        if self.item_embeddings is not None:
            if not self.item_embeddings.shape[0] >= item_features.shape[1]:
                raise ValueError(
                    "The item feature matrix specifies more "
                    "features than there are estimated "
                    "feature embeddings: {} vs {}.".format(
                        self.item_embeddings.shape[0], item_features.shape[1]
                    )
                )

        user_features = self._to_cython_dtype(user_features)
        item_features = self._to_cython_dtype(item_features)

        return user_features, item_features

    def _get_positives_lookup_matrix(self, interactions):

        mat = interactions.tocsr()

        if not mat.has_sorted_indices:
            return mat.sorted_indices()
        else:
            return mat

    def _to_cython_dtype(self, mat):

        if mat.dtype != CYTHON_DTYPE:
            return mat.astype(CYTHON_DTYPE)
        else:
            return mat

    def _process_sample_weight(self, interactions, sample_weight):

        if sample_weight is not None:

            if self.loss == "warp-kos":
                raise NotImplementedError(
                    "k-OS loss with sample weights " "not implemented."
                )

            if not isinstance(sample_weight, sp.coo_matrix):
                raise ValueError("Sample_weight must be a COO matrix.")

            if sample_weight.shape != interactions.shape:
                raise ValueError(
                    "Sample weight and interactions " "matrices must be the same shape"
                )

            if not (
                np.array_equal(interactions.row, sample_weight.row)
                and np.array_equal(interactions.col, sample_weight.col)
            ):
                raise ValueError(
                    "Sample weight and interaction matrix "
                    "entries must be in the same order"
                )

            if sample_weight.data.dtype != CYTHON_DTYPE:
                sample_weight_data = sample_weight.data.astype(CYTHON_DTYPE)
            else:
                sample_weight_data = sample_weight.data
        else:
            if np.array_equiv(interactions.data, 1.0):
                # Re-use interactions data if they are all
                # ones
                sample_weight_data = interactions.data
            else:
                # Otherwise allocate a new array of ones
                sample_weight_data = np.ones_like(interactions.data, dtype=CYTHON_DTYPE)

        return sample_weight_data

    def _get_lightfm_data(self):

        lightfm_data = FastLightFM(
            self.item_embeddings,
            self.item_embedding_gradients,
            self.item_embedding_momentum,
            self.item_biases,
            self.item_bias_gradients,
            self.item_bias_momentum,
            self.user_embeddings,
            self.user_embedding_gradients,
            self.user_embedding_momentum,
            self.user_biases,
            self.user_bias_gradients,
            self.user_bias_momentum,
            self.no_components,
            int(self.learning_schedule == "adadelta"),
            self.learning_rate,
            self.rho,
            self.epsilon,
            self.max_sampled,
        )

        return lightfm_data

    def _check_finite(self):

        for parameter in (
            self.item_embeddings,
            self.item_biases,
            self.user_embeddings,
            self.user_biases,
        ):
            # A sum of an array that contains non-finite values
            # will also be non-finite, and we avoid creating a
            # large boolean temporary.
            if not np.isfinite(np.sum(parameter)):
                raise ValueError(
                    "Not all estimated parameters are finite,"
                    " your model may have diverged. Try decreasing"
                    " the learning rate or normalising feature values"
                    " and sample weights"
                )

    def _check_input_finite(self, data):

        if not np.isfinite(np.sum(data)):
            raise ValueError(
                "Not all input values are finite. "
                "Check the input for NaNs and infinite values."
            )

    @staticmethod
    def _progress(n, verbose):
        # Use `tqdm` if available,
        # otherwise fallback to `range()`.
        if not verbose:
            return range(n)

        try:
            from tqdm import trange

            return trange(n, desc="Epoch")
        except ImportError:

            def verbose_range():
                for i in range(n):
                    print("Epoch {}".format(i))
                    yield i

            return verbose_range()

    def fit(
        self,
        interactions,
        user_features=None,
        item_features=None,
        sample_weight=None,
        epochs=1,
        num_threads=1,
        verbose=False,
    ):
        """
        Fit the model.

        For details on how to use feature matrices, see the documentation
        on the :class:`lightfm.LightFM` class.

        Arguments
        ---------

        interactions: np.float32 coo_matrix of shape [n_users, n_items]
             the matrix containing
             user-item interactions. Will be converted to
             numpy.float32 dtype if it is not of that type.
        user_features: np.float32 csr_matrix of shape [n_users, n_user_features], optional
             Each row contains that user's weights over features.
        item_features: np.float32 csr_matrix of shape [n_items, n_item_features], optional
             Each row contains that item's weights over features.
        sample_weight: np.float32 coo_matrix of shape [n_users, n_items], optional
             matrix with entries expressing weights of individual
             interactions from the interactions matrix.
             Its row and col arrays must be the same as
             those of the interactions matrix. For memory
             efficiency its possible to use the same arrays
             for both weights and interaction matrices.
             Defaults to weight 1.0 for all interactions.
             Not implemented for the k-OS loss.
        epochs: int, optional
             number of epochs to run
        num_threads: int, optional
             Number of parallel computation threads to use. Should
             not be higher than the number of physical cores.
        verbose: bool, optional
             whether to print progress messages.
             If `tqdm` is installed, a progress bar will be displayed instead.

        Returns
        -------

        LightFM instance
            the fitted model

        """

        # Discard old results, if any
        self._reset_state()

        return self.fit_partial(
            interactions,
            user_features=user_features,
            item_features=item_features,
            sample_weight=sample_weight,
            epochs=epochs,
            num_threads=num_threads,
            verbose=verbose,
        )

    def fit_partial(
        self,
        interactions,
        user_features=None,
        item_features=None,
        sample_weight=None,
        epochs=1,
        num_threads=1,
        verbose=False,
    ):
        """
        Fit the model.

        Fit the model. Unlike fit, repeated calls to this method will
        cause training to resume from the current model state.

        For details on how to use feature matrices, see the documentation
        on the :class:`lightfm.LightFM` class.

        Arguments
        ---------

        interactions: np.float32 coo_matrix of shape [n_users, n_items]
             the matrix containing
             user-item interactions. Will be converted to
             numpy.float32 dtype if it is not of that type.
        user_features: np.float32 csr_matrix of shape [n_users, n_user_features], optional
             Each row contains that user's weights over features.
        item_features: np.float32 csr_matrix of shape [n_items, n_item_features], optional
             Each row contains that item's weights over features.
        sample_weight: np.float32 coo_matrix of shape [n_users, n_items], optional
             matrix with entries expressing weights of individual
             interactions from the interactions matrix.
             Its row and col arrays must be the same as
             those of the interactions matrix. For memory
             efficiency its possible to use the same arrays
             for both weights and interaction matrices.
             Defaults to weight 1.0 for all interactions.
             Not implemented for the k-OS loss.
        epochs: int, optional
             number of epochs to run
        num_threads: int, optional
             Number of parallel computation threads to use. Should
             not be higher than the number of physical cores.
        verbose: bool, optional
             whether to print progress messages.
             If `tqdm` is installed, a progress bar will be displayed instead.

        Returns
        -------

        LightFM instance
            the fitted model
        """

        # We need this in the COO format.
        # If that's already true, this is a no-op.
        interactions = interactions.tocoo()

        if interactions.dtype != CYTHON_DTYPE:
            interactions.data = interactions.data.astype(CYTHON_DTYPE)

        sample_weight_data = self._process_sample_weight(interactions, sample_weight)

        n_users, n_items = interactions.shape
        (user_features, item_features) = self._construct_feature_matrices(
            n_users, n_items, user_features, item_features
        )

        for input_data in (
            user_features.data,
            item_features.data,
            interactions.data,
            sample_weight_data,
        ):
            self._check_input_finite(input_data)
        if self.item_embeddings is None:
            # Initialise latent factors only if this is the first call
            # to fit_partial.
            self._initialize(
                self.no_components, item_features.shape[1], user_features.shape[1]
            )

        # Check that the dimensionality of the feature matrices has
        # not changed between runs.
        if not item_features.shape[1] == self.item_embeddings.shape[0]:
            raise ValueError("Incorrect number of features in item_features")

        if not user_features.shape[1] == self.user_embeddings.shape[0]:
            raise ValueError("Incorrect number of features in user_features")

        if num_threads < 1:
            raise ValueError("Number of threads must be 1 or larger.")

        for _ in self._progress(epochs, verbose=verbose):
            self._run_epoch(
                item_features,
                user_features,
                interactions,
                sample_weight_data,
                num_threads,
                self.loss,
            )

            self._check_finite()

        return self

    def _run_epoch(
        self,
        item_features,
        user_features,
        interactions,
        sample_weight,
        num_threads,
        loss,
    ):
        """
        Run an individual epoch.
        """

        if loss in ("warp", "bpr", "warp-kos"):
            # The CSR conversion needs to happen before shuffle indices are created.
            # Calling .tocsr may result in a change in the data arrays of the COO matrix,
            positives_lookup = CSRMatrix(
                self._get_positives_lookup_matrix(interactions)
            )

        # Create shuffle indexes.
        shuffle_indices = np.arange(len(interactions.data), dtype=np.int32)
        self.random_state.shuffle(shuffle_indices)

        lightfm_data = self._get_lightfm_data()

        # Call the estimation routines.
        if loss == "warp":
            fit_warp(
                CSRMatrix(item_features),
                CSRMatrix(user_features),
                positives_lookup,
                interactions.row,
                interactions.col,
                interactions.data,
                sample_weight,
                shuffle_indices,
                lightfm_data,
                self.learning_rate,
                self.item_alpha,
                self.user_alpha,
                num_threads,
                self.random_state,
            )
        elif loss == "bpr":
            fit_bpr(
                CSRMatrix(item_features),
                CSRMatrix(user_features),
                positives_lookup,
                interactions.row,
                interactions.col,
                interactions.data,
                sample_weight,
                shuffle_indices,
                lightfm_data,
                self.learning_rate,
                self.item_alpha,
                self.user_alpha,
                num_threads,
                self.random_state,
            )
        elif loss == "warp-kos":
            fit_warp_kos(
                CSRMatrix(item_features),
                CSRMatrix(user_features),
                positives_lookup,
                interactions.row,
                shuffle_indices,
                lightfm_data,
                self.learning_rate,
                self.item_alpha,
                self.user_alpha,
                self.k,
                self.n,
                num_threads,
                self.random_state,
            )
        else:
            fit_logistic(
                CSRMatrix(item_features),
                CSRMatrix(user_features),
                interactions.row,
                interactions.col,
                interactions.data,
                sample_weight,
                shuffle_indices,
                lightfm_data,
                self.learning_rate,
                self.item_alpha,
                self.user_alpha,
                num_threads,
            )

    def predict(
        self, user_ids, item_ids, item_features=None, user_features=None, num_threads=1
    ):
        """
        Compute the recommendation score for user-item pairs.

        For details on how to use feature matrices, see the documentation
        on the :class:`lightfm.LightFM` class.

        Arguments
        ---------

        user_ids: integer or np.int32 array of shape [n_pairs,]
             single user id or an array containing the user ids for the
             user-item pairs for which a prediction is to be computed. Note
             that these are LightFM's internal id's, i.e. the index of the
             user in the interaction matrix used for fitting the model.
        item_ids: np.int32 array of shape [n_pairs,]
             an array containing the item ids for the user-item pairs for which
             a prediction is to be computed. Note that these are LightFM's
             internal id's, i.e. the index of the item in the interaction
             matrix used for fitting the model.
        user_features: np.float32 csr_matrix of shape [n_users, n_user_features], optional
             Each row contains that user's weights over features.
        item_features: np.float32 csr_matrix of shape [n_items, n_item_features], optional
             Each row contains that item's weights over features.
        num_threads: int, optional
             Number of parallel computation threads to use. Should
             not be higher than the number of physical cores.

        Returns
        -------

        np.float32 array of shape [n_pairs,]
            Numpy array containing the recommendation scores for pairs defined
            by the inputs.

        Notes
        -----

        As indicated above, this method returns an array of scores corresponding to the
        score assigned by the model to _pairs of inputs_. Importantly, this means the
        i-th element of the output array corresponds to the score for the i-th user-item
        pair in the input arrays.

        Concretely, you should expect the `lfm.predict([0, 1], [8, 9])` to return an
        array of np.float32 that may look something like `[0.42  0.31]`, where `0.42` is
        the score assigned to the user-item pair `(0, 8)` and `0.31` the score assigned
        to pair `(1, 9)` respectively.

        In other words, if you wish to generate the score for a few items (e.g.
        `[7, 8, 9]`) for two users (e.g. `[0, 1]`), a proper way to call this method
        would be to use `lfm.predict([0, 0, 0, 1, 1, 1], [7, 8, 9, 7, 8, 9])`, and
        _not_ `lfm.predict([0, 1], [7, 8, 9])` as you may initially expect (this will
        throw an exception!).

        """

        self._check_initialized()

        if isinstance(user_ids, int):
            user_ids = np.repeat(np.int32(user_ids), len(item_ids))

        if isinstance(user_ids, (list, tuple)):
            user_ids = np.array(user_ids, dtype=np.int32)

        if isinstance(item_ids, (list, tuple)):
            item_ids = np.array(item_ids, dtype=np.int32)

        if len(user_ids) != len(item_ids):
            raise ValueError(
                f"Expected the number of user IDs ({len(user_ids)}) to equal the number"
                f" of item IDs ({len(item_ids)})"
            )

        if user_ids.dtype != np.int32:
            user_ids = user_ids.astype(np.int32)
        if item_ids.dtype != np.int32:
            item_ids = item_ids.astype(np.int32)

        if num_threads < 1:
            raise ValueError("Number of threads must be 1 or larger.")

        if user_ids.min() < 0 or item_ids.min() < 0:
            raise ValueError(
                "User or item ids cannot be negative. "
                "Check your inputs for negative numbers "
                "or very large numbers that can overflow."
            )

        n_users = user_ids.max() + 1
        n_items = item_ids.max() + 1

        (user_features, item_features) = self._construct_feature_matrices(
            n_users, n_items, user_features, item_features
        )

        lightfm_data = self._get_lightfm_data()

        predictions = np.empty(len(user_ids), dtype=np.float32)

        predict_lightfm(
            CSRMatrix(item_features),
            CSRMatrix(user_features),
            user_ids,
            item_ids,
            predictions,
            lightfm_data,
            num_threads,
        )

        return predictions

    def _check_test_train_intersections(self, test_mat, train_mat):
        if train_mat is not None:
            n_intersections = test_mat.multiply(train_mat).nnz
            if n_intersections:
                raise ValueError(
                    "Test interactions matrix and train interactions "
                    "matrix share %d interactions. This will cause "
                    "incorrect evaluation, check your data split." % n_intersections
                )

    def predict_rank(
        self,
        test_interactions,
        train_interactions=None,
        item_features=None,
        user_features=None,
        num_threads=1,
        check_intersections=True,
    ):
        """
        Predict the rank of selected interactions. Computes recommendation
        rankings across all items for every user in interactions and calculates
        the rank of all non-zero entries in the recommendation ranking, with 0
        meaning the top of the list (most recommended) and n_items - 1 being
        the end of the list (least recommended).

        Performs best when only a handful of interactions need to be evaluated
        per user. If you need to compute predictions for many items for every
        user, use the predict method instead.

        For details on how to use feature matrices, see the documentation
        on the :class:`lightfm.LightFM` class.

        Arguments
        ---------

        test_interactions: np.float32 csr_matrix of shape [n_users, n_items]
             Non-zero entries denote the user-item pairs
             whose rank will be computed.
        train_interactions: np.float32 csr_matrix of shape [n_users, n_items], optional
             Non-zero entries denote the user-item pairs which will be excluded
             from rank computation. Use to exclude training set interactions
             from being scored and ranked for evaluation.
        user_features: np.float32 csr_matrix of shape [n_users, n_user_features], optional
             Each row contains that user's weights over features.
        item_features: np.float32 csr_matrix of shape [n_items, n_item_features], optional
             Each row contains that item's weights over features.
        num_threads: int, optional
             Number of parallel computation threads to use.
             Should not be higher than the number of physical cores.
        check_intersections: bool, optional, True by default,
            Only relevant when train_interactions are supplied.
            A flag that signals whether the test and train matrices should be checked
            for intersections to prevent optimistic ranks / wrong evaluation / bad data split.

        Returns
        -------

        np.float32 csr_matrix of shape [n_users, n_items]
            the [i, j]-th entry of the matrix will contain the rank of the j-th
            item in the sorted recommendations list for the i-th user.
            The degree of sparsity of this matrix will be equal to that of the
            input interactions matrix.
        """

        self._check_initialized()

        if num_threads < 1:
            raise ValueError("Number of threads must be 1 or larger.")

        if check_intersections:
            self._check_test_train_intersections(test_interactions, train_interactions)

        n_users, n_items = test_interactions.shape

        (user_features, item_features) = self._construct_feature_matrices(
            n_users, n_items, user_features, item_features
        )

        if not item_features.shape[1] == self.item_embeddings.shape[0]:
            raise ValueError("Incorrect number of features in item_features")

        if not user_features.shape[1] == self.user_embeddings.shape[0]:
            raise ValueError("Incorrect number of features in user_features")

        test_interactions = test_interactions.tocsr()
        test_interactions = self._to_cython_dtype(test_interactions)

        if train_interactions is None:
            train_interactions = sp.csr_matrix((n_users, n_items), dtype=CYTHON_DTYPE)
        else:
            train_interactions = train_interactions.tocsr()
            train_interactions = self._to_cython_dtype(train_interactions)

        ranks = sp.csr_matrix(
            (
                np.zeros_like(test_interactions.data),
                test_interactions.indices,
                test_interactions.indptr,
            ),
            shape=test_interactions.shape,
        )

        lightfm_data = self._get_lightfm_data()

        predict_ranks(
            CSRMatrix(item_features),
            CSRMatrix(user_features),
            CSRMatrix(test_interactions),
            CSRMatrix(train_interactions),
            ranks.data,
            lightfm_data,
            num_threads,
        )

        return ranks

    def get_item_representations(self, features=None):
        """
        Get the latent representations for items given model and features.

        Arguments
        ---------

        features: np.float32 csr_matrix of shape [n_items, n_item_features], optional
             Each row contains that item's weights over features.
             An identity matrix will be used if not supplied.

        Returns
        -------

        (item_biases, item_embeddings):
                (np.float32 array of shape n_items,
                 np.float32 array of shape [n_items, num_components]
            Biases and latent representations for items.
        """

        self._check_initialized()

        if features is None:
            return self.item_biases, self.item_embeddings

        features = sp.csr_matrix(features, dtype=CYTHON_DTYPE)

        return features * self.item_biases, features * self.item_embeddings

    def get_user_representations(self, features=None):
        """
        Get the latent representations for users given model and features.

        Arguments
        ---------

        features: np.float32 csr_matrix of shape [n_users, n_user_features], optional
             Each row contains that user's weights over features.
             An identity matrix will be used if not supplied.

        Returns
        -------

        (user_biases, user_embeddings):
                (np.float32 array of shape n_users
                 np.float32 array of shape [n_users, num_components]
            Biases and latent representations for users.
        """

        self._check_initialized()

        if features is None:
            return self.user_biases, self.user_embeddings

        features = sp.csr_matrix(features, dtype=CYTHON_DTYPE)

        return features * self.user_biases, features * self.user_embeddings

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Arguments
        ---------

        deep: boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------

        params : mapping of string to any
            Parameter names mapped to their values.
        """

        params = {
            "loss": self.loss,
            "learning_schedule": self.learning_schedule,
            "no_components": self.no_components,
            "learning_rate": self.learning_rate,
            "k": self.k,
            "n": self.n,
            "rho": self.rho,
            "epsilon": self.epsilon,
            "max_sampled": self.max_sampled,
            "item_alpha": self.item_alpha,
            "user_alpha": self.user_alpha,
            "random_state": self.random_state,
        }

        return params

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Returns
        -------

        self
        """

        valid_params = self.get_params()

        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    "Invalid parameter %s for estimator %s. "
                    "Check the list of available parameters "
                    "with `estimator.get_params().keys()`."
                    % (key, self.__class__.__name__)
                )

            setattr(self, key, value)

        return self

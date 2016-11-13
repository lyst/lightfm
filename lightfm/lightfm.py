from __future__ import print_function

import numpy as np

import scipy.sparse as sp

from ._lightfm_fast import (CSRMatrix, FastLightFM,
                            fit_bpr, fit_logistic, fit_warp,
                            fit_warp_kos, predict_lightfm, predict_ranks)


__all__ = ['LightFM']


CYTHON_DTYPE = np.float32


class LightFM(object):
    """
    A hybrid recommender model.

    Parameters
    ----------

    no_components: int, optional
        the dimensionality of the feature latent embeddings.
    k: int, optional
         for k-OS training, the k-th positive example will be selected from the n positive
         examples sampled for every user.
    n: int, optional
         for k-OS training, maximum number of positives sampled for each update.
    learning_schedule: string, optional
         one of ('adagrad', 'adadelta').
    loss: string, optional
         one of  ('logistic', 'bpr', 'warp', 'warp-kos'): the loss function to use.
    learning_rate: float, optional
         initial learning rate for the adagrad learning schedule.
    rho: float, optional
         moving average coefficient for the adadelta learning schedule.
    epsilon: float, optional
        conditioning parameter for the adadelta learning schedule.
    item_alpha: float, optional
        L2 penalty on item features
    user_alpha: float, optional
        L2 penalty on user features.
    max_sampled: int, optional
        maximum number of negative samples used during WARP fitting. It requires
        a lot of sampling to find negative triplets for users that are already
        well represented by the model; this can lead to very long training times
        and overfitting. Setting this to a higher number will generally lead
        to longer training times, but may in some cases improve accuracy.
    random_state: int seed, RandomState instance, or None
        The seed of the pseudo random number generator to use when shuffling the data and
        initializing the parameters.

    Attributes
    ----------

    item_embeddings: np.float32 array of shape [n_item_features, n_components]
         Contains the estimated latent vectors for item features. The [i, j]-th entry
         gives the value of the j-th component for the i-th item feature. In the simplest
         case where the item feature matrix is an identity matrix, the i-th row
         will represent the i-th item latent vector.
    user_embeddings: np.float32 array of shape [n_user_features, n_components]
         Contains the estimated latent vectors for user features. The [i, j]-th entry
         gives the value of the j-th component for the i-th user feature. In the simplest
         case where the user feature matrix is an identity matrix, the i-th row
         will represent the i-th user latent vector.
    item_biases: np.float32 array of shape [n_item_features,]
         Contains the biases for item_features.
    user_biases: np.float32 array of shape [n_user_features,]
         Contains the biases for user_features.

    Notes
    -----

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
    - k-OS WARP: k-th order statistic loss [3]_. A modification of WARP that uses the k-th
      positive example for any given user as a basis for pairwise updates.

    Two learning rate schedules are available:

    - adagrad: [4]_
    - adadelta: [5]_

    References
    ----------

    .. [1] Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from implicit feedback."
           Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial
           Intelligence. AUAI Press, 2009.
    .. [2] Weston, Jason, Samy Bengio, and Nicolas Usunier. "Wsabie: Scaling up to large
           vocabulary image annotation." IJCAI. Vol. 11. 2011.
    .. [3] Weston, Jason, Hector Yee, and Ron J. Weiss. "Learning to rank recommendations with
           the k-order statistic loss."
           Proceedings of the 7th ACM conference on Recommender systems. ACM, 2013.
    .. [4] Duchi, John, Elad Hazan, and Yoram Singer. "Adaptive subgradient methods
           for online learning and stochastic optimization."
           The Journal of Machine Learning Research 12 (2011): 2121-2159.
    .. [5] Zeiler, Matthew D. "ADADELTA: An adaptive learning rate method."
           arXiv preprint arXiv:1212.5701 (2012).
    """

    def __init__(self, no_components=10, k=5, n=10,
                 learning_schedule='adagrad',
                 loss='logistic',
                 learning_rate=0.05, rho=0.95, epsilon=1e-6,
                 item_alpha=0.0, user_alpha=0.0, max_sampled=10,
                 random_state=None):

        assert item_alpha >= 0.0
        assert user_alpha >= 0.0
        assert no_components > 0
        assert k > 0
        assert n > 0
        assert 0 < rho < 1
        assert epsilon >= 0
        assert learning_schedule in ('adagrad', 'adadelta')
        assert loss in ('logistic', 'warp', 'bpr', 'warp-kos')

        if max_sampled < 1:
            raise ValueError('max_sampled must be a positive integer')

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

    def _initialize(self, no_components, no_item_features, no_user_features):
        """
        Initialise internal latent representations.
        """

        # Initialise item features.
        self.item_embeddings = ((self.random_state.rand(no_item_features, no_components) - 0.5) /
                                no_components).astype(np.float32)
        self.item_embedding_gradients = np.zeros_like(self.item_embeddings)
        self.item_embedding_momentum = np.zeros_like(self.item_embeddings)
        self.item_biases = np.zeros(no_item_features, dtype=np.float32)
        self.item_bias_gradients = np.zeros_like(self.item_biases)
        self.item_bias_momentum = np.zeros_like(self.item_biases)

        # Initialise user features.
        self.user_embeddings = ((self.random_state.rand(no_user_features, no_components) - 0.5) /
                                no_components).astype(np.float32)
        self.user_embedding_gradients = np.zeros_like(self.user_embeddings)
        self.user_embedding_momentum = np.zeros_like(self.user_embeddings)
        self.user_biases = np.zeros(no_user_features, dtype=np.float32)
        self.user_bias_gradients = np.zeros_like(self.user_biases)
        self.user_bias_momentum = np.zeros_like(self.user_biases)

        if self.learning_schedule == 'adagrad':
            self.item_embedding_gradients += 1
            self.item_bias_gradients += 1
            self.user_embedding_gradients += 1
            self.user_bias_gradients += 1

    def _construct_feature_matrices(self, n_users, n_items, user_features,
                                    item_features):

        if user_features is None:
            user_features = sp.identity(n_users,
                                        dtype=CYTHON_DTYPE,
                                        format='csr')
        else:
            user_features = user_features.tocsr()

        if item_features is None:
            item_features = sp.identity(n_items,
                                        dtype=CYTHON_DTYPE,
                                        format='csr')
        else:
            item_features = item_features.tocsr()

        if n_users > user_features.shape[0]:
            raise Exception('Number of user feature rows does not equal '
                            'the number of users')

        if n_items > item_features.shape[0]:
            raise Exception('Number of item feature rows does not equal '
                            'the number of items')

        # If we already have embeddings, verify that
        # we have them for all the supplied features
        if self.user_embeddings is not None:
            assert self.user_embeddings.shape[0] >= user_features.shape[1]

        if self.item_embeddings is not None:
            assert self.item_embeddings.shape[0] >= item_features.shape[1]

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

            if self.loss == 'warp-kos':
                raise NotImplementedError('k-OS loss with sample weights '
                                          'not implemented.')

            if not isinstance(sample_weight, sp.coo_matrix):
                raise ValueError('Sample_weight must be a COO matrix.')

            if sample_weight.shape != interactions.shape:
                raise ValueError('Sample weight and interactions '
                                 'matrices must be the same shape')

            if not (np.array_equal(interactions.row,
                                   sample_weight.row) and
                    np.array_equal(interactions.col,
                                   sample_weight.col)):
                raise ValueError('Sample weight and interaction matrix '
                                 'entries must be in the same order')

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
                sample_weight_data = np.ones_like(interactions.data,
                                                  dtype=CYTHON_DTYPE)

        return sample_weight_data

    def _get_lightfm_data(self):

        lightfm_data = FastLightFM(self.item_embeddings,
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
                                   int(self.learning_schedule == 'adadelta'),
                                   self.learning_rate,
                                   self.rho,
                                   self.epsilon,
                                   self.max_sampled)

        return lightfm_data

    def fit(self, interactions,
            user_features=None, item_features=None,
            sample_weight=None,
            epochs=1, num_threads=1, verbose=False):
        """
        Fit the model.

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
             efficiency its ssible to use the same arrays
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

        Returns
        -------

        LightFM instance
            the fitted model

        """

        # Discard old results, if any
        self._reset_state()

        return self.fit_partial(interactions,
                                user_features=user_features,
                                item_features=item_features,
                                sample_weight=sample_weight,
                                epochs=epochs,
                                num_threads=num_threads,
                                verbose=verbose)

    def fit_partial(self, interactions,
                    user_features=None, item_features=None,
                    sample_weight=None,
                    epochs=1, num_threads=1, verbose=False):
        """
        Fit the model.

        Fit the model. Unlike fit, repeated calls to this method will
        cause trainig to resume from the current model state.

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
             efficiency its ssible to use the same arrays
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

        sample_weight_data = self._process_sample_weight(interactions,
                                                         sample_weight)

        n_users, n_items = interactions.shape
        (user_features,
         item_features) = self._construct_feature_matrices(n_users,
                                                           n_items,
                                                           user_features,
                                                           item_features)

        user_features = self._to_cython_dtype(user_features)
        item_features = self._to_cython_dtype(item_features)
        sample_weight = (self._to_cython_dtype(sample_weight)
                         if sample_weight is not None else
                         np.ones(interactions.getnnz(),
                                 dtype=CYTHON_DTYPE))

        if self.item_embeddings is None:
            # Initialise latent factors only if this is the first call
            # to fit_partial.
            self._initialize(self.no_components,
                             item_features.shape[1],
                             user_features.shape[1])

        # Check that the dimensionality of the feature matrices has
        # not changed between runs.
        if not item_features.shape[1] == self.item_embeddings.shape[0]:
            raise ValueError('Incorrect number of features in item_features')

        if not user_features.shape[1] == self.user_embeddings.shape[0]:
            raise ValueError('Incorrect number of features in user_features')

        for epoch in range(epochs):

            if verbose:
                print('Epoch %s' % epoch)

            self._run_epoch(item_features,
                            user_features,
                            interactions,
                            sample_weight_data,
                            num_threads,
                            self.loss)

        return self

    def _run_epoch(self, item_features, user_features, interactions,
                   sample_weight, num_threads, loss):
        """
        Run an individual epoch.
        """

        if loss in ('warp', 'bpr', 'warp-kos'):
            # The CSR conversion needs to happen before shuffle indices are created.
            # Calling .tocsr may result in a change in the data arrays of the COO matrix,
            positives_lookup = CSRMatrix(self._get_positives_lookup_matrix(interactions))

        # Create shuffle indexes.
        shuffle_indices = np.arange(len(interactions.data), dtype=np.int32)
        self.random_state.shuffle(shuffle_indices)

        lightfm_data = self._get_lightfm_data()

        # Call the estimation routines.
        if loss == 'warp':
            fit_warp(CSRMatrix(item_features),
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
                     self.random_state)
        elif loss == 'bpr':
            fit_bpr(CSRMatrix(item_features),
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
                    self.random_state)
        elif loss == 'warp-kos':
            fit_warp_kos(CSRMatrix(item_features),
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
                         self.random_state)
        else:
            fit_logistic(CSRMatrix(item_features),
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
                         num_threads)

    def predict(self, user_ids, item_ids, item_features=None, user_features=None, num_threads=1):
        """
        Compute the recommendation score for user-item pairs.

        Arguments
        ---------

        user_ids: integer or np.int32 array of shape [n_pairs,]
             single user id or an array containing the user ids for the user-item pairs for which
             a prediction is to be computed
        item_ids: np.int32 array of shape [n_pairs,]
             an array containing the item ids for the user-item pairs for which
             a prediction is to be computed.
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
            Numpy array containig the recommendation scores for pairs defined by the inputs.
        """

        if not isinstance(user_ids, np.ndarray):
            user_ids = np.repeat(np.int32(user_ids), len(item_ids))

        assert len(user_ids) == len(item_ids)

        if user_ids.dtype != np.int32:
            user_ids = user_ids.astype(np.int32)
        if item_ids.dtype != np.int32:
            item_ids = item_ids.astype(np.int32)

        n_users = user_ids.max() + 1
        n_items = item_ids.max() + 1

        (user_features,
         item_features) = self._construct_feature_matrices(n_users,
                                                           n_items,
                                                           user_features,
                                                           item_features)

        user_features = self._to_cython_dtype(user_features)
        item_features = self._to_cython_dtype(item_features)

        lightfm_data = self._get_lightfm_data()

        predictions = np.empty(len(user_ids), dtype=np.float64)

        predict_lightfm(CSRMatrix(item_features),
                        CSRMatrix(user_features),
                        user_ids,
                        item_ids,
                        predictions,
                        lightfm_data,
                        num_threads)

        return predictions

    def predict_rank(self, test_interactions, train_interactions=None,
                     item_features=None, user_features=None, num_threads=1):
        """
        Predict the rank of selected interactions. Computes recommendation rankings across all items
        for every user in interactions and calculates the rank of all non-zero entries
        in the recommendation ranking, with 0 meaning the top of the list (most recommended)
        and n_items - 1 being the end of the list (least recommended).

        Performs best when only a handful of interactions need to be evaluated per user. If you
        need to compute predictions for many items for every user, use the predict method instead.

        Arguments
        ---------

        test_interactions: np.float32 csr_matrix of shape [n_users, n_items]
             Non-zero entries denote the user-item pairs whose rank will be computed.
        train_interactions: np.float32 csr_matrix of shape [n_users, n_items], optional
             Non-zero entries denote the user-item pairs which will be excluded from
             rank computation. Use to exclude training set interactions from being scored
             and ranked for evaluation.
        user_features: np.float32 csr_matrix of shape [n_users, n_user_features], optional
             Each row contains that user's weights over features.
        item_features: np.float32 csr_matrix of shape [n_items, n_item_features], optional
             Each row contains that item's weights over features.
        num_threads: int, optional
             Number of parallel computation threads to use. Should
             not be higher than the number of physical cores.

        Returns
        -------

        np.float32 csr_matrix of shape [n_users, n_items]
            the [i, j]-th entry of the matrix will contain the rank of the j-th item
            in the sorted recommendations list for the i-th user. The degree
            of sparsity of this matrix will be equal to that of the input interactions
            matrix.
        """

        n_users, n_items = test_interactions.shape

        (user_features,
         item_features) = self._construct_feature_matrices(n_users,
                                                           n_items,
                                                           user_features,
                                                           item_features)

        if not item_features.shape[1] == self.item_embeddings.shape[0]:
            raise ValueError('Incorrect number of features in item_features')

        if not user_features.shape[1] == self.user_embeddings.shape[0]:
            raise ValueError('Incorrect number of features in user_features')

        test_interactions = test_interactions.tocsr()
        test_interactions = self._to_cython_dtype(test_interactions)

        if train_interactions is None:
            train_interactions = sp.csr_matrix((n_users, n_items),
                                               dtype=CYTHON_DTYPE)
        else:
            train_interactions = train_interactions.tocsr()
            train_interactions = self._to_cython_dtype(train_interactions)

        ranks = sp.csr_matrix((np.zeros_like(test_interactions.data),
                               test_interactions.indices,
                               test_interactions.indptr),
                              shape=test_interactions.shape)

        lightfm_data = self._get_lightfm_data()

        predict_ranks(CSRMatrix(item_features),
                      CSRMatrix(user_features),
                      CSRMatrix(test_interactions),
                      CSRMatrix(train_interactions),
                      ranks.data,
                      lightfm_data,
                      num_threads)

        return ranks

    def get_params(self, deep=True):
        """Get parameters for this estimator.

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
        params = {'loss': self.loss,
                  'learning_schedule': self.learning_schedule,
                  'no_components': self.no_components,
                  'learning_rate': self.learning_rate,
                  'k': self.k,
                  'n': self.n,
                  'rho': self.rho,
                  'epsilon': self.epsilon,
                  'max_sampled': self.max_sampled,
                  'item_alpha': self.item_alpha,
                  'user_alpha': self.user_alpha,
                  'random_state': self.random_state}
        return params

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Returns
        -------

        self
        """
        valid_params = self.get_params()
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self.__class__.__name__))
            setattr(self, key, value)
        return self

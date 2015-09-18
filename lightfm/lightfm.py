from __future__ import print_function

import numpy as np

import scipy.sparse as sp

from .lightfm_fast import (CSRMatrix, FastLightFM,
                           fit_logistic, predict_lightfm,
                           fit_warp, fit_bpr, fit_warp_kos)


CYTHON_DTYPE = np.int32


class LightFM(object):

    def __init__(self, no_components=10, k=5, n=10,
                 learning_schedule='adagrad',
                 loss='logistic',
                 learning_rate=0.05, rho=0.95, epsilon=1e-6,
                 item_alpha=0.0, user_alpha=0.0):
        """
        Initialise the model.

        Four loss functions are available:
        - logistic: useful when both positive (1) and negative (-1) interactions
                    are present.
        - BPR: Bayesian Personalised Ranking [1] pairwise loss. Maximises the
               prediction difference between a positive example and a randomly
               chosen negative example. Useful when only positive interactions
               are present and optimising ROC AUC is desired.
        - WARP: Weighted Approximate-Rank Pairwise [2] loss. Maximises
                the rank of positive examples by repeatedly sampling negative
                examples until rank violating one is found. Useful when only
                positive interactions are present and optimising the top of
                the recommendation list (precision@k) is desired.
        - k-OS WARP: k-th order statistic loss [3]. A modification of WARP that uses the k-th
                     positive example for any given user as a basis for pairwise updates.

        Two learning rate schedules are available:
        - adagrad: [4]
        - adadelta: [5]

        Parameters:
        - integer no_components: the dimensionality of the feature latent embeddings. Default: 10
        - int k: for k-OS training, the k-th positive example will be selected from the n positive
                 examples sampled for every user. Default: 5
        - int n: for k-OS training, maximum number of positives sampled for each update. Default: 10
        - string learning_schedule, one of ('adagrad', 'adadelta'). Default: 'adagrad'
        - string loss ('logistic', 'bpr', 'warp', 'warp-kos'): the loss function to use. Default: 'logistic'
        - float learning_rate: initial learning rate for the adagrad learning schedule. Default: 0.05
        - float rho: moving average coefficient for the adadelta learning schedule. Default: 0.95
        - float epsilon: conditioning parameter for the adadelta learning schedule. Default: 1e-6
        - float item_alpha: L2 penalty on item features. Default: 0.0
        - float user_alpha: L2 penalty on user features. Default: 0.0

        [1] Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from implicit feedback."
            Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial
            Intelligence. AUAI Press, 2009.
        [2] Weston, Jason, Samy Bengio, and Nicolas Usunier. "Wsabie: Scaling up to large
            vocabulary image annotation." IJCAI. Vol. 11. 2011.
        [3] Weston, Jason, Hector Yee, and Ron J. Weiss. "Learning to rank recommendations with
            the k-order statistic loss."
            Proceedings of the 7th ACM conference on Recommender systems. ACM, 2013.
        [4] Duchi, John, Elad Hazan, and Yoram Singer. "Adaptive subgradient methods
            for online learning and stochastic optimization."
            The Journal of Machine Learning Research 12 (2011): 2121-2159.
        [5] Zeiler, Matthew D. "ADADELTA: An adaptive learning rate method."
            arXiv preprint arXiv:1212.5701 (2012).
        """

        assert item_alpha >= 0.0
        assert user_alpha >= 0.0
        assert no_components > 0
        assert k > 0
        assert n > 0
        assert 0 < rho < 1
        assert epsilon >= 0
        assert learning_schedule in ('adagrad', 'adadelta')
        assert loss in ('logistic', 'warp', 'bpr', 'warp-kos')

        self.loss = loss
        self.learning_schedule = learning_schedule

        self.no_components = no_components
        self.learning_rate = learning_rate

        self.k = int(k)
        self.n = int(n)

        self.rho = rho
        self.epsilon = epsilon

        self.item_alpha = item_alpha
        self.user_alpha = user_alpha

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
        self.item_embeddings = ((np.random.rand(no_item_features, no_components) - 0.5)
                                / no_components).astype(np.float32)
        self.item_embedding_gradients = np.zeros_like(self.item_embeddings)
        self.item_embedding_momentum = np.zeros_like(self.item_embeddings)
        self.item_biases = np.zeros(no_item_features, dtype=np.float32)
        self.item_bias_gradients = np.zeros_like(self.item_biases)
        self.item_bias_momentum = np.zeros_like(self.item_biases)

        # Initialise user features.
        self.user_embeddings = ((np.random.rand(no_user_features, no_components) - 0.5)
                                / no_components).astype(np.float32)
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
                                        dtype=np.int32).tocsr()
        else:
            user_features = user_features.tocsr()

        if item_features is None:
            item_features = sp.identity(n_items,
                                        dtype=np.int32).tocsr()
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

    def fit(self, interactions, user_features=None, item_features=None,
            epochs=1, num_threads=1, verbose=False):

        # Discard old results, if any
        self._reset_state()

        return self.fit_partial(interactions,
                                user_features=user_features,
                                item_features=item_features,
                                epochs=epochs,
                                num_threads=num_threads,
                                verbose=verbose)

    def fit_partial(self, interactions, user_features=None, item_features=None,
                    epochs=1, num_threads=1, verbose=False):
        """
        Fit the model. Repeated calls to this function will resume training from
        the point where the last call finished.

        Arguments:
        - coo_matrix interactions: matrix of shape [n_users, n_items] containing
                                   user-item interactions
        - csr_matrix user_features: array of shape [n_users, n_user_features].
                                    Each row contains that user's weights
                                    over features.
        - csr_matrix item_features: array of shape [n_items, n_item_features].
                                    Each row contains that item's weights
                                    over features.

        - int epochs: number of epochs to run. Default: 1
        - int num_threads: number of parallel computation threads to use. Should
                           not be higher than the number of physical cores.
                           Default: 1
        - bool verbose: whether to print progress messages.
        """

        # We need this in the COO format.
        # If that's already true, this is a no-op.
        interactions = interactions.tocoo()

        n_users, n_items = interactions.shape
        (user_features,
         item_features) = self._construct_feature_matrices(n_users,
                                                           n_items,
                                                           user_features,
                                                           item_features)

        interactions = self._to_cython_dtype(interactions)
        user_features = self._to_cython_dtype(user_features)
        item_features = self._to_cython_dtype(item_features)

        if self.item_embeddings is None:
            # Initialise latent factors only if this is the first call
            # to fit_partial.
            self._initialize(self.no_components,
                             item_features.shape[1],
                             user_features.shape[1])

        # Check that the dimensionality of the feature matrices has
        # not changed between runs.
        if not item_features.shape[1] == self.item_embeddings.shape[0]:
            raise Exception('Incorrect number of features in item_features')

        if not user_features.shape[1] == self.user_embeddings.shape[0]:
            raise Exception('Incorrect number of features in user_features')

        for epoch in range(epochs):

            if verbose:
                print('Epoch %s' % epoch)

            self._run_epoch(item_features,
                            user_features,
                            interactions,
                            num_threads,
                            self.loss)

        return self

    def _run_epoch(self, item_features, user_features, interactions, num_threads, loss):
        """
        Run an individual epoch.
        """

        # Create shuffle indexes.
        shuffle_indices = np.arange(len(interactions.data), dtype=np.int32)
        np.random.shuffle(shuffle_indices)

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
                                   self.epsilon)

        # Call the estimation routines.
        if loss == 'warp':
            fit_warp(CSRMatrix(item_features),
                     CSRMatrix(user_features),
                     CSRMatrix(self._get_positives_lookup_matrix(interactions)),
                     interactions.row,
                     interactions.col,
                     interactions.data,
                     shuffle_indices,
                     lightfm_data,
                     self.learning_rate,
                     self.item_alpha,
                     self.user_alpha,
                     num_threads)
        elif loss == 'bpr':
            fit_bpr(CSRMatrix(item_features),
                    CSRMatrix(user_features),
                    CSRMatrix(self._get_positives_lookup_matrix(interactions)),
                    interactions.row,
                    interactions.col,
                    interactions.data,
                    shuffle_indices,
                    lightfm_data,
                    self.learning_rate,
                    self.item_alpha,
                    self.user_alpha,
                    num_threads)
        elif loss == 'warp-kos':
            fit_warp_kos(CSRMatrix(item_features),
                         CSRMatrix(user_features),
                         CSRMatrix(self._get_positives_lookup_matrix(interactions)),
                         interactions.row,
                         shuffle_indices,
                         lightfm_data,
                         self.learning_rate,
                         self.item_alpha,
                         self.user_alpha,
                         self.k,
                         self.n,
                         num_threads)
        else:
            fit_logistic(CSRMatrix(item_features),
                         CSRMatrix(user_features),
                         interactions.row,
                         interactions.col,
                         interactions.data,
                         shuffle_indices,
                         lightfm_data,
                         self.learning_rate,
                         self.item_alpha,
                         self.user_alpha,
                         num_threads)

    def predict(self, user_ids, item_ids, item_features=None, user_features=None, num_threads=1):
        """
        Predict the probability of a positive interaction.

        Arguments:
        - csr_matrix item_features: array of shape [n_samples, n_item_features].
                                    Each row contains that row's item's weights
                                    over features.
        - csr_matrix user_features: array of shape [n_samples, n_user_features].
                                    Each row contains that row's user's weights
                                    over features.
        - int num_threads: number of parallel computation threads to use. Should
                           not be higher than the number of physical cores.
                           Default: 1

        Returns:
        - numpy array predictions: [n_samples] array of positive class probabilities.
        """

        assert len(user_ids) == len(item_ids)

        n_users = user_ids.max() + 1
        n_items = item_ids.max() + 1

        (user_features,
         item_features) = self._construct_feature_matrices(n_users,
                                                           n_items,
                                                           user_features,
                                                           item_features)

        user_features = self._to_cython_dtype(user_features)
        item_features = self._to_cython_dtype(item_features)

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
                                   self.epsilon)

        predictions = np.empty(len(user_ids), dtype=np.float64)

        predict_lightfm(CSRMatrix(item_features),
                        CSRMatrix(user_features),
                        user_ids,
                        item_ids,
                        predictions,
                        lightfm_data,
                        num_threads)

        return predictions

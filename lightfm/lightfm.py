from __future__ import print_function

import numpy as np

import scipy.sparse as sp

from .lightfm_fast import (CSRMatrix, FastLightFM,
                           fit_lightfm, predict_lightfm)


class LightFM(object):

    def __init__(self, no_components=10, learning_rate=0.05, item_alpha=0.0, user_alpha=0.0):
        """
        Initialise the model.

        Parameters:
        - integer no_components: the dimensionality of the feature latent embeddings. Default: 10
        - float learning_rate: initial learning rate. Default: 0.05
        - float item_alpha: L2 penalty on item features. Default: 0.0
        - float user_alpha: L2 penalty on user features. Default: 0.0
        """

        assert item_alpha >= 0.0
        assert user_alpha >= 0.0
        assert no_components > 0

        self.no_components = no_components
        self.learning_rate = learning_rate

        self.item_alpha = item_alpha
        self.user_alpha = user_alpha

        self._reset_state()

    def _reset_state(self):

        self.item_embeddings = None
        self.item_embedding_gradients = None
        self.item_biases = None
        self.item_bias_gradients = None

        self.user_embeddings = None
        self.user_embedding_gradients = None
        self.user_biases = None
        self.user_bias_gradients = None

    def _initialize(self, no_components, no_item_features, no_user_features):
        """
        Initialise internal latent representations.
        """

        # Initialise item features.
        self.item_embeddings = ((np.random.rand(no_item_features, no_components) - 0.5)
                                / no_components)
        self.item_embedding_gradients = np.ones_like(self.item_embeddings)
        self.item_biases = np.zeros(no_item_features, dtype=np.float64)
        self.item_bias_gradients = np.ones_like(self.item_biases)

        # Initialise user features.
        self.user_embeddings = ((np.random.rand(no_user_features, no_components) - 0.5)
                                / no_components)
        self.user_embedding_gradients = np.ones_like(self.user_embeddings)
        self.user_biases = np.zeros(no_user_features, dtype=np.float64)
        self.user_bias_gradients = np.ones_like(self.user_biases)

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
        Fit the model.

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
                            num_threads)

        return self

    def _run_epoch(self, item_features, user_features, interactions, num_threads):
        """
        Run an individual epoch.
        """

        # Create shuffle indexes.
        shuffle_indices = np.arange(len(interactions.data), dtype=np.int32)
        np.random.shuffle(shuffle_indices)

        lightfm_data = FastLightFM(self.item_embeddings,
                                   self.item_embedding_gradients,
                                   self.item_biases,
                                   self.item_bias_gradients,
                                   self.user_embeddings,
                                   self.user_embedding_gradients,
                                   self.user_biases,
                                   self.user_bias_gradients,
                                   self.no_components)

        # Call the estimation routines.
        fit_lightfm(CSRMatrix(item_features),
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

        lightfm_data = FastLightFM(self.item_embeddings,
                                   self.item_embedding_gradients,
                                   self.item_biases,
                                   self.item_bias_gradients,
                                   self.user_embeddings,
                                   self.user_embedding_gradients,
                                   self.user_biases,
                                   self.user_bias_gradients,
                                   self.no_components)

        predictions = np.empty(len(user_ids), dtype=np.float64)

        predict_lightfm(CSRMatrix(item_features),
                        CSRMatrix(user_features),
                        user_ids,
                        item_ids,
                        predictions,
                        lightfm_data,
                        num_threads)

        return predictions

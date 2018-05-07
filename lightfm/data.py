import array

import numpy as np

import scipy.sparse as sp


class _IncrementalCOOMatrix(object):

    def __init__(self, shape, dtype):

        if dtype is np.int32:
            type_flag = "i"
        elif dtype is np.int64:
            type_flag = "l"
        elif dtype is np.float32:
            type_flag = "f"
        elif dtype is np.float64:
            type_flag = "d"
        else:
            raise Exception("Dtype not supported.")

        self.shape = shape
        self.dtype = dtype

        self.rows = array.array("i")
        self.cols = array.array("i")
        self.data = array.array(type_flag)

    def append(self, i, j, v):

        m, n = self.shape

        if i >= m or j >= n:
            raise Exception("Index out of bounds")

        self.rows.append(i)
        self.cols.append(j)
        self.data.append(v)

    def tocoo(self):

        rows = np.frombuffer(self.rows, dtype=np.int32)
        cols = np.frombuffer(self.cols, dtype=np.int32)
        data = np.frombuffer(self.data, dtype=self.dtype)

        return sp.coo_matrix((data, (rows, cols)), shape=self.shape)

    def __len__(self):

        return len(self.data)


class Dataset(object):
    """
    Tool for building interaction and feature matrices, taking care of the
    mapping between user/item ids and feature names and internal feature indices.

    To create a dataset:
    - Create an instance of the `Dataset` class.
    - Call `fit` (or `fit_partial`), supplying user/item ids and feature names
      that you want to use in your model. This will create internal mappings that
      translate the ids and feature names to internal indices used by the LightFM
      model.
    - Call `build_interactions` with an iterable of (user id, item id) or (user id,
      item id, weight) to build an interactions and weights matrix.
    - Call `build_user/item_features` with iterables of (user/item id, [features])
      or (user/item id, {feature: feature weight}) to build feature matrices.
    - To add new user/item ids or features, call `fit_partial` again. You will need
      to resize your LightFM model to be able to use the new features.

    Parameters
    ----------

    user_identity_features: bool, optional
        Create a unique feature for every user in addition to other features.
        If true (default), a latent vector will be allocated for every user. This
        is a reasonable default for most applications, but should be set to false
        if there is very little data for every user.
    item_identity_features: bool, optional
        Create a unique feature for every item in addition to other features.
        If true (default), a latent vector will be allocated for every item. This
        is a reasonable default for most applications, but should be set to false
        if there is very little data for every item.
    """

    def __init__(self, user_identity_features=True, item_identity_features=True):

        self._user_identity_features = user_identity_features
        self._item_identity_features = item_identity_features

        self._user_id_mapping = {}
        self._item_id_mapping = {}
        self._user_feature_mapping = {}
        self._item_feature_mapping = {}

    def _check_fitted(self):

        if not self._user_id_mapping or not self._item_id_mapping:
            raise ValueError("You must call fit first to build the item and user " "id mappings.")

    def fit(self, users, items, user_features=None, item_features=None):
        """
        Fit the user/item id and feature name mappings.

        Calling fit the second time will reset existing mappings.

        Parameters
        ----------

        users: iterable of user ids
        items: iterable of item ids
        user_features: iterable of user features, optional
        item_features: iterable of item features, optional
        """

        self._user_id_mapping = {}
        self._item_id_mapping = {}
        self._user_feature_mapping = {}
        self._item_feature_mapping = {}

        return self.fit_partial(users, items, user_features, item_features)

    def fit_partial(self, users=None, items=None, user_features=None, item_features=None):
        """
        Fit the user/item id and feature name mappings.

        Calling fit the second time will add new entries to existing mappings.

        Parameters
        ----------

        users: iterable of user ids, optional
        items: iterable of item ids, optional
        user_features: iterable of user features, optional
        item_features: iterable of item features, optional
        """

        if users is not None:
            for user_id in users:
                self._user_id_mapping.setdefault(user_id, len(self._user_id_mapping))

                if self._user_identity_features:
                    self._user_feature_mapping.setdefault(user_id, len(self._user_feature_mapping))

        if items is not None:
            for item_id in items:
                self._item_id_mapping.setdefault(item_id, len(self._item_id_mapping))

                if self._item_identity_features:
                    self._item_feature_mapping.setdefault(item_id, len(self._item_feature_mapping))

        if user_features is not None:
            for user_feature in user_features:
                self._user_feature_mapping.setdefault(user_feature, len(self._user_feature_mapping))

        if item_features is not None:
            for item_feature in item_features:
                self._item_feature_mapping.setdefault(item_feature, len(self._item_feature_mapping))

    def _unpack_datum(self, datum):

        if len(datum) == 3:
            (user_id, item_id, weight) = datum
        elif len(datum) == 2:
            (user_id, item_id) = datum
            weight = 1.0
        else:
            raise ValueError(
                "Expecting tuples of (user_id, item_id, weight) "
                "or (user_id, item_id). Got {}".format(datum)
            )

        user_idx = self._user_id_mapping.get(user_id)
        item_idx = self._item_id_mapping.get(item_id)

        if user_idx is None:
            raise ValueError(
                "User id {} not in user id mapping. Make sure "
                "you call the fit method.".format(user_id)
            )

        if item_idx is None:
            raise ValueError(
                "Item id {} not in item id mapping. Make sure "
                "you call the fit method.".format(item_id)
            )

        return (user_idx, item_idx, weight)

    def interactions_shape(self):
        """
        Return a tuple of (num users, num items).
        """

        return (len(self._user_id_mapping), len(self._item_id_mapping))

    def build_interactions(self, data):
        """
        Build an interaction matrix.

        Two matrices will be returned: a (num_users, num_items)
        COO matrix with interactions, and a (num_users, num_items)
        matrix with the corresponding interaction weights.

        Parameters
        ----------

        data: iterable of (user_id, item_id) or (user_id, item_id, weight)
            An iterable of interactions. The user and item ids will be
            translated to internal model indices using the mappings
            constructed during the fit call. If weights are not provided
            they will be assumed to be 1.0.

        Returns
        -------

        (interactions, weights): COO matrix, COO matrix
            Two COO matrices: the interactions matrix
            and the corresponding weights matrix.
        """

        interactions = _IncrementalCOOMatrix(self.interactions_shape(), np.int32)
        weights = _IncrementalCOOMatrix(self.interactions_shape(), np.float32)

        for datum in data:
            user_idx, item_idx, weight = self._unpack_datum(datum)

            interactions.append(user_idx, item_idx, 1)
            weights.append(user_idx, item_idx, weight)

        return (interactions.tocoo(), weights.tocoo())

    def _iter_features(self, features):

        if isinstance(features, dict):
            for entry in features.items():
                yield entry

        else:
            for feature_name in features:
                yield (feature_name, 1.0)

    def _process_user_features(self, datum):

        if len(datum) != 2:
            raise ValueError("Expected tuples of (user_id, features), " "got {}.".format(datum))

        user_id, features = datum

        if user_id not in self._user_id_mapping:
            raise ValueError("User id {} not in user id mappings.".format(user_id))

        user_idx = self._user_id_mapping[user_id]

        for (feature, weight) in self._iter_features(features):
            if feature not in self._user_feature_mapping:
                raise ValueError(
                    "Feature {} not in user feature mapping. " "Call fit first.".format(feature)
                )

            feature_idx = self._user_feature_mapping[feature]

            yield (user_idx, feature_idx, weight)

    def user_features_shape(self):
        """
        Return the shape of the user features matrix.

        Returns
        -------

        (num user ids, num user features): tuple of ints
            The shape.
        """

        return (len(self._user_id_mapping), len(self._user_feature_mapping))

    def build_user_features(self, data):
        """
        Build a user features matrix out of an iterable of the form
        (user id, [list of feature names]) or (user id, {feature name: feature weight}).

        Parameters
        ----------

        data: iterable of the form
            (user id, [list of feature names]) or (user id,
            {feature name: feature weight}).
            User and feature ids will be translated to internal indices
            constructed during the fit call.

        Returns
        -------

        feature matrix: COO matrix (num users, num features)
            Matrix of user features.
        """

        features = _IncrementalCOOMatrix(self.user_features_shape(), np.float32)

        if self._user_identity_features:
            for (user_id, user_idx) in self._user_id_mapping.items():
                features.append(user_idx, self._user_feature_mapping[user_id], 1.0)

        for datum in data:
            for (user_idx, feature_idx, weight) in self._process_user_features(datum):
                features.append(user_idx, feature_idx, weight)

        return features.tocoo()

    def _process_item_features(self, datum):

        if len(datum) != 2:
            raise ValueError("Expected tuples of (item_id, features), " "got {}.".format(datum))

        item_id, features = datum

        if item_id not in self._item_id_mapping:
            raise ValueError("Item id {} not in item id mappings.".format(item_id))

        item_idx = self._item_id_mapping[item_id]

        for (feature, weight) in self._iter_features(features):
            if feature not in self._item_feature_mapping:
                raise ValueError(
                    "Feature {} not in item feature mapping. " "Call fit first.".format(feature)
                )

            feature_idx = self._item_feature_mapping[feature]

            yield (item_idx, feature_idx, weight)

    def item_features_shape(self):
        """
        Return the shape of the item features matrix.

        Returns
        -------

        (num item ids, num item features): tuple of ints
            The shape.
        """

        return (len(self._item_id_mapping), len(self._item_feature_mapping))

    def build_item_features(self, data):
        """
        Build a item features matrix out of an iterable of the form
        (item id, [list of feature names]) or (item id, {feature name: feature weight}).

        Parameters
        ----------

        data: iterable of the form
            (item id, [list of feature names]) or (item id,
            {feature name: feature weight}).
            Item and feature ids will be translated to internal indices
            constructed during the fit call.

        Returns
        -------

        feature matrix: COO matrix (num items, num features)
            Matrix of item features.
        """

        features = _IncrementalCOOMatrix(self.item_features_shape(), np.float32)

        if self._item_identity_features:
            for (item_id, item_idx) in self._item_id_mapping.items():
                features.append(item_idx, self._item_feature_mapping[item_id], 1.0)

        for datum in data:
            for (item_idx, feature_idx, weight) in self._process_item_features(datum):
                features.append(item_idx, feature_idx, weight)

        return features.tocoo()

    def model_dimensions(self):
        """
        Returns a tuple that characterizes the number of user/item feature
        embeddings in a LightFM model for this dataset.
        """

        return (len(self._user_feature_mapping), len(self._item_feature_mapping))

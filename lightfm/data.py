import array

import scipy.sparse as sp


class IncrementalCOOMatrix(object):

    def __init__(self, shape, dtype):

        if dtype is np.int32:
            type_flag = 'i'
        elif dtype is np.int64:
            type_flag = 'l'
        elif dtype is np.float32:
            type_flag = 'f'
        elif dtype is np.float64:
            type_flag = 'd'
        else:
            raise Exception('Dtype not supported.')

        self.shape = shape
        self.dtype = dtype

        self.rows = array.array('i')
        self.cols = array.array('i')
        self.data = array.array(type_flag)

    def append(self, i, j, v):

        m, n = self.shape

        if (i >= m or j >= n):
            raise Exception('Index out of bounds')

        self.rows.append(i)
        self.cols.append(j)
        self.data.append(v)

    def tocoo(self):

        rows = np.frombuffer(self.rows, dtype=np.int32)
        cols = np.frombuffer(self.cols, dtype=np.int32)
        data = np.frombuffer(self.data, dtype=self.dtype)

        return sp.coo_matrix((data, (rows, cols)),
                             shape=self.shape)

    def __len__(self):

        return len(self.data)


class ImplicitInteractions(object):

    def __init__(self, user_identity_features=True, item_identity_features=True):

        self._user_identity_features = user_identity_features
        self._item_identity_features = item_identity_features

        self._user_id_mapping = {}
        self._item_id_mapping = {}

    def _initialise_features(self):

        self._user_feature_mapping = {}
        self._item_feature_mapping = {}

        self._user_features_row = array.array('i')
        self._user_features_col = array.array('i')
        self._user_features_value = array.array('f')

        self._item_features_row = array.array('i')
        self._item_features_col = array.array('i')
        self._item_features_value = array.array('f')

    def _unpack_datum(self, datum):

        if len(datum) == 3:
            (user_id, item_id, weight) = datum
        elif len(datum) == 2:
            (user_id, item_id) = datum
            weight = 1.0
        else:
            raise ValueError('Expecting tuples of (user_id, item_id, weight) '
                             'or (user_id, item_id). Got {}'.format(datum))

        user_idx = self._user_id_mapping.get(user_id)
        item_idx = self._item_id_mapping.get(item_id)

        if user_idx is None:
            raise ValueError('User id {} not in user id mapping. Make sure '
                             'you call the fit method.'.format(user_id))
        if item_idx is None:
            raise ValueError('Item id {} not in item id mapping. Make sure '
                             'you call the fit method.'.format(item_id))

        return (user_idx, item_idx, weight)

    def _check_fitted(self):

        if not self._user_id_mapping or not self._item_id_mapping:
            raise ValueError('You must call fit first to build the item and user '
                             'id mappings.')

    def fit_user_mapping(self, data):

        self._user_id_mapping = {}

        for user_id in data:
            self._user_id_mapping.setdefault(user_id, len(self._user_id_mapping))

    def fit_item_mapping(self, data):

        self._item_id_mapping = {}

        for item_id in data:
            self._item_id_mapping.setdefault(item_id, len(self._item_id_mapping))

    def _unpack_feature(self, datum):

        if len(datum) != 2:
            raise ValueError('Expected tuples of (user_id, features), '
                             'got {}.'.format(datum))

        return datum

    def _iter_features(self, features):

        if isinstance(features, dict):
            for entry in dict.items():
                yield entry
        else:
            for feature_name in features:
                yield (feature_name, 1.0)

    def fit_user_feature_mapping(self, data):

        self._user_feature_mapping = {}

        for datum in data:
            user_id, features = self._unpack_feature(datum)

            if user_id not in self._user_id_mapping:
                raise ValueError('User id {} not in user id mapping. '
                                 'Make sure you call fit_user_id_mapping '
                                 'first.'.format(user_id))

            for (feature_name, _) in self._iter_features(features):
                self._user_feature_mapping.setdefault(feature_name,
                                                      len(self._user_feature_mapping))

    def fit_item_feature_mapping(self, data):

        self._item_feature_mapping = {}

        for datum in data:
            item_id, features = self._unpack_feature(datum)

            if item_id not in self._item_id_mapping:
                raise ValueError('Item id {} not in item id mapping. '
                                 'Make sure you call fit_item_id_mapping '
                                 'first.'.format(item_id))

            for (feature_name, _) in self._iter_features(features):
                self._item_feature_mapping.setdefault(feature_name,
                                                      len(self._item_feature_mapping))

    def build_interactions_matrix(self, data):

        interactions = IncrementalCOOMatrix(self.interactions_shape(), np.int32)
        weights = IncrementalCOOMatrix(self.interactions_shape(), np.float32)

        for datum in data:
            user_idx, item_idx, weight = self._unpack_datum(datum)

            interactions.append(user_idx, item_idx, 1.0)
            weights.append(user_idx, item_idx, weight)

        return (interactions.tocoo(),
                weights.tocoo())

    def interactions_shape(self):

        self._check_fitted()

        return (len(self._user_id_mapping),
                len(self._item_id_mapping))

    def _process_features(self, datum):

        if len(datum) != 2:
            raise ValueError('Expected tuples of (user_id, features), '
                             'got {}.'.format(datum))

        user_id, features = datum

        if user_id not in self._user_id_mapping:
            raise ValueError('User id {} not in user id mappings.'
                             .format(user_id))

        user_idx = self._user_id_mapping[user_id]

    def build_user_features(self, data):

        self._initialise_features()

        for (user_id, features) in data:

import imp
import os
import pickle

import numpy as np

import scipy.sparse as sp

from sklearn.metrics import roc_auc_score

from tests.utils import precision_at_k, full_auc

from lightfm import LightFM

imp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..',
                        'examples',
                        'movielens',
                        'data.py')
movielens_data = imp.load_source('movielens_data',
                                 imp_path)


def _get_feature_matrices(interactions):

    no_users, no_items = interactions.shape

    user_features = sp.identity(no_users,
                                dtype=np.int32).tocsr()
    item_features = sp.identity(no_items,
                                dtype=np.int32).tocsr()

    return (user_features.tocsr(),
            item_features.tocsr())


train, test = movielens_data.get_movielens_data()

(train_user_features,
 train_item_features) = _get_feature_matrices(train)
(test_user_features,
 test_item_features) = _get_feature_matrices(test)


def test_movielens_accuracy():

    model = LightFM()
    model.fit_partial(train,
                      epochs=10)

    train_predictions = model.predict(train.row,
                                      train.col)
    test_predictions = model.predict(test.row,
                                     test.col)

    assert roc_auc_score(train.data, train_predictions) > 0.84
    assert roc_auc_score(test.data, test_predictions) > 0.76


def test_logistic_precision():

    model = LightFM()
    model.fit_partial(train,
                      epochs=10)

    train_precision = precision_at_k(model,
                                     train,
                                     10)
    test_precision = precision_at_k(model,
                                    test,
                                    10)

    full_train_auc = full_auc(model, train)
    full_test_auc = full_auc(model, test)

    assert train_precision > 0.3
    assert test_precision > 0.03

    assert full_train_auc > 0.79
    assert full_test_auc > 0.74


def test_bpr_precision():

    model = LightFM(learning_rate=0.05,
                    loss='bpr')

    model.fit_partial(train,
                      epochs=10)

    train_precision = precision_at_k(model,
                                     train,
                                     10)
    test_precision = precision_at_k(model,
                                    test,
                                    10)

    full_train_auc = full_auc(model, train)
    full_test_auc = full_auc(model, test)

    assert train_precision > 0.31
    assert test_precision > 0.04

    assert full_train_auc > 0.86
    assert full_test_auc > 0.84


def test_bpr_precision_multithreaded():

    model = LightFM(learning_rate=0.05,
                    loss='bpr')

    model.fit_partial(train,
                      epochs=10,
                      num_threads=2)

    train_precision = precision_at_k(model,
                                     train,
                                     10)
    test_precision = precision_at_k(model,
                                    test,
                                    10)

    full_train_auc = full_auc(model, train)
    full_test_auc = full_auc(model, test)

    assert train_precision > 0.31
    assert test_precision > 0.04

    assert full_train_auc > 0.86
    assert full_test_auc > 0.84


def test_warp_precision():

    model = LightFM(learning_rate=0.05,
                    loss='warp')

    model.fit_partial(train,
                      epochs=10)

    train_precision = precision_at_k(model,
                                     train,
                                     10)
    test_precision = precision_at_k(model,
                                    test,
                                    10)

    full_train_auc = full_auc(model, train)
    full_test_auc = full_auc(model, test)

    assert train_precision > 0.45
    assert test_precision > 0.07

    assert full_train_auc > 0.94
    assert full_test_auc > 0.9


def test_warp_precision_multithreaded():

    model = LightFM(learning_rate=0.05,
                    loss='warp')

    model.fit_partial(train,
                      epochs=10,
                      num_threads=2)

    train_precision = precision_at_k(model,
                                     train,
                                     10)
    test_precision = precision_at_k(model,
                                    test,
                                    10)

    full_train_auc = full_auc(model, train)
    full_test_auc = full_auc(model, test)

    assert train_precision > 0.45
    assert test_precision > 0.07

    assert full_train_auc > 0.94
    assert full_test_auc > 0.9


def test_warp_precision_adadelta():

    model = LightFM(learning_schedule='adadelta',
                    rho=0.95,
                    epsilon=0.000001,
                    loss='warp')

    model.fit_partial(train,
                      epochs=10,
                      num_threads=1)

    train_precision = precision_at_k(model,
                                     train,
                                     10)
    test_precision = precision_at_k(model,
                                    test,
                                    10)

    full_train_auc = full_auc(model, train)
    full_test_auc = full_auc(model, test)

    assert train_precision > 0.45
    assert test_precision > 0.07

    assert full_train_auc > 0.94
    assert full_test_auc > 0.9


def test_warp_precision_adadelta_multithreaded():

    model = LightFM(learning_schedule='adadelta',
                    rho=0.95,
                    epsilon=0.000001,
                    loss='warp')

    model.fit_partial(train,
                      epochs=10,
                      num_threads=2)

    train_precision = precision_at_k(model,
                                     train,
                                     10)
    test_precision = precision_at_k(model,
                                    test,
                                    10)

    full_train_auc = full_auc(model, train)
    full_test_auc = full_auc(model, test)

    assert train_precision > 0.45
    assert test_precision > 0.07

    assert full_train_auc > 0.94
    assert full_test_auc > 0.9


def test_warp_precision_max_sampled():

    model = LightFM(learning_rate=0.05,
                    max_sampled=1,
                    loss='warp')

    # This is equivalent to a no-op pass
    # over the training data
    model.max_sampled = 0

    model.fit_partial(train,
                      epochs=1)

    full_train_auc = full_auc(model, train)
    full_test_auc = full_auc(model, test)

    # The AUC should be no better than random
    assert full_train_auc < 0.55
    assert full_test_auc < 0.55


def test_warp_kos_precision():

    # Remove all negative examples
    training = train.copy()
    training.data[training.data < 1] = 0
    training = training.tocsr()
    training.eliminate_zeros()

    model = LightFM(learning_rate=0.05, k=5,
                    loss='warp-kos')

    model.fit_partial(training,
                      epochs=10)

    train_precision = precision_at_k(model,
                                     training,
                                     10)
    test_precision = precision_at_k(model,
                                    test,
                                    10)

    full_train_auc = full_auc(model, training)
    full_test_auc = full_auc(model, test)

    assert train_precision > 0.44
    assert test_precision > 0.06

    assert full_train_auc > 0.9
    assert full_test_auc > 0.87


def test_warp_stability():

    learning_rates = (0.05, 0.1, 0.5)

    for lrate in learning_rates:

        model = LightFM(learning_rate=lrate,
                        loss='warp')
        model.fit_partial(train,
                          epochs=10)

        assert not np.isnan(model.user_embeddings).any()
        assert not np.isnan(model.item_embeddings).any()


def test_movielens_genre_accuracy():

    item_features = movielens_data.get_movielens_item_metadata(use_item_ids=False)

    assert item_features.shape[1] < item_features.shape[0]

    model = LightFM()
    model.fit_partial(train,
                      item_features=item_features,
                      epochs=10)

    train_predictions = model.predict(train.row,
                                      train.col,
                                      item_features=item_features)
    test_predictions = model.predict(test.row,
                                     test.col,
                                     item_features=item_features)

    assert roc_auc_score(train.data, train_predictions) > 0.75
    assert roc_auc_score(test.data, test_predictions) > 0.69


def test_movielens_both_accuracy():
    """
    Accuracy with both genre metadata and item-specific
    features shoul be no worse than with just item-specific
    features (though more training may be necessary).
    """

    item_features = movielens_data.get_movielens_item_metadata(use_item_ids=True)

    model = LightFM()
    model.fit_partial(train,
                      item_features=item_features,
                      epochs=15)

    train_predictions = model.predict(train.row,
                                      train.col,
                                      item_features=item_features)
    test_predictions = model.predict(test.row,
                                     test.col,
                                     item_features=item_features)

    assert roc_auc_score(train.data, train_predictions) > 0.84
    assert roc_auc_score(test.data, test_predictions) > 0.75


def test_movielens_accuracy_fit():

    model = LightFM()
    model.fit(train,
              epochs=10)

    train_predictions = model.predict(train.row,
                                      train.col)
    test_predictions = model.predict(test.row,
                                     test.col)

    assert roc_auc_score(train.data, train_predictions) > 0.84
    assert roc_auc_score(test.data, test_predictions) > 0.76


def test_movielens_accuracy_pickle():

    model = LightFM()
    model.fit(train,
              epochs=10)

    model = pickle.loads(pickle.dumps(model))

    train_predictions = model.predict(train.row,
                                      train.col)
    test_predictions = model.predict(test.row,
                                     test.col)

    assert roc_auc_score(train.data, train_predictions) > 0.84
    assert roc_auc_score(test.data, test_predictions) > 0.76


def test_movielens_accuracy_resume():

    model = LightFM()

    for _ in range(10):
        model.fit_partial(train,
                          epochs=1)

    train_predictions = model.predict(train.row,
                                      train.col)
    test_predictions = model.predict(test.row,
                                     test.col)

    assert roc_auc_score(train.data, train_predictions) > 0.84
    assert roc_auc_score(test.data, test_predictions) > 0.76


def test_state_reset():

    model = LightFM()

    model.fit(train,
              epochs=1)

    assert np.mean(model.user_embedding_gradients) > 1.0

    model.fit(train,
              epochs=0)
    assert np.all(model.user_embedding_gradients == 1.0)


def test_user_supplied_features_accuracy():

    model = LightFM()
    model.fit_partial(train,
                      user_features=train_user_features,
                      item_features=train_item_features,
                      epochs=10)

    train_predictions = model.predict(train.row,
                                      train.col,
                                      user_features=train_user_features,
                                      item_features=train_item_features)
    test_predictions = model.predict(test.row,
                                     test.col,
                                     user_features=test_user_features,
                                     item_features=test_item_features)

    assert roc_auc_score(train.data, train_predictions) > 0.84
    assert roc_auc_score(test.data, test_predictions) > 0.76


def test_zeros_weights_accuracy():

    # When very small weights are used
    # accuracy should be no better than
    # random.
    train_mod = train.copy()
    train_mod.data[train.data == -1] = 0
    train_mod.data[train.data == 1] = 1e-04

    for loss in ('logistic', 'bpr', 'warp', 'warp-kos'):
        model = LightFM(loss=loss)
        model.fit_partial(train_mod,
                          epochs=10)

        train_predictions = model.predict(train.row,
                                          train.col)
        test_predictions = model.predict(test.row,
                                         test.col)

        assert 0.45 < roc_auc_score(train.data, train_predictions) < 0.55
        assert 0.45 < roc_auc_score(test.data, test_predictions) < 0.55


def test_hogwild_accuracy():

    # Should get comparable accuracy with 2 threads
    model = LightFM()
    model.fit_partial(train,
                      epochs=10,
                      num_threads=2)

    train_predictions = model.predict(train.row,
                                      train.col,
                                      num_threads=2)
    test_predictions = model.predict(test.row,
                                     test.col,
                                     num_threads=2)

    assert roc_auc_score(train.data, train_predictions) > 0.84
    assert roc_auc_score(test.data, test_predictions) > 0.76


def test_movielens_excessive_regularization():

    # Should perform poorly with high regularization
    model = LightFM(no_components=10,
                    item_alpha=1.0,
                    user_alpha=1.0)
    model.fit_partial(train,
                      epochs=10)

    train_predictions = model.predict(train.row,
                                      train.col)
    test_predictions = model.predict(test.row,
                                     test.col)

    assert roc_auc_score(train.data, train_predictions) < 0.6
    assert roc_auc_score(test.data, test_predictions) < 0.6


def test_overfitting():

    # Let's massivly overfit
    model = LightFM(no_components=50)
    model.fit_partial(train,
                      epochs=30)

    train_predictions = model.predict(train.row,
                                      train.col)
    test_predictions = model.predict(test.row,
                                     test.col)
    overfit_train = roc_auc_score(train.data, train_predictions)
    overfit_test = roc_auc_score(test.data, test_predictions)

    assert overfit_train > 0.99
    assert overfit_test < 0.75


def test_regularization():

    # Let's regularize
    model = LightFM(no_components=50,
                    item_alpha=0.0001,
                    user_alpha=0.0001)
    model.fit_partial(train,
                      epochs=30)

    train_predictions = model.predict(train.row,
                                      train.col)
    test_predictions = model.predict(test.row,
                                     test.col)

    assert roc_auc_score(train.data, train_predictions) > 0.80
    assert roc_auc_score(test.data, test_predictions) > 0.75


def test_training_schedules():

    model = LightFM(no_components=10,
                    learning_schedule='adagrad')
    model.fit_partial(train,
                      epochs=0)

    assert (model.item_embedding_gradients == 1).all()
    assert (model.item_embedding_momentum == 0).all()
    assert (model.item_bias_gradients == 1).all()
    assert (model.item_bias_momentum == 0).all()

    assert (model.user_embedding_gradients == 1).all()
    assert (model.user_embedding_momentum == 0).all()
    assert (model.user_bias_gradients == 1).all()
    assert (model.user_bias_momentum == 0).all()

    model.fit_partial(train,
                      epochs=1)

    assert (model.item_embedding_gradients > 1).any()
    assert (model.item_embedding_momentum == 0).all()
    assert (model.item_bias_gradients > 1).any()
    assert (model.item_bias_momentum == 0).all()

    assert (model.user_embedding_gradients > 1).any()
    assert (model.user_embedding_momentum == 0).all()
    assert (model.user_bias_gradients > 1).any()
    assert (model.user_bias_momentum == 0).all()

    model = LightFM(no_components=10,
                    learning_schedule='adadelta')
    model.fit_partial(train,
                      epochs=0)

    assert (model.item_embedding_gradients == 0).all()
    assert (model.item_embedding_momentum == 0).all()
    assert (model.item_bias_gradients == 0).all()
    assert (model.item_bias_momentum == 0).all()

    assert (model.user_embedding_gradients == 0).all()
    assert (model.user_embedding_momentum == 0).all()
    assert (model.user_bias_gradients == 0).all()
    assert (model.user_bias_momentum == 0).all()

    model.fit_partial(train,
                      epochs=1)

    assert (model.item_embedding_gradients > 0).any()
    assert (model.item_embedding_momentum > 0).any()
    assert (model.item_bias_gradients > 0).any()
    assert (model.item_bias_momentum > 0).any()

    assert (model.user_embedding_gradients > 0).any()
    assert (model.user_embedding_momentum > 0).any()
    assert (model.user_bias_gradients > 0).any()
    assert (model.user_bias_momentum > 0).any()

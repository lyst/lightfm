import pickle

import numpy as np

import scipy.sparse as sp
from scipy import stats

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, RandomizedSearchCV

from lightfm.lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import auc_score, precision_at_k


SEED = 10


def _get_metrics(model, train_set, test_set):

    train_set = train_set.tocsr()
    test_set = test_set.tocsr()

    train_set.data[train_set.data < 0] = 0.0
    test_set.data[test_set.data < 0] = 0.0

    train_set.eliminate_zeros()
    test_set.eliminate_zeros()

    return (
        precision_at_k(model, train_set).mean(),
        precision_at_k(model, test_set).mean(),
        auc_score(model, train_set).mean(),
        auc_score(model, test_set).mean(),
    )


def _get_feature_matrices(interactions):

    no_users, no_items = interactions.shape

    user_features = sp.identity(no_users, dtype=np.int32).tocsr()
    item_features = sp.identity(no_items, dtype=np.int32).tocsr()

    return (user_features.tocsr(), item_features.tocsr())


def _binarize(dataset):

    positives = dataset.data >= 4.0
    dataset.data[positives] = 1.0
    dataset.data[np.logical_not(positives)] = -1.0

    return dataset


movielens = fetch_movielens()
train, test = _binarize(movielens["train"]), _binarize(movielens["test"])


(train_user_features, train_item_features) = _get_feature_matrices(train)
(test_user_features, test_item_features) = _get_feature_matrices(test)


def test_movielens_accuracy():

    model = LightFM(random_state=SEED)
    model.fit_partial(train, epochs=10)

    train_predictions = model.predict(train.row, train.col)
    test_predictions = model.predict(test.row, test.col)

    assert roc_auc_score(train.data, train_predictions) > 0.84
    assert roc_auc_score(test.data, test_predictions) > 0.76


def test_logistic_precision():

    model = LightFM(random_state=SEED)
    model.fit_partial(train, epochs=10)

    (train_precision, test_precision, full_train_auc, full_test_auc) = _get_metrics(
        model, train, test
    )

    assert train_precision > 0.3
    assert test_precision > 0.03

    assert full_train_auc > 0.79
    assert full_test_auc > 0.73


def test_bpr_precision():

    model = LightFM(learning_rate=0.05, loss="bpr", random_state=SEED)

    model.fit_partial(train, epochs=10)

    (train_precision, test_precision, full_train_auc, full_test_auc) = _get_metrics(
        model, train, test
    )

    assert train_precision > 0.45
    assert test_precision > 0.07

    assert full_train_auc > 0.91
    assert full_test_auc > 0.87


def test_bpr_precision_multithreaded():

    model = LightFM(learning_rate=0.05, loss="bpr", random_state=SEED)

    model.fit_partial(train, epochs=10, num_threads=4)

    (train_precision, test_precision, full_train_auc, full_test_auc) = _get_metrics(
        model, train, test
    )

    assert train_precision > 0.45
    assert test_precision > 0.07

    assert full_train_auc > 0.91
    assert full_test_auc > 0.87


def test_warp_precision():

    model = LightFM(learning_rate=0.05, loss="warp", random_state=SEED)

    model.fit_partial(train, epochs=10)

    (train_precision, test_precision, full_train_auc, full_test_auc) = _get_metrics(
        model, train, test
    )

    assert train_precision > 0.45
    assert test_precision > 0.07

    assert full_train_auc > 0.94
    assert full_test_auc > 0.9


def test_warp_precision_high_interaction_values():

    model = LightFM(learning_rate=0.05, loss="warp", random_state=SEED)

    _train = train.copy()
    _train.data = _train.data * 5

    model.fit_partial(_train, epochs=10)

    (train_precision, test_precision, full_train_auc, full_test_auc) = _get_metrics(
        model, _train, test
    )

    assert train_precision > 0.45
    assert test_precision > 0.07

    assert full_train_auc > 0.93
    assert full_test_auc > 0.9


def test_bpr_precision_high_interaction_values():

    model = LightFM(learning_rate=0.05, loss="bpr", random_state=SEED)

    _train = train.copy()
    _train.data = _train.data * 5

    model.fit_partial(_train, epochs=10)

    (train_precision, test_precision, full_train_auc, full_test_auc) = _get_metrics(
        model, _train, test
    )

    assert train_precision > 0.31
    assert test_precision > 0.04

    assert full_train_auc > 0.86
    assert full_test_auc > 0.84


def test_warp_precision_multithreaded():

    model = LightFM(learning_rate=0.05, loss="warp", random_state=SEED)

    model.fit_partial(train, epochs=10, num_threads=4)

    (train_precision, test_precision, full_train_auc, full_test_auc) = _get_metrics(
        model, train, test
    )

    assert train_precision > 0.45
    assert test_precision > 0.07

    assert full_train_auc > 0.9
    assert full_test_auc > 0.9


def test_warp_precision_adadelta():

    model = LightFM(
        learning_schedule="adadelta",
        rho=0.95,
        epsilon=0.000001,
        loss="warp",
        random_state=SEED,
    )

    model.fit_partial(train, epochs=10, num_threads=1)

    (train_precision, test_precision, full_train_auc, full_test_auc) = _get_metrics(
        model, train, test
    )

    assert train_precision > 0.45
    assert test_precision > 0.07

    assert full_train_auc > 0.94
    assert full_test_auc > 0.9


def test_warp_precision_adadelta_multithreaded():

    model = LightFM(
        learning_schedule="adadelta",
        rho=0.95,
        epsilon=0.000001,
        loss="warp",
        random_state=SEED,
    )

    model.fit_partial(train, epochs=10, num_threads=4)

    (train_precision, test_precision, full_train_auc, full_test_auc) = _get_metrics(
        model, train, test
    )

    assert train_precision > 0.45
    assert test_precision > 0.07

    assert full_train_auc > 0.9
    assert full_test_auc > 0.9


def test_warp_precision_max_sampled():

    model = LightFM(learning_rate=0.05, max_sampled=1, loss="warp", random_state=SEED)

    # This is equivalent to a no-op pass
    # over the training data
    model.max_sampled = 0

    model.fit_partial(train, epochs=1)

    (train_precision, test_precision, full_train_auc, full_test_auc) = _get_metrics(
        model, train, test
    )

    # The AUC should be no better than random
    assert full_train_auc < 0.55
    assert full_test_auc < 0.55


def test_warp_kos_precision():

    # Remove all negative examples
    training = train.copy()
    training.data[training.data < 1] = 0
    training = training.tocsr()
    training.eliminate_zeros()

    model = LightFM(learning_rate=0.05, k=5, loss="warp-kos", random_state=SEED)

    model.fit_partial(training, epochs=10)

    (train_precision, test_precision, full_train_auc, full_test_auc) = _get_metrics(
        model, train, test
    )

    assert train_precision > 0.44
    assert test_precision > 0.06

    assert full_train_auc > 0.9
    assert full_test_auc > 0.87


def test_warp_stability():

    learning_rates = (0.05, 0.1, 0.5)

    for lrate in learning_rates:

        model = LightFM(learning_rate=lrate, loss="warp", random_state=SEED)
        model.fit_partial(train, epochs=10)

        assert not np.isnan(model.user_embeddings).any()
        assert not np.isnan(model.item_embeddings).any()


def test_movielens_genre_accuracy():

    item_features = fetch_movielens(indicator_features=False, genre_features=True)[
        "item_features"
    ]

    assert item_features.shape[1] < item_features.shape[0]

    model = LightFM(random_state=SEED)
    model.fit_partial(train, item_features=item_features, epochs=10)

    train_predictions = model.predict(train.row, train.col, item_features=item_features)
    test_predictions = model.predict(test.row, test.col, item_features=item_features)

    assert roc_auc_score(train.data, train_predictions) > 0.75
    assert roc_auc_score(test.data, test_predictions) > 0.69


def test_get_representations():

    model = LightFM(random_state=SEED)
    model.fit_partial(train, epochs=10)

    num_users, num_items = train.shape

    for (item_features, user_features) in (
        (None, None),
        (
            (sp.identity(num_items) + sp.random(num_items, num_items)),
            (sp.identity(num_users) + sp.random(num_users, num_users)),
        ),
    ):

        test_predictions = model.predict(
            test.row, test.col, user_features=user_features, item_features=item_features
        )

        item_biases, item_latent = model.get_item_representations(item_features)
        user_biases, user_latent = model.get_user_representations(user_features)

        assert item_latent.dtype == np.float32
        assert user_latent.dtype == np.float32

        predictions = (
            (user_latent[test.row] * item_latent[test.col]).sum(axis=1)
            + user_biases[test.row]
            + item_biases[test.col]
        )

        assert np.allclose(test_predictions, predictions, atol=0.000001)


def test_movielens_both_accuracy():
    """
    Accuracy with both genre metadata and item-specific
    features should be no worse than with just item-specific
    features (though more training may be necessary).
    """

    item_features = fetch_movielens(indicator_features=True, genre_features=True)[
        "item_features"
    ]

    model = LightFM(random_state=SEED)
    model.fit_partial(train, item_features=item_features, epochs=15)

    train_predictions = model.predict(train.row, train.col, item_features=item_features)
    test_predictions = model.predict(test.row, test.col, item_features=item_features)

    assert roc_auc_score(train.data, train_predictions) > 0.84
    assert roc_auc_score(test.data, test_predictions) > 0.75


def test_movielens_accuracy_fit():

    model = LightFM(random_state=SEED)
    model.fit(train, epochs=10)

    train_predictions = model.predict(train.row, train.col)
    test_predictions = model.predict(test.row, test.col)

    assert roc_auc_score(train.data, train_predictions) > 0.84
    assert roc_auc_score(test.data, test_predictions) > 0.76


def test_movielens_accuracy_pickle():

    model = LightFM(random_state=SEED)
    model.fit(train, epochs=10)

    model = pickle.loads(pickle.dumps(model))

    train_predictions = model.predict(train.row, train.col)
    test_predictions = model.predict(test.row, test.col)

    assert roc_auc_score(train.data, train_predictions) > 0.84
    assert roc_auc_score(test.data, test_predictions) > 0.76


def test_movielens_accuracy_resume():

    model = LightFM(random_state=SEED)

    for _ in range(10):
        model.fit_partial(train, epochs=1)

    train_predictions = model.predict(train.row, train.col)
    test_predictions = model.predict(test.row, test.col)

    assert roc_auc_score(train.data, train_predictions) > 0.84
    assert roc_auc_score(test.data, test_predictions) > 0.76


def test_movielens_accuracy_sample_weights():
    # Scaling weights down and learning rate up
    # by the same amount should result in
    # roughly the same accuracy

    scale = 0.5
    weights = train.copy()
    weights.data = np.ones(train.getnnz(), dtype=np.float32) * scale

    for (loss, exp_score) in (("logistic", 0.74), ("bpr", 0.84), ("warp", 0.89)):
        model = LightFM(loss=loss, random_state=SEED)
        model.learning_rate * 1.0 / scale

        model.fit_partial(train, sample_weight=weights, epochs=10)

        (train_precision, test_precision, full_train_auc, full_test_auc) = _get_metrics(
            model, train, test
        )

        assert full_train_auc > exp_score


def test_movielens_accuracy_sample_weights_grad_accumulation():

    # Set weights to zero for all even-numbered users
    # and check that they have not accumulated any
    # gradient updates.

    weights = train.copy()
    weights.data = np.ones(train.getnnz(), dtype=np.float32)
    even_users = weights.row % 2 == 0
    weights.data *= even_users

    even_idx = np.arange(train.shape[0]) % 2 == 0
    odd_idx = np.arange(train.shape[0]) % 2 != 0

    for loss in ("logistic", "bpr", "warp"):
        model = LightFM(loss=loss, random_state=SEED)

        model.fit_partial(train, sample_weight=weights, epochs=1)

        assert np.allclose(model.user_embedding_gradients[odd_idx], 1.0)
        assert np.allclose(model.user_bias_gradients[odd_idx], 1.0)

        assert not np.allclose(model.user_embedding_gradients[even_idx], 1.0)
        assert not np.allclose(model.user_bias_gradients[even_idx], 1.0)


def test_state_reset():

    model = LightFM(random_state=SEED)

    model.fit(train, epochs=1)

    assert np.mean(model.user_embedding_gradients) > 1.0

    model.fit(train, epochs=0)
    assert np.all(model.user_embedding_gradients == 1.0)


def test_user_supplied_features_accuracy():

    model = LightFM(random_state=SEED)
    model.fit_partial(
        train,
        user_features=train_user_features,
        item_features=train_item_features,
        epochs=10,
    )

    train_predictions = model.predict(
        train.row,
        train.col,
        user_features=train_user_features,
        item_features=train_item_features,
    )
    test_predictions = model.predict(
        test.row,
        test.col,
        user_features=test_user_features,
        item_features=test_item_features,
    )

    assert roc_auc_score(train.data, train_predictions) > 0.84
    assert roc_auc_score(test.data, test_predictions) > 0.76


def test_zeros_negative_accuracy():

    # Should get the same accuracy when zeros are used to
    # denote negative interactions
    train.data[train.data == -1] = 0
    model = LightFM(random_state=SEED)
    model.fit_partial(train, epochs=10)

    train_predictions = model.predict(train.row, train.col)
    test_predictions = model.predict(test.row, test.col)

    assert roc_auc_score(train.data, train_predictions) > 0.84
    assert roc_auc_score(test.data, test_predictions) > 0.76


def test_zero_weights_accuracy():

    # When very small weights are used
    # accuracy should be no better than
    # random.
    weights = train.copy()
    weights.data = np.zeros(train.getnnz(), dtype=np.float32)

    for loss in ("logistic", "bpr", "warp"):
        model = LightFM(loss=loss, random_state=SEED)
        model.fit_partial(train, sample_weight=weights, epochs=10)

        train_predictions = model.predict(train.row, train.col)
        test_predictions = model.predict(test.row, test.col)

        assert 0.45 < roc_auc_score(train.data, train_predictions) < 0.55
        assert 0.45 < roc_auc_score(test.data, test_predictions) < 0.55


def test_hogwild_accuracy():

    # Should get comparable accuracy with 2 threads
    model = LightFM(random_state=SEED)
    model.fit_partial(train, epochs=10, num_threads=2)

    train_predictions = model.predict(train.row, train.col, num_threads=2)
    test_predictions = model.predict(test.row, test.col, num_threads=2)

    assert roc_auc_score(train.data, train_predictions) > 0.84
    assert roc_auc_score(test.data, test_predictions) > 0.76


def test_movielens_excessive_regularization():

    for loss in ("logistic", "warp", "bpr", "warp-kos"):

        # Should perform poorly with high regularization.
        # Check that regularization does not accumulate
        # until it reaches infinity.
        model = LightFM(
            no_components=10,
            item_alpha=1.0,
            user_alpha=1.0,
            loss=loss,
            random_state=SEED,
        )
        model.fit_partial(train, epochs=10, num_threads=4)

        train_predictions = model.predict(train.row, train.col)
        test_predictions = model.predict(test.row, test.col)

        assert roc_auc_score(train.data, train_predictions) < 0.65
        assert roc_auc_score(test.data, test_predictions) < 0.65


def test_overfitting():

    # Let's massivly overfit
    model = LightFM(no_components=50, random_state=SEED)
    model.fit_partial(train, epochs=30)

    train_predictions = model.predict(train.row, train.col)
    test_predictions = model.predict(test.row, test.col)
    overfit_train = roc_auc_score(train.data, train_predictions)
    overfit_test = roc_auc_score(test.data, test_predictions)

    assert overfit_train > 0.99
    assert overfit_test < 0.75


def test_regularization():

    # Let's regularize
    model = LightFM(
        no_components=50, item_alpha=0.0001, user_alpha=0.0001, random_state=SEED
    )
    model.fit_partial(train, epochs=30)

    train_predictions = model.predict(train.row, train.col)
    test_predictions = model.predict(test.row, test.col)

    assert roc_auc_score(train.data, train_predictions) > 0.80
    assert roc_auc_score(test.data, test_predictions) > 0.75


def test_training_schedules():

    model = LightFM(no_components=10, learning_schedule="adagrad", random_state=SEED)
    model.fit_partial(train, epochs=0)

    assert (model.item_embedding_gradients == 1).all()
    assert (model.item_embedding_momentum == 0).all()
    assert (model.item_bias_gradients == 1).all()
    assert (model.item_bias_momentum == 0).all()

    assert (model.user_embedding_gradients == 1).all()
    assert (model.user_embedding_momentum == 0).all()
    assert (model.user_bias_gradients == 1).all()
    assert (model.user_bias_momentum == 0).all()

    model.fit_partial(train, epochs=1)

    assert (model.item_embedding_gradients > 1).any()
    assert (model.item_embedding_momentum == 0).all()
    assert (model.item_bias_gradients > 1).any()
    assert (model.item_bias_momentum == 0).all()

    assert (model.user_embedding_gradients > 1).any()
    assert (model.user_embedding_momentum == 0).all()
    assert (model.user_bias_gradients > 1).any()
    assert (model.user_bias_momentum == 0).all()

    model = LightFM(no_components=10, learning_schedule="adadelta", random_state=SEED)
    model.fit_partial(train, epochs=0)

    assert (model.item_embedding_gradients == 0).all()
    assert (model.item_embedding_momentum == 0).all()
    assert (model.item_bias_gradients == 0).all()
    assert (model.item_bias_momentum == 0).all()

    assert (model.user_embedding_gradients == 0).all()
    assert (model.user_embedding_momentum == 0).all()
    assert (model.user_bias_gradients == 0).all()
    assert (model.user_bias_momentum == 0).all()

    model.fit_partial(train, epochs=1)

    assert (model.item_embedding_gradients > 0).any()
    assert (model.item_embedding_momentum > 0).any()
    assert (model.item_bias_gradients > 0).any()
    assert (model.item_bias_momentum > 0).any()

    assert (model.user_embedding_gradients > 0).any()
    assert (model.user_embedding_momentum > 0).any()
    assert (model.user_bias_gradients > 0).any()
    assert (model.user_bias_momentum > 0).any()


def test_random_state_fixing():

    model = LightFM(learning_rate=0.05, loss="warp", random_state=SEED)

    model.fit_partial(train, epochs=2)

    model_2 = LightFM(learning_rate=0.05, loss="warp", random_state=SEED)

    model_2.fit_partial(train, epochs=2)

    assert np.all(model.user_embeddings == model_2.user_embeddings)
    assert np.all(model.item_embeddings == model_2.item_embeddings)


def test_random_state_advanced():
    # Check that using the random state
    # to seed rand_r in Cython advances
    # the random generator state.

    model = LightFM(learning_rate=0.05, loss="warp", random_state=SEED)

    model.fit_partial(train, epochs=1)

    rng_state = model.random_state.get_state()[1].copy()

    model.fit_partial(train, epochs=1)

    assert not np.all(rng_state == model.random_state.get_state()[1])


def test_sklearn_cv():

    model = LightFM(loss="warp", random_state=42)

    # Set distributions for hyperparameters
    randint = stats.randint(low=1, high=65)
    randint.random_state = 42
    gamma = stats.gamma(a=1.2, loc=0, scale=0.13)
    gamma.random_state = 42
    distr = {"no_components": randint, "learning_rate": gamma}

    # Custom score function
    def scorer(est, x, y=None):
        return precision_at_k(est, x).mean()

    # Dummy custom CV to ensure shape preservation.
    class CV(KFold):
        def split(self, X, y=None, groups=None):
            idx = np.arange(X.shape[0])
            for _ in range(self.n_splits):
                yield idx, idx

    cv = CV(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=distr,
        n_iter=2,
        scoring=scorer,
        random_state=42,
        cv=cv,
    )
    search.fit(train)
    assert search.best_params_["no_components"] == 58

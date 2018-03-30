from lightfm.data import Dataset


def test_fitting():

    users, items = 10, 100

    dataset = Dataset()
    dataset.fit(range(users), range(items))

    assert dataset.interactions_shape() == (users, items)
    assert dataset.user_features_shape() == (users, users)
    assert dataset.item_features_shape() == (items, items)

    assert dataset.build_interactions_matrix([])[0].shape == (users, items)
    assert dataset.build_user_features([]).getnnz() == users
    assert dataset.build_item_features([]).getnnz() == items


def test_fitting_no_identity():

    users, items = 10, 100

    dataset = Dataset(user_identity_features=False,
                      item_identity_features=False)
    dataset.fit(range(users), range(items))

    assert dataset.interactions_shape() == (users, items)
    assert dataset.user_features_shape() == (users, 0)
    assert dataset.item_features_shape() == (items, 0)

    assert dataset.build_interactions_matrix([])[0].shape == (users, items)
    assert dataset.build_user_features([]).getnnz() == 0
    assert dataset.build_item_features([]).getnnz() == 0

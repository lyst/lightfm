import itertools
import os
import zipfile

import numpy as np

import requests

import scipy.sparse as sp


def _get_movielens_path():
    """
    Get path to the movielens dataset file.
    """

    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'movielens.zip')


def _download_movielens(dest_path):
    """
    Download the dataset.
    """

    url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
    req = requests.get(url, stream=True)

    with open(dest_path, 'wb') as fd:
        for chunk in req.iter_content():
            fd.write(chunk)


def _get_raw_movielens_data():
    """
    Return the raw lines of the train and test files.
    """

    path = _get_movielens_path()

    if not os.path.isfile(path):
        _download_movielens(path)

    with zipfile.ZipFile(path) as datafile:
        return (datafile.read('ml-100k/ua.base').decode().split('\n'),
                datafile.read('ml-100k/ua.test').decode().split('\n'))


def _parse(data):
    """
    Parse movielens dataset lines.
    """

    for line in data:

        if not line:
            continue

        uid, iid, rating, timestamp = [int(x) for x in line.split('\t')]

        yield uid, iid, rating, timestamp


def _build_interaction_matrix(rows, cols, data):
    """
    Build the training matrix (no_users, no_items),
    with ratings >= 4.0 being marked as positive and
    the rest as negative.
    """

    mat = sp.lil_matrix((rows, cols), dtype=np.int32)

    for uid, iid, rating, timestamp in data:
        if rating >= 4.0:
            mat[uid, iid] = 1.0
        else:
            mat[uid, iid] = -1.0

    return mat.tocoo()


def _get_movie_raw_metadata():
    """
    Get raw lines of the genre file.
    """

    path = _get_movielens_path()

    if not os.path.isfile(path):
        _download_movielens(path)

    with zipfile.ZipFile(path) as datafile:
        return datafile.read('ml-100k/u.item').decode(errors='ignore').split('\n')


def get_movielens_item_metadata(use_item_ids):
    """
    Build a matrix of genre features (no_items, no_features).

    If use_item_ids is True, per-item features will also be used.
    """

    features = {}
    genre_set = set()

    for line in _get_movie_raw_metadata():

        if not line:
            continue

        splt = line.split('|')
        item_id = int(splt[0])

        genres = [idx for idx, val in
                  zip(range(len(splt[5:])), splt[5:])
                  if int(val) > 0]

        if use_item_ids:
            # Add item-specific features too
            genres.append(item_id)

        for genre_id in genres:
            genre_set.add(genre_id)

        features[item_id] = genres

    mat = sp.lil_matrix((len(features) + 1,
                         len(genre_set)),
                        dtype=np.int32)

    for item_id, genre_ids in features.items():
        for genre_id in genre_ids:
            mat[item_id, genre_id] = 1

    return mat


def get_movielens_data():
    """
    Return (train_interactions, test_interactions).
    """

    train_data, test_data = _get_raw_movielens_data()

    uids = set()
    iids = set()

    for uid, iid, rating, timestamp in itertools.chain(_parse(train_data),
                                                       _parse(test_data)):
        uids.add(uid)
        iids.add(iid)

    rows = max(uids) + 1
    cols = max(iids) + 1

    return (_build_interaction_matrix(rows, cols, _parse(train_data)),
            _build_interaction_matrix(rows, cols, _parse(test_data)))

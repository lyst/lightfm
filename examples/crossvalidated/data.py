import codecs
try:
    from html.parser import HTMLParser
except ImportError:
    # Python 2 backward compat
    from HTMLParser import HTMLParser
import os
import re
import subprocess

from lxml import etree

import numpy as np

import requests

import scipy.sparse as sp

from sklearn.feature_extraction import DictVectorizer


class MLStripper(HTMLParser):

    def __init__(self):
        self.reset()
        self.fed = []

    def handle_data(self, d):

        self.fed.append(d)

    def get_data(self):

        return ''.join(self.fed)


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def _process_body(body):

    body = re.sub('[^a-zA-Z]+', ' ', body.lower())

    return [x for x in body.split(' ') if len(x) > 2]


def _get_data_path(fname):

    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        fname)


def _get_download_path():

        return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'crossvalidated.7z')


def _download():
    """
    Download the dataset.
    """

    url = 'http://archive.org/download/stackexchange/stats.stackexchange.com.7z'
    req = requests.get(url, stream=True)

    download_path = _get_download_path()
    data_path = _get_data_path('')

    if not os.path.isfile(download_path):
        print('Downloading data...')
        with open(download_path, 'wb') as fd:
            for chunk in req.iter_content():
                fd.write(chunk)

    with open(os.devnull, 'w') as fnull:
        print('Extracting data...')
        try:
            subprocess.check_call(['7za', 'x', download_path],
                                  cwd=data_path, stdout=fnull)
        except (OSError, subprocess.CalledProcessError):
            raise Exception('You must install p7zip to extract the data.')


def _get_raw_data(fname):
    """
    Return the raw lines of the train and test files.
    """

    path = _get_data_path(fname)

    if not os.path.isfile(path):
        _download()

    return codecs.open(path, 'r', encoding='utf-8')


def _process_post_tags(tags_string):

    return [x for x in tags_string.replace('<', ' ').replace('>', ' ').split(' ') if x]


def _process_about(about):

    clean_about = (strip_tags(about)
                   .replace('\n', ' ')
                   .lower())
    tokens = _process_body(clean_about)

    return tokens


def _read_raw_post_data():

    with _get_raw_data('Posts.xml') as datafile:

        for i, line in enumerate(datafile):
            try:
                datum = dict(etree.fromstring(line).items())

                post_id = datum['Id']
                parent_post_id = datum.get('ParentId', None)
                user_id = datum.get('OwnerUserId', None)

                tags = _process_post_tags(datum.get('Tags', ''))

                if None in (post_id, user_id):
                    continue

            except etree.XMLSyntaxError:
                continue

            yield int(post_id), int(user_id), int(parent_post_id) if parent_post_id else None, tags


def _read_raw_user_data():

    with _get_raw_data('Users.xml') as datafile:
        for i, line in enumerate(datafile):

            try:
                datum = dict(etree.fromstring(line).items())

                user_id = datum['Id']
                about_me = datum.get('AboutMe', '')

                yield int(user_id), _process_about(about_me)

            except etree.XMLSyntaxError:
                pass


def read_data():
    """
    Construct a user-thread matrix, where a user interacts
    with a thread if they post an answer in it.
    """

    user_mapping = {}
    post_mapping = {}

    question_tags = {}
    uids = []
    pids = []
    data = []

    for (post_id, user_id,
         parent_post_id, tags) in _read_raw_post_data():

        if None in (post_id, user_id):
            continue

        if user_id == -1:
            continue

        if parent_post_id is None:
            # This is a question

            pid = post_mapping.setdefault(post_id,
                                          len(post_mapping))
            tag_dict = question_tags.setdefault(pid, {})

            for tag in tags:
                tag_dict[tag] = 1

            tag_dict['intercept'] = 1

        else:
            # This is an answer
            uid = user_mapping.setdefault(user_id,
                                          len(user_mapping))
            pid = post_mapping.setdefault(parent_post_id,
                                          len(post_mapping))

            uids.append(uid)
            pids.append(pid)
            data.append(1)

    user_about = {}
    for (user_id, about) in _read_raw_user_data():

        if user_id == -1:
            continue

        if user_id not in user_mapping:
            continue

        uid = user_mapping[user_id]
        user_about[uid] = {x: 1 for x in about + ['user_id:' + str(uid)]}

    interaction_matrix = sp.coo_matrix((data, (uids, pids)),
                                       shape=(len(user_mapping),
                                              len(post_mapping)),
                                       dtype=np.int32)

    # Select only those questions that have any answers
    answered = np.squeeze(np.array(interaction_matrix.sum(axis=0))) > 0
    interaction_matrix = interaction_matrix.tocsr()[:, answered]
    interaction_matrix.data = np.ones_like(interaction_matrix.data)

    active_users = np.squeeze(np.array(interaction_matrix.sum(axis=1))) > 1
    interaction_matrix = interaction_matrix[active_users]

    tag_list = [question_tags.get(x, {}) for x in range(len(post_mapping))]

    vectorizer = DictVectorizer(dtype=np.int32)
    tag_matrix = vectorizer.fit_transform(tag_list)
    tag_matrix = tag_matrix.tocsr()[answered]
    assert tag_matrix.shape[0] == interaction_matrix.shape[1]

    about_list = [user_about.get(x, {}) for x in range(len(user_mapping))]
    about_vectorizer = DictVectorizer(dtype=np.int32)
    about_matrix = about_vectorizer.fit_transform(about_list)
    about_matrix = about_matrix.tocsr()[active_users]
    assert about_matrix.shape[0] == interaction_matrix.shape[0]

    assert (np.squeeze(np.array(about_matrix.sum(axis=1))) > 0).all()

    return interaction_matrix, tag_matrix, about_matrix, vectorizer, about_vectorizer

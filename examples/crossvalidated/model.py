import random

import lightfm

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def train_test_split(interactions):

    train = interactions.copy()
    test = interactions.copy()

    for i in range(len(train.data)):
        if random.random() < 0.2:
            train.data[i] = 0
        else:
            test.data[i] = 0

    train.eliminate_zeros()
    test.eliminate_zeros()

    return train, test


def fit_content_models(interactions, post_features):

    models = []

    for user_row in interactions:
        y = np.squeeze(np.array(user_row.todense()))

        model = LogisticRegression(C=0.4)
        try:
            model.fit(post_features, y)
        except ValueError:
            # Just one class
            pass

        models.append(model)

    return models


def auc_content_models(models, interactions, post_features):

    auc_scores= []

    for i in range(len(models)):

        model = models[i]
        y = np.squeeze(np.array(interactions[i].todense()))

        try:
            score = roc_auc_score(y, model.decision_function(post_features))
            auc_scores.append(score)
        except ValueError:
            # Just one class
            pass

    return sum(auc_scores) / len(auc_scores)


def fit_lightfm_model(interactions, post_features=None, user_features=None, epochs=30):

    model = lightfm.LightFM(loss='warp',
                            learning_rate=0.01,
                            learning_schedule='adagrad',
                            user_alpha=0.0001,
                            item_alpha=0.0001,
                            no_components=30)

    model.fit(interactions,
              item_features=post_features,
              user_features=user_features,
              num_threads=4,
              epochs=epochs)

    return model


def auc_lightfm(model, interactions, post_features=None, user_features=None):

    no_users, no_items = interactions.shape

    pid_array = np.arange(no_items, dtype=np.int32)

    scores = []

    for i in range(interactions.shape[0]):
        uid_array = np.empty(no_items, dtype=np.int32)
        uid_array.fill(i)
        predictions = model.predict(uid_array,
                                    pid_array,
                                    item_features=post_features,
                                    user_features=user_features,
                                    num_threads=4)
        y = np.squeeze(np.array(interactions[i].todense()))

        try:
            scores.append(roc_auc_score(y, predictions))
        except ValueError:
            # Just one class
            pass

    return sum(scores) / len(scores)


def similar_tags(model, vectorizer, tag, number=10):

    tag_idx = vectorizer.vocabulary_[tag]

    tag_embedding = model.item_embeddings[tag_idx]

    sim = (np.dot(model.item_embeddings, tag_embedding)
           / np.linalg.norm(model.item_embeddings, axis=1))

    return np.array(vectorizer.get_feature_names())[np.argsort(-sim)[1:1 + number]].tolist()

from __future__ import print_function

from .lightfm import LightFM

from sklearn.base import BaseEstimator


def make_sklearn_LightFM():
    doc = """
          LightFM recommender compatible with Scikit-Learn's BaseEstimator API.
          """
    doc += LightFM.__doc__[LightFM.__doc__.find('.')+2:]
    return type('SKLearnLightFM', (LightFM, BaseEstimator), {'__doc__': doc})


SKLearnLightFM = make_sklearn_LightFM()

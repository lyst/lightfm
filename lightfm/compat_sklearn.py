from __future__ import print_function

from .lightfm import LightFM

from sklearn.base import BaseEstimator

class SKLearnLightFM(LightFM, BaseEstimator):
    """
    LightFM recommender compatible with Scikit-Learn's BaseEstimator API.
    """

# Append the docstring of the original LightFM recommender
SKLearnLightFM.__doc__ += LightFM.__doc__[LightFM.__doc__.find('.')+2:]

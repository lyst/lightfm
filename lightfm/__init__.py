try:
    from .compat_sklearn import SKLearnLightFM as LightFM
except ImportError:
    from .lightfm import LightFM

__version__ = '1.9'

__all__ = ['LightFM', 'datasets', 'evaluation']

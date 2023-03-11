try:
    __LIGHTFM_SETUP__
except NameError:
    from .lightfm import LightFM

__version__ = "1.17"

__all__ = ["LightFM", "datasets", "evaluation"]

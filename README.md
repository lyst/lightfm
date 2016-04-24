# LightFM

![LightFM logo](lightfm.png)

[![Circle CI](https://circleci.com/gh/lyst/lightfm.svg?style=svg)](https://circleci.com/gh/lyst/lightfm)
[![Travis CI](https://travis-ci.org/lyst/lightfm.svg?branch=master)](https://travis-ci.org/lyst/lightfm)

LightFM is a Python implementation of a number of popular recommendation algorithms for both implicit and explicit feedback. It's easy to use, fast (via multithreaded model estimation), and produces high quality results.

It also makes it possible to incorporate both item and user metadata into the traditional matrix factorization algorithms It represents each user and item as the sum of the latent representations of their features, thus allowing recommendations to generalise to new items (via item features) and to new users (via user features).

For more details, see the [Documentation](http://lyst.github.io/lightfm/docs/quickstart.html).

## Development
Pull requests are welcome. To install for development:

1. Clone the repository: `git clone git@github.com:lyst/lightfm.git`
2. Install it for development using pip: `cd lightfm && pip install -e .`
3. You can run tests by running `python setupy.py test`.

When making changes to the `.pyx` extension files, you'll need to run `python setup.py cythonize` in order to produce the extension `.c` files before running `pip install -e .`.

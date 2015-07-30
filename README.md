# lightfm-python

A Python implementation of LightFM, a hybrid recommendation algorithm.

## Installation
Install from pypi using pip: `pip install lightfm`.

## Examples
Check the `examples` directory for examples. The Movielens example shows how to use `lightfm` on the Movielens dataset, both with and without using movie metadata.

## Development
Pull requests are welcome. To install for development:

1. Clone the repository: `git clone git@github.com:lyst/lightfm.git`
2. Install it for development using pip: `cd lightfm && pip install -e .`
3. You can run tests by running `python setupy.py test`.

When making changes to the `.pyx` extension files, you'll need to run `python setup.py cythonize` in order to produce the extension `.c` files before running `pip install -e .`.
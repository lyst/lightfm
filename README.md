# LightFM

![LightFM logo](lightfm.png)

| Build status | |
|---|---|
| Linux |[![Circle CI](https://circleci.com/gh/lyst/lightfm.svg?style=svg)](https://circleci.com/gh/lyst/lightfm)|
| OSX (OpenMP disabled)|[![Travis CI](https://travis-ci.org/lyst/lightfm.svg?branch=master)](https://travis-ci.org/lyst/lightfm)|
| Windows (OpenMP disabled) |[![Appveyor](https://ci.appveyor.com/api/projects/status/6cqpqb6969i1h4p7/branch/master?svg=true)](https://ci.appveyor.com/project/maciejkula/lightfm/branch/master)|

[![Gitter chat](https://badges.gitter.im/gitterHQ/gitter.png)](https://gitter.im/lightfm-rec/Lobby) [![PyPI](https://img.shields.io/pypi/v/lightfm.svg)](https://pypi.python.org/pypi/lightfm/)

LightFM is a Python implementation of a number of popular recommendation algorithms for both implicit and explicit feedback. It's easy to use, fast (via multithreaded model estimation), and produces high quality results.

It also makes it possible to incorporate both item and user metadata into the traditional matrix factorization algorithms It represents each user and item as the sum of the latent representations of their features, thus allowing recommendations to generalise to new items (via item features) and to new users (via user features).

For more details, see the [Documentation](http://lyst.github.io/lightfm/docs/home.html).

## Articles and tutorials on using LightFM
1. [Learning to Rank Sketchfab Models with LightFM](http://blog.ethanrosenthal.com/2016/11/07/implicit-mf-part-2/)
2. [Metadata Embeddings for User and Item Cold-start Recommendations](http://building-babylon.net/2016/01/26/metadata-embeddings-for-user-and-item-cold-start-recommendations/)
3. [Recommendation Systems - Learn Python for Data Science](https://www.youtube.com/watch?v=9gBC9R-msAk)

## How to cite
Please cite LightFM if it helps your research. You can use the following BibTeX entry:
```
@inproceedings{DBLP:conf/recsys/Kula15,
  author    = {Maciej Kula},
  editor    = {Toine Bogers and
               Marijn Koolen},
  title     = {Metadata Embeddings for User and Item Cold-start Recommendations},
  booktitle = {Proceedings of the 2nd Workshop on New Trends on Content-Based Recommender
               Systems co-located with 9th {ACM} Conference on Recommender Systems
               (RecSys 2015), Vienna, Austria, September 16-20, 2015.},
  series    = {{CEUR} Workshop Proceedings},
  volume    = {1448},
  pages     = {14--21},
  publisher = {CEUR-WS.org},
  year      = {2015},
  url       = {http://ceur-ws.org/Vol-1448/paper4.pdf},
}
```

## Development
Pull requests are welcome. To install for development:

1. Clone the repository: `git clone git@github.com:lyst/lightfm.git`
2. Install it for development using pip: `cd lightfm && pip install -e .`
3. You can run tests by running `python setup.py test`.

When making changes to the `.pyx` extension files, you'll need to run `python setup.py cythonize` in order to produce the extension `.c` files before running `pip install -e .`.

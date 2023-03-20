# Changelog

## [1.17][2023-03-19]

### Fixed

- Re-Cythonized cython files to fix compilation errors with newer compilers.
- Fixed `np.object` usage in tests.

## [1.16][2020-11-27]

### Addded
- Set the `LIGHTFM_NO_CFLAGS` environment variable when building LightFM to prevent it from setting
  `-ffast-math` or `-march=native` compiler flags.

### Changed
- `predict` now returns float32 predictions.

## [1.15][2018-05-26]
### Added
- Added a check that there is no overlap between test and train in `predict_ranks` (thanks to [@artdgn](https://github.com/artdgn)).
- Added dataset builder functionality.
### Fixed
- Fixed error message when item features have the wrong dimensions.
- Predict now checks for overflow in inputs to predict.
- WARP fitting is now numerically stable when there are very few items to
  draw negative samples from (< max_sampled).

## [1.14][2017-11-18]
### Added
- added additional input checks for non-normal inputs (NaNs, infinites) for features
- added additional input checks for non-normal inputs (NaNs, infinites) for interactions
- cross validation module with dataset splitting utilities
### Changed
- LightFM model now raises a ValueError (instead of assertion) when the number of supplied
  features exceeds the number of estimated feature embeddings.
- Warn and delete downloaded file when Movielens download is corrputed. This happens in the wild
  cofuses users terribly.

## [1.13][2017-05-20]
### Added
- added get_{user/item}_representations functions to facilitate extracting the latent representations out of the model.
### Fixed
- recall_at_k and precision_at_k now work correctly at k=1 (thanks to Zank Bennett).
- Moved Movielens data to data release to prevent grouplens server flakiness from affecting users.
- Fix segfault when trying to predict from a model that has not been fitted.

## [1.12][2017-01-26]
### Changed
- Ranks are now computed pessimistically: when two items are tied, the positive item is assumed to have higher rank. This will lead to zero precision scores for models that predict all zeros, for example.
- The model will raise a ValueError if, during fitting, any of the parameters become non-finite (NaN or +/- infinity).
- Added mid-epoch regularization when a lot of regularization is used. This reduces the likelihood of numerical instability at high regularization rates.


## [1.11][2016-12-26]
### Changed
- negative samples in BPR are now drawn from the empirical distributions of positives. This improves accuracy slightly on the Movielens 100k dataset.

### Fixed
- incorrect calculation of BPR loss (thanks to @TimonVS for reporting this).


## [1.10][2016-11-25]
### Added
- added recall@k evaluation function
### Fixed
- added >=0.17.0 scipy depdendency to setup.py
- fixed segfaults on when duplicate entries are present in input COO matrices (thanks to Florian
  Wilhelm for the bug report).

## [1.9][2016-05-25]
### Fixed
- fixed gradient accumulation in adagrad (the feature value is now correctly used when accumulating gradient).
  Thanks to Benjamin Wilson for the bug report.
- all interaction values greater than 0.0 are now treated as positives for ranking losses.
### Added
- max_sampled hyperparameter for WARP losses. This allows trading off accuracy for WARP training time: a smaller value
  will mean less negative sampling and faster training when the model is near the optimum.
- Added a sample_weight argument to fit and fit_partial functions. A high value will now increase the size of the SGD step taken for that interaction.
- Added an evaluation module for more efficient evaluation of learning-to-rank models.
- Added a random_state keyword argument to LightFM to allow repeatable model runs.
### Changed
- By default, an OpenMP-less version will be built on OSX. This allows much easier installation at the expense of
performance.
- The default value of the max_sampled argument is now 10. This represents a decent default value that allows fast training.

## [1.8][2016-01-14]
### Changed
- fix scipy missing from requirements in setup.py
- remove dependency on glibc by including a translation of the musl rand_r implementation

## [1.7][2015-10-14]
### Changed
- fixed bug where item momentum would be incorrectly used in adadelta training for user features (thanks to Jong Wook Kim @jongwook for the bug report).
- user and item features are now floats (instead of ints), allowing fractional feature weights to be used when fitting models.

## [1.6][2015-09-29]
### Changed
- when installing into an Anaconda distribution, drop -march=native compiler flag
  due to assembler issues.
- when installing on OSX, search macports and homebrew install location for gcc
  version 5.x

## [1.5][2015-09-24]
### Changed
- when installing on OSX, search macports install location for gcc

## [1.4][2015-09-18]
### Changed
- input matrices automatically converted to correct dtype if necessary

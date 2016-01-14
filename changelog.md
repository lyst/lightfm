# Changelog

## [Unreleased][unreleased]
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

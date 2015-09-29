# Changelog

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

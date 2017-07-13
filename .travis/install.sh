#!/bin/bash

set -e

# Download and install conda
if [[ $TRAVIS_OS_NAME == 'osx' ]]; then
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ~/miniconda.sh
else
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
fi
bash ~/miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
rm ~/miniconda.sh

# Set up the right Python version
conda install -y python=$PYTHON_VERSION

# Install dependencies
conda install -y numpy scipy requests scikit-learn pytest

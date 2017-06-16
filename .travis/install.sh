#!/bin/bash

set -e

# Download and install conda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
rm ~/miniconda.sh

# Set up the right Python version
conda install python=$PYTHON_VERSION

# Install dependencies
conda install numpy scipy requests scikit-learn pytest

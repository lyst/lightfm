FROM ubuntu:16.04

RUN apt-get update
RUN apt-get install -y libxml2 libxslt-dev wget bzip2 gcc

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

RUN conda install pytest jupyter

ENV PYTHONDONTWRITEBYTECODE 1

ADD . /home/lightfm/
WORKDIR /home/

RUN pip install lightfm

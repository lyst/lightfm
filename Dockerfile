FROM ubuntu:15.04

RUN apt-get update
RUN apt-get install -y libxml2 libxslt-dev
RUN apt-get install -y python-numpy python-scipy python-scikits-learn python-pip
RUN pip install pytest jupyter

ENV PYTHONDONTWRITEBYTECODE 1

ADD . /home/lightfm/
WORKDIR /home/lightfm/

RUN pip install .

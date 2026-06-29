FROM debian:trixie-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    gcc \
    libgomp1 \
    libxml2 \
    libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1

ADD . /home/lightfm/
WORKDIR /home/lightfm/

RUN pip install --break-system-packages -e .

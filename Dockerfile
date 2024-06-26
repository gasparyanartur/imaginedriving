FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

## Set non-interactive to prevent asking for user inputs blocking image creation.
ENV DEBIAN_FRONTEND=noninteractive

## Set timezone as it is required by some packages.
ENV TZ=Europe/Berlin

# CUDA Home, required to find CUDA in some packages.
ENV CUDA_HOME="/usr/local/cuda"  

ENV CUDA_VERSION=11.8.0
ENV CUDA_VER=118
ENV OS_VERSION=22.04

RUN apt-get update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3.10-distutils \
        build-essential \
        cmake \
        curl \
        ffmpeg \
        git \
        sudo \
        vim-tiny \
        wget \
        python-is-python3 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove
    
COPY requirements.txt .
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

WORKDIR /workspace
CMD /bin/bash -l
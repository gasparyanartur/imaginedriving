FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV PYTHONNOUSERSITE=True
## Set non-interactive to prevent asking for user inputs blocking image creation.
ENV DEBIAN_FRONTEND=noninteractive
## Set timezone as it is required by some packages.
ENV TZ=Europe/Berlin
## CUDA Home, required to find CUDA in some packages.
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH=/opt/mambaforge/bin:$PATH

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    nano \
    wget \
    curl

# Mambaforge
RUN cd /tmp && \
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh" && \
    bash Mambaforge-$(uname)-$(uname -m).sh -fp /opt/mambaforge -b && \
    rm Mambaforge*sh

RUN mamba install \
    python=3.11 \
    pytorch=2.2.0 \
    torchvision \
    pytorch-cuda=12.1 \
    transformers \
    accelerate \
    diffusers \
    safetensors \
    xformers \
    matplotlib \
    scipy \
    pandas \
    jupyter \
    jupyterlab \
    nb_conda_kernels \
    ipykernel \
    tqdm \
    -c conda-forge \
    -c pytorch \
    -c nvidia \
    -c huggingface \
    -c xformers \
    -y

# Create nonroot user, setup env
RUN useradd -m -d /home/user -g root -G sudo -u 1000 user
RUN usermod -aG sudo user
# User password
RUN echo "user:user" | chpasswd
# Ensure sudo group users are not asked for password
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
#Switch to new user and workdir
USER 1000
WORKDIR /home/user
# Add local user binary folder to Path variable
ENV PATH="${PATH}:/home/user/.local/bin"
SHELL ["/bin/bash", "-c"]

WORKDIR /workspace

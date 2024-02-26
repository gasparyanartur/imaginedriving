FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV PYTHONNOUSERSITE=True
## Set non-interactive to prevent asking for user inputs blocking image creation.
ENV DEBIAN_FRONTEND=noninteractive
## Set timezone as it is required by some packages.
ENV TZ=Europe/Berlin
## CUDA Home, required to find CUDA in some packages.
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH=/opt/mambaforge/bin:$PATH
ENV HF_HOME=/proj/nlp4adas/users/x_artga/nerf-thesis/.cache
ENV MPLCONFIGDIR=/proj/nlp4adas/users/x_artga/nerf-thesis/.cache

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

COPY environment.yml .
RUN mamba env update -f environment.yml 

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

RUN echo "source activate env-diffusion" > ~/.bashrc
ENV PATH /opt/mambaforge/envs/env-diffusion/bin:$PATH
SHELL ["/bin/bash", "-c"]

WORKDIR /workspace

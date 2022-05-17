# Base image must at least have pytorch and CUDA installed.
# We are using NVIDIA NGC's PyTorch image here, see: https://ngc.nvidia.com/catalog/containers/nvidia:pytorch for latest version
# See https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html#framework-matrix-2021 for installed python, pytorch, etc. versions
FROM nvcr.io/nvidia/pytorch:20.12-py3

# Set path to CUDA
ENV CUDA_HOME=/usr/local/cuda

# Install additional programs
RUN apt update && \
    apt install -y build-essential \
    sudo \
    lsb-release \
    htop \
    wget \
    gnupg \
    git \
    curl \
    ca-certificates \
    vim \
    tmux && \
    rm -rf /var/lib/apt/lists

# Update pip
RUN SHA=ToUcHMe which python3
RUN python3 -m pip install --upgrade pip

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

# Install dependencies 
RUN python3 -m pip install pybind11
RUN python3 -m pip install six
RUN python3 -m pip install regex
RUN python3 -m pip install numpy
RUN python3 -m pip install wandb
RUN python3 -m pip install deepspeed
RUN python3 -m pip install nltk

# RUN conda install -c conda-forge xgboost

# Specify a new user (USER_NAME and USER_UID are specified via --build-arg)
ARG USER_UID
ARG USER_NAME
ENV USER_GID=$USER_UID
ENV USER_GROUP="users"

# Create the user
RUN mkdir /home/$USER_NAME 
RUN useradd -l -d /home/$USER_NAME -u $USER_UID -g $USER_GROUP $USER_NAME

# Change owner of home dir
RUN chown -R ${USER_UID}:${USER_GID} /home/$USER_NAME/

# Change owner of conda/python dir
# - This is needed to be able to install transformers inside the container using pip -e 
RUN chown ${USER_UID}:${USER_GID} /opt/conda/lib/python3.8/site-packages/
RUN chown -R ${USER_UID}:${USER_GID} /opt/conda/bin/

# Set workdir when starting container
WORKDIR /Megatron-DeepSpeed

# Add workdir to PYTHONPATH
ENV PYTHONPATH="$PYTHONPATH:/Megatron-DeepSpeed"

CMD ["/bin/bash"]
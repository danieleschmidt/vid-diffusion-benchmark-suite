FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    sudo \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgomp1 \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create vscode user
RUN groupadd --gid 1000 vscode \
    && useradd --uid 1000 --gid vscode --shell /bin/bash --create-home vscode \
    && echo 'vscode ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Install conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -ya

# Add conda to path
ENV PATH=/opt/conda/bin:$PATH

# Install Python packages
RUN pip install --upgrade pip && \
    pip install \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

# Install development tools
RUN pip install \
    black \
    isort \
    pylint \
    mypy \
    pre-commit \
    jupyter \
    jupyterlab \
    streamlit \
    pytest \
    pytest-cov \
    pytest-mock \
    requests \
    psutil \
    numpy \
    scipy \
    opencv-python \
    Pillow

USER vscode
WORKDIR /workspace
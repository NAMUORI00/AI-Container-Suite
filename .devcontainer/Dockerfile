FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# 시스템 기본 패키지 업데이트 및 설치
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    ca-certificates \
    build-essential \
    pkg-config \
    software-properties-common \
    unzip \
    vim \
    libssl-dev \
    libffi-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Miniconda 설치 (최신 버전)
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh && \
    ln -s $CONDA_DIR/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find $CONDA_DIR/ -follow -type f -name '*.a' -delete && \
    find $CONDA_DIR/ -follow -type f -name '*.js.map' -delete && \
    $CONDA_DIR/bin/conda clean -afy

# 환경 변수 설정
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 작업 디렉토리 생성
RUN mkdir -p /workspace/src /workspace/notebooks

# 현재 작업 디렉토리 설정
WORKDIR /workspace

# 사용자 설정 (선택 사항)
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# 사용자 생성
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

# 소유권 설정
RUN chown -R $USERNAME:$USERNAME /workspace

# 환경 설정 파일 복사
# Docker 빌드 컨텍스트 내의 파일을 복사
COPY .devcontainer/environment.yml /tmp/environment.yml
COPY .devcontainer/requirements.txt /tmp/requirements.txt

# Conda 환경 생성 및 활성화
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy && \
    echo "conda activate pytorch_env" >> ~/.bashrc

# NumPy 설치 (TensorRT는 건너뛰고 나중에 필요할 때 설치)
RUN . $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate pytorch_env && \
    conda install -y numpy=1.24.* && \
    conda clean -afy

# 필요한 Python 패키지 설치
RUN . $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate pytorch_env && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install pycuda

# 기본 사용자 전환
USER $USERNAME
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
    fontconfig \
    fonts-nanum* \
    fonts-noto-cjk \
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

# 폰트 캐시 업데이트 및 한글 폰트 설정
RUN fc-cache -fv && \
    mkdir -p /tmp/matplotlib_config

# Matplotlib 설정 파일 생성
RUN echo "import matplotlib.pyplot as plt" > /tmp/matplotlib_config/matplotlibrc.py && \
    echo "plt.rcParams['font.family'] = 'NanumGothic'" >> /tmp/matplotlib_config/matplotlibrc.py && \
    echo "plt.rcParams['axes.unicode_minus'] = False" >> /tmp/matplotlib_config/matplotlibrc.py && \
    mkdir -p /home/$USERNAME/.config/matplotlib && \
    cp /tmp/matplotlib_config/matplotlibrc.py /home/$USERNAME/.config/matplotlib/ && \
    chown -R $USERNAME:$USERNAME /home/$USERNAME/.config

# 주피터 노트북에서 sudo 실행을 위한 추가 설정
# 현대적인 방식으로 jupyter에서 sudo 명령어 사용 지원
RUN . $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate pytorch_env && \
    pip install ipython ipykernel

# sudo 설정: 패스워드 없이 sudo 명령어 실행 가능하도록 설정
RUN echo "Defaults env_keep += \"PYTHONPATH PYTHONUSERBASE JUPYTER_PATH JUPYTER_DATA_DIR JUPYTER_RUNTIME_DIR JUPYTER_CONFIG_DIR\"" >> /etc/sudoers.d/jupyter && \
    chmod 0440 /etc/sudoers.d/jupyter && \
    echo 'Defaults    !requiretty' >> /etc/sudoers.d/jupyter

# 노트북 내 !sudo 명령어 사용 설정 파일 생성
RUN mkdir -p /home/$USERNAME/.jupyter && \
    echo "c.TerminalInteractiveShell.extra_env = {'PYTHONPATH': ':'.join(sys.path)}" >> /home/$USERNAME/.jupyter/jupyter_config.py && \
    chown -R $USERNAME:$USERNAME /home/$USERNAME/.jupyter

# 기본 사용자 전환
USER $USERNAME
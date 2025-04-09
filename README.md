# 딥러닝 개발용 Docker 환경 설정

NVIDIA CUDA, PyTorch, TensorRT 및 Conda를 포함한 딥러닝 개발용 Docker 환경을 VS Code DevContainer로 설정하기 위한 프로젝트입니다.

## 요구사항

이 프로젝트는 다음 요구사항을 충족합니다:

- NVIDIA CUDA 12.1 최신 기반 이미지 사용 (`nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04`)
- PyTorch 2.2 이상 최신 버전, TensorRT 10.0 이상 최신 버전, Conda 미니포지 최신 버전 설치
- VSCode DevContainer로 개발할 수 있는 설정 제공
- 컨테이너 내부에 'workspace' 폴더 구조 설정 (notebooks, src 폴더 포함)
- 필요한 파이썬 라이브러리 설치를 위한 requirements.txt 및 environment.yml 파일 구성
- CUDA와 GPU 작동 확인을 위한 간단한 테스트 코드 예시 포함

## 파일 구조

```
.
├── .devcontainer/
│   ├── Dockerfile          # Docker 이미지 빌드 설정
│   └── devcontainer.json   # VS Code DevContainer 설정
├── environment.yml         # Conda 환경 설정
├── requirements.txt        # 추가 Python 패키지
├── workspace/
│   ├── notebooks/
│   │   └── example.ipynb   # PyTorch CUDA 테스트용 노트북
│   └── src/
│       └── main.py         # 간단한 GPU 확인 코드
└── README.md               # 이 파일
```

## 운영체제 호환성

이 도커 환경은 다음 운영체제에서 사용할 수 있습니다:

- **Linux**: 완벽하게 지원됩니다. 대부분의 Linux 배포판(Ubuntu, Debian, CentOS 등)에서 Docker 및 NVIDIA Container Toolkit을 설치하여 사용할 수 있습니다.
- **Windows**: Windows 10/11 Professional 이상 버전에서 지원됩니다.
  - WSL2(Windows Subsystem for Linux 2)가 설치되어 있어야 합니다.
  - Docker Desktop for Windows가 WSL2 백엔드로 구성되어 있어야 합니다.
  - NVIDIA GPU 사용을 위해 WSL2용 NVIDIA 드라이버가 설치되어 있어야 합니다.
- **macOS**: 
  - M1/M2/M3(Apple Silicon) Mac: PyTorch는 지원되나 NVIDIA CUDA는 지원되지 않습니다. 일부 기능은 제한될 수 있습니다.
  - Intel Mac: CUDA를 지원하지 않기 때문에 GPU 가속 관련 기능은 사용할 수 없습니다.

## 사용 방법

1. 필수 요구사항:
   - [Docker Desktop](https://www.docker.com/products/docker-desktop/) 설치
   - [VS Code](https://code.visualstudio.com/) 설치
   - VS Code의 [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) 확장 설치
   - NVIDIA GPU와 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) 설치

2. 이 저장소를 로컬에 복제합니다.

3. VS Code에서 복제한 폴더를 엽니다.

4. VS Code 명령 팔레트(`F1` 또는 `Ctrl+Shift+P`)를 열고 "Remote-Containers: Reopen in Container"를 선택합니다.

5. 컨테이너가 빌드되고 VS Code가 컨테이너 내부 환경에서 실행됩니다. 이 과정은 처음 실행 시 몇 분 정도 소요될 수 있습니다.

6. 컨테이너가 실행되면 다음을 수행할 수 있습니다:
   - `/workspace/notebooks/example.ipynb` 노트북을 열어서 PyTorch와 CUDA가 작동하는지 테스트
   - 터미널에서 `python /workspace/src/main.py`를 실행하여 GPU 작동 상태를 확인

## Docker 직접 빌드 및 실행 방법

VS Code DevContainer를 사용하지 않고 직접 Docker 명령어로 환경을 구축하고 싶은 경우, 다음 명령어를 사용할 수 있습니다:

### Docker 이미지 빌드

프로젝트 루트 디렉토리에서 다음 명령어를 실행하여 Docker 이미지를 빌드합니다:

```bash
# Windows CMD
docker build --pull --rm -f ".devcontainer\Dockerfile" -t huggingfacetutorial:latest .

# Windows PowerShell
docker build --pull --rm -f ".devcontainer/Dockerfile" -t huggingfacetutorial:latest .

# Linux/macOS
docker build --pull --rm -f ".devcontainer/Dockerfile" -t huggingfacetutorial:latest .
```

### Docker 컨테이너 실행

빌드된 이미지로 GPU를 사용하는 컨테이너를 실행합니다:

```bash
docker run --gpus all -it --name huggingface_dev -v "$(pwd):/workspace" -p 8888:8888 huggingfacetutorial:latest
```

### Jupyter Notebook 서버 실행 (선택사항)

컨테이너 내부에서 Jupyter Notebook 서버를 실행하려면:

```bash
# 컨테이너 내부에서 실행
conda activate pytorch_env
cd /workspace
jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root
```

그런 다음 웹 브라우저에서 출력된 URL(일반적으로 `http://127.0.0.1:8888/?token=...`)에 접속하면 Jupyter 노트북을 사용할 수 있습니다.

## Jupyter 노트북 활용 가이드

이 환경에는 데이터 분석, 모델 학습 및 테스트를 위한 다양한 Jupyter 기능이 포함되어 있습니다:

### 주요 Jupyter 기능

- **JupyterLab**: 향상된 웹 기반 인터페이스로, 노트북, 코드 편집기, 터미널 등을 통합 환경에서 사용 가능
- **Jupyter Notebooks**: 코드 실행, 마크다운 문서화, 데이터 시각화를 하나의 문서에서 수행 가능
- **Interactive Widgets**: `ipywidgets`를 통해 대화형 UI 요소 제공
- **코드 포맷팅**: `jupyter-black` 패키지를 통해 노트북 내에서 코드 자동 포맷팅 지원

### Jupyter 서버 시작하기

#### VS Code에서 직접 사용

VS Code에서 Jupyter 노트북(.ipynb)을 직접 열고 실행할 수 있습니다:

1. VS Code에서 `.ipynb` 파일 열기
2. 오른쪽 상단에서 커널 선택 ("Select Kernel" → "Python Environments" → "pytorch_env")
3. 셀 실행 버튼이나 단축키(Shift+Enter)를 사용하여 코드 실행

#### JupyterLab 서버 시작하기

1. 컨테이너 터미널에서:
```bash
conda activate pytorch_env
cd /workspace
jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root
```

2. 출력된 URL(token 포함)을 브라우저에 붙여넣어 접속

#### Jupyter Notebook 서버 시작하기

```bash
conda activate pytorch_env
cd /workspace
jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root
```

### 노트북 예제 활용하기

이 프로젝트는 다음과 같은 노트북 예제를 제공합니다:

1. **`example.ipynb`**: 
   - PyTorch와 CUDA 설정 확인
   - GPU 성능 벤치마크
   - 간단한 CNN 모델 훈련 예제

2. **`test_notebook.ipynb`**: 
   - 데이터 생성 및 시각화 예제
   - matplotlib를 사용한 그래프 작성
   - pytest를 활용한 테스트 실행

노트북을 사용하여 직접 코드를 수정하고 실행해 보세요!

## GPU 테스트

GPU가 올바르게 설정되었는지 확인하려면:

1. Jupyter 노트북 방식:
   - VS Code에서 `/workspace/notebooks/example.ipynb` 열기
   - 모든 셀을 순차적으로 실행
   - 결과를 통해 CUDA 가용성, GPU 성능, TensorRT 지원 여부 확인

2. Python 스크립트 방식:
   - 터미널에서 다음 명령어 실행:
   ```bash
   python /workspace/src/main.py
   ```
   - 출력 결과를 통해 시스템 정보, CUDA 가용성, GPU 성능, TensorRT 지원 여부 확인

## 생성에 사용된 프롬프트

```
PyTorch, CUDA, Conda, TensorRT가 지원되는 딥러닝 개발용 도커 환경을 만들고 싶습니다.

요구사항:

NVIDIA CUDA 12.1 최신 기반 이미지 사용 (nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04)
PyTorch 2.2 이상 최신 버전, TensorRT 10.0 이상 최신 버전, Conda 미니포지 최신 버전 설치
VSCode DevContainer로 개발할 예정이므로 관련 설정 필요
컨테이너 내부에 'workspace' 폴더 구조 설정 (notebooks, src 폴더 포함)
필요한 파이썬 라이브러리 설치를 위한 requirements.txt 및 environment.yml 파일 구성
CUDA와 GPU 작동 확인을 위한 간단한 테스트 코드 예시 포함
다음 파일들을 생성해주세요:

.devcontainer/Dockerfile
.devcontainer/devcontainer.json
environment.yml
requirements.txt
workspace/notebooks/example.ipynb (PyTorch CUDA 테스트용)
workspace/src/main.py (간단한 GPU 확인 코드)
사용한 프롬프트는 readme에 추가해주세요
```

## 커스터마이징

필요에 따라 다음 파일을 수정하여 환경을 커스터마이징할 수 있습니다:

- `environment.yml`: Conda 환경 및 패키지 버전 수정
- `requirements.txt`: 추가 Python 패키지 수정
- `.devcontainer/Dockerfile`: Docker 이미지 빌드 설정 수정
- `.devcontainer/devcontainer.json`: VS Code 확장 및 설정 수정

## 주의사항

- Docker 컨테이너를 처음 빌드할 때 다소 시간이 걸릴 수 있습니다.
- 호스트 시스템에 호환되는 NVIDIA GPU 드라이버가 설치되어 있어야 합니다.
- 충분한 디스크 공간이 필요합니다 (최소 10GB 이상 권장).
- VSCode DevContainer 설정이 제공되어 있어 별도의 도커 명령어 실행 없이 VSCode에서 바로 컨테이너를 실행할 수 있습니다.
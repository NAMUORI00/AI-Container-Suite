# AI-Container-Suite

NVIDIA CUDA, PyTorch, Hugging Face Transformers 라이브러리를 포함한 딥러닝 개발용 Docker 환경을 VS Code DevContainer로 설정한 통합 AI 개발 환경입니다. 이 컨테이너 모음을 통해 최신 자연어 처리(NLP), 컴퓨터 비전(CV), 다양한 생성형 AI 모델을 학습하고 테스트할 수 있습니다.

## 주요 기능 및 특징

이 프로젝트는 다음과 같은 기능과 특징을 제공합니다:

- NVIDIA CUDA 12.1 기반 최신 이미지 사용 (`nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04`)
- PyTorch 2.2 이상 및 Hugging Face Transformers 라이브러리 지원
- Conda 환경 기반 패키지 관리로 의존성 충돌 방지
- VS Code DevContainer를 활용한 일관된 개발 환경 제공
- Jupyter Notebook에서 sudo 권한 사용 가능 (시스템 패키지 설치 등에 유용)
- 데이터 시각화 및 분석을 위한 예제 노트북 포함
- 한글 폰트 지원으로 데이터 시각화 시 한글 출력 가능

## 파일 구조

```
.
├── .devcontainer/
│   ├── Dockerfile          # Docker 이미지 빌드 설정
│   └── devcontainer.json   # VS Code DevContainer 설정
├── data/                   # 데이터 저장 디렉토리
├── workspace/
│   ├── notebooks/
│   │   ├── data_visualization.ipynb   # 데이터 시각화 예제 노트북
│   │   └── test_notebook.ipynb        # 테스트 및 기능 확인용 노트북
│   └── src/
│       └── main.py         # 기본 소스 코드
└── README.md               # 이 파일 (프로젝트 설명서)
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

## 시작하기

1. 필수 요구사항:
   - [Docker Desktop](https://www.docker.com/products/docker-desktop/) 설치
   - [VS Code](https://code.visualstudio.com/) 설치
   - VS Code의 [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) 확장 설치
   - NVIDIA GPU와 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) 설치

2. 이 저장소를 로컬에 복제합니다.

3. VS Code에서 복제한 폴더를 엽니다.

4. VS Code 명령 팔레트(`F1` 또는 `Ctrl+Shift+P`)를 열고 "Dev Containers: Reopen in Container"를 선택합니다.

5. 컨테이너가 빌드되고 VS Code가 컨테이너 내부 환경에서 실행됩니다. 이 과정은 처음 실행 시 몇 분 정도 소요될 수 있습니다.

6. 컨테이너가 실행되면 다음을 수행할 수 있습니다:
   - `/workspace/notebooks/data_visualization.ipynb` 노트북으로 데이터 시각화 예제 확인
   - `/workspace/notebooks/test_notebook.ipynb` 노트북으로 환경 설정 테스트
   - 터미널에서 `python /workspace/src/main.py`를 실행하여 기본 기능 확인

## Docker 직접 빌드 및 실행 방법

VS Code DevContainer를 사용하지 않고 직접 Docker 명령어로 환경을 구축하고 싶은 경우, 다음 명령어를 사용할 수 있습니다:

### Docker 이미지 빌드

프로젝트 루트 디렉토리에서 다음 명령어를 실행하여 Docker 이미지를 빌드합니다:

```bash
# Windows CMD
docker build --pull --rm -f ".devcontainer\Dockerfile" -t ai-container-suite:latest .

# Windows PowerShell
docker build --pull --rm -f ".devcontainer/Dockerfile" -t ai-container-suite:latest .

# Linux/macOS
docker build --pull --rm -f ".devcontainer/Dockerfile" -t ai-container-suite:latest .
```

### Docker 컨테이너 실행

빌드된 이미지로 GPU를 사용하는 컨테이너를 실행합니다:

```bash
docker run --gpus all -it --name ai_container_suite -v "$(pwd):/workspace" -p 8888:8888 ai-container-suite:latest
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
- **코드 포맷팅**: 노트북 내에서 코드 자동 포맷팅 지원

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

1. **`data_visualization.ipynb`**: 
   - 데이터 생성 및 시각화 예제
   - matplotlib를 사용한 그래프 작성
   - 한글 폰트 설정 및 사용 예제

2. **`test_notebook.ipynb`**: 
   - PyTorch와 CUDA 설정 확인
   - 기본 환경 설정 테스트
   - GPU 가용성 확인

## GPU 테스트

GPU가 올바르게 설정되었는지 확인하려면:

1. Jupyter 노트북 방식:
   - VS Code에서 `/workspace/notebooks/test_notebook.ipynb` 열기
   - 모든 셀을 순차적으로 실행
   - 결과를 통해 CUDA 가용성, GPU 성능, Hugging Face 모델 호출 지원 여부 확인

2. Python 스크립트 방식:
   - 터미널에서 다음 명령어 실행:
   ```bash
   python /workspace/src/main.py
   ```
   - 출력 결과를 통해 시스템 정보, CUDA 가용성, GPU 성능 확인

## 주요 설치된 패키지

- **PyTorch**: 딥러닝 프레임워크 (버전 2.2.x)
- **Hugging Face Transformers**: 최신 NLP 모델 및 다양한 생성형 AI 모델 지원
- **NumPy, Pandas**: 데이터 처리 및 분석
- **Matplotlib, Seaborn**: 데이터 시각화
- **Jupyter**: 대화형 개발 환경
- **ipywidgets**: 인터랙티브 위젯
- **CUDA 관련 라이브러리**: GPU 가속 연산 지원

## 커스터마이징

필요에 따라 다음 파일을 수정하여 환경을 커스터마이징할 수 있습니다:

- `.devcontainer/environment.yml`: Conda 환경 및 패키지 버전 수정
- `.devcontainer/requirements.txt`: 추가 Python 패키지 수정
- `.devcontainer/Dockerfile`: Docker 이미지 빌드 설정 수정
- `.devcontainer/devcontainer.json`: VS Code 확장 및 설정 수정

## 주의사항

- Docker 컨테이너를 처음 빌드할 때 다소 시간이 걸릴 수 있습니다 (10-20분).
- 호스트 시스템에 호환되는 NVIDIA GPU 드라이버가 설치되어 있어야 합니다.
- 충분한 디스크 공간이 필요합니다 (최소 10GB 이상 권장).
- VS Code DevContainer 설정이 제공되어 있어 별도의 도커 명령어 실행 없이 VS Code에서 바로 컨테이너를 실행할 수 있습니다.
- Jupyter Notebook에서 sudo 명령어 사용 시 주의가 필요합니다.

## 문제 해결

- **CUDA 오류**: NVIDIA 드라이버가 최신 버전인지 확인하고, `nvidia-smi` 명령어로 GPU 상태 확인
- **메모리 부족 오류**: 대용량 모델 로드 시 메모리가 부족한 경우, 모델 설정에서 `low_cpu_mem_usage=True`, `device_map="auto"` 등의 옵션 활용
- **Jupyter 오류**: 커널이 응답하지 않는 경우, 커널 재시작 후 시도
- **Docker 빌드 실패**: Docker 로그를 확인하여 구체적인 오류 메시지 분석 후 해결

## 커밋 메시지 컨벤션

이 프로젝트는 [Conventional Commits](https://www.conventionalcommits.org/ko/v1.0.0/) 규칙을 따릅니다. 커밋 메시지는 다음 형식을 준수해야 합니다:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### 커밋 타입 (type)

- `feat`: 새로운 기능 추가
- `fix`: 버그 수정
- `docs`: 문서 변경
- `style`: 코드 포맷팅, 세미콜론 누락 등 (코드 변경 없음)
- `refactor`: 코드 리팩토링
- `test`: 테스트 코드 추가 또는 수정
- `chore`: 빌드 프로세스, 패키지 매니저 설정 등의 변경 (소스코드 변경 없음)
- `perf`: 성능 개선 코드
- `ci`: CI 설정 파일 및 스크립트 변경
- `build`: 빌드 시스템 또는 외부 의존성 변경
- `revert`: 이전 커밋으로 되돌리는 경우

### 예시

```bash
# 새로운 기능 추가
git commit -m "feat: 한글 폰트 지원 추가"

# 특정 범위(scope)의 버그 수정
git commit -m "fix(notebook): Jupyter 노트북에서 GPU 메모리 누수 문제 해결"

# 문서 업데이트
git commit -m "docs: README.md 설치 가이드 업데이트"

# 여러 줄 커밋 메시지
git commit -m "feat: 데이터 시각화 기능 추가

- Matplotlib을 사용한 차트 생성 기능 
- Seaborn을 활용한 통계 그래프 지원
- 대용량 데이터셋 처리를 위한 메모리 최적화

Resolves: #123"
```

## 기여 방법

1. 이 레포지토리를 포크(Fork)합니다.
2. 새로운 기능 브랜치를 생성합니다 (`git checkout -b feature/amazing-feature`).
3. 변경 사항을 위 커밋 메시지 컨벤션에 맞춰 커밋합니다 (예: `git commit -m "feat: 새로운 기능 추가"`).
4. 브랜치에 푸시합니다 (`git push origin feature/amazing-feature`).
5. 풀 리퀘스트를 제출합니다.

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.
# 세종 페르소나 AI 실습

## 1. 처음 설치

JupyterLab의 Launcher에서 **Terminal**을 연다. 아래 명령으로 GPU 이름에 `NVIDIA L40S`가 표시되는지 확인한다.

```bash
nvidia-smi
```

프로젝트를 내려받고 해당 폴더로 이동한다.

```bash
cd ~
git clone https://github.com/whyz-dev/persona-ai.git
cd persona-ai
```

Python 3.12 가상환경과 필요한 패키지를 설치한다. 이 과정은 최초 한 번 수행한다.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate

uv pip install "vllm==0.29.0" -r requirements.txt --torch-backend=auto
```

설치가 끝나면 같은 터미널에서 다음 단계로 진행한다.

## 2. 터미널 1: 모델 서버 실행

다음 명령으로 모델을 GPU에 불러온다. 처음 실행할 때는 모델 파일도 다운로드한다.

```bash
vllm serve Qwen/Qwen3.5-27B-FP8 \
  --host 127.0.0.1 \
  --language-model-only \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager
```

`Application startup complete`가 표시될 때까지 기다린다. 다운로드가 끝난 뒤에도 모델을 불러오는 시간이 필요하다.

대화하는 동안 **터미널 1의 모델 서버를 계속 실행해 둔다.**

## 3. 터미널 2: 대화 프로그램 실행

JupyterLab에서 새 Terminal을 열고 다음 명령을 입력한다.

```bash
cd ~/persona-ai
source .venv/bin/activate
curl -fsS http://127.0.0.1:8000/v1/models
```

결과에 `Qwen/Qwen3.5-27B-FP8`이 표시되면 대화 프로그램을 실행한다.

```bash
python app.py --base-url http://127.0.0.1:8000/v1
```

모델 서버에 연결하려면 `--base-url`을 포함한 명령 전체를 입력한다.

## 4. 세종과 대화

`나:`가 표시되면 자신의 이름, 사는 시대, 하는 일을 소개한다. 이후 궁금한 점이나 자신의 이야기를 자유롭게 입력한다.

> 저는 2026년에서 온 지민입니다. 책을 만드는 일을 합니다.

| 입력 | 기능 |
| --- | --- |
| `/근거` | 직전 질문과 관련해 검색된 역사 자료와 출처 확인 |
| `/종료` | 대화 종료 |

한 번에 몇 문장씩 입력하며 대화를 이어 간다. 관련 역사 자료를 보고 싶으면 `/근거`를 입력한다.

## 5. 실습 종료

1. 터미널 2에서 `/종료`를 입력한다.
2. 터미널 1에서 `Ctrl+C`를 눌러 모델 서버를 종료한다.
3. GPU 사용을 마쳤으면 VESSL에서 해당 워크스페이스를 중지한다. 브라우저를 닫는 것만으로 워크스페이스가 중지되지는 않는다.

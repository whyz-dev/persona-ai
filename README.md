# persona-ai

1449년 말의 세종과 자유롭게 대화하는 CLI 프로그램이다. LangChain으로 역사 자료를 검색하고 모델에 대화와 함께 전달한다. 기본 실습 환경은 VESSL의 L40S 한 대와 vLLM으로 실행하는 Qwen3.5-27B-FP8이다. 하오체, 성격 해석과 방문객을 만나는 장면에는 창작이 포함되어 있다.

## 실습 구성

교사 한 명이 GPU 서버 한 대를 사용한다. JupyterLab에서 터미널 두 개를 열어 다음 프로그램을 각각 실행한다. 아래 Linux 명령은 노트북 셀이 아닌 Jupyter의 **Terminal**에 입력한다.

```text
터미널 1: vLLM → Qwen 모델을 GPU에 불러와 API로 제공
터미널 2: app.py → 역사 자료·프롬프트·대화 기록을 구성하여 API에 요청
```

모델 서버를 계속 실행해 두면 프롬프트나 자료를 수정할 때 대화 프로그램만 다시 실행할 수 있다. 두 프로그램은 같은 VESSL 서버에서 통신하므로 이 실습에는 별도 포트 공개나 SSH 터널 설정이 필요하지 않다.

## 프로젝트 구조

```text
app.py             모델 로딩, 자료 검색, CLI 대화
prompts.json       안내 문구, 첫 대사, 페르소나 프롬프트
data.json          역사 자료 18개, 검색어, 출처 28곳, 검수 메모
requirements.txt   서버 접속용 실행 패키지
requirements-local.txt  로컬 GPU 실행용 추가 패키지
README.md          실행 및 자료 편집 안내
```

가상환경, 모델 파일, 개인 설정과 `local/`의 검증 기록은 Git에서 제외한다.

## VESSL·Jupyter에서 처음 실행

### 1. GPU 확인과 프로젝트 다운로드

JupyterLab의 Launcher에서 Terminal을 연다. 먼저 GPU 이름에 `NVIDIA L40S`가 표시되는지 확인한다.

```bash
nvidia-smi
```

처음 실행할 때 저장소를 내려받는다. 이미 `~/persona-ai`가 있으면 해당 폴더로 이동하여 2단계부터 진행한다. 가상환경과 패키지 설치까지 마쳤으면 아래의 '다시 실행과 업데이트' 절차를 사용한다.

```bash
cd ~
git clone https://github.com/whyz-dev/persona-ai.git
cd persona-ai
```

### 2. Python 환경과 패키지 설치

Jupyter 자체의 Python 환경과 구분하여 프로젝트에 Python 3.12 가상환경을 만든다. `uv`는 Python 환경과 패키지를 설치하는 도구이다. 설치는 최초 한 번 수행한다.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate

uv pip install "vllm==0.29.0" -r requirements.txt --torch-backend=auto
```

`requirements.txt`는 대화 프로그램의 패키지 목록이며, vLLM은 위 명령에서 함께 설치한다. 이 환경에는 `requirements-local.txt`를 설치하지 않는다. 설치 방식은 [uv 공식 안내](https://docs.astral.sh/uv/getting-started/installation/)와 [vLLM 공식 안내](https://docs.vllm.ai/en/v0.29.0/getting_started/installation/gpu/)를 따른다.

### 3. 터미널 1에서 Qwen 모델 서버 실행

L40S 한 대에서 실행하기 위해 공식 FP8 모델을 사용한다. 처음 실행하면 모델 파일을 다운로드하고 GPU에 불러온다. [Qwen3.5-27B의 vLLM 실행 안내](https://recipes.vllm.ai/Qwen/Qwen3.5-27B)를 바탕으로, 실습에서는 입력 한도와 동시 처리 수를 줄였다.

```bash
vllm serve Qwen/Qwen3.5-27B-FP8 \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name sejong-qwen27b \
  --language-model-only \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --api-key persona-lab
```

| 설정 | 의미 |
| --- | --- |
| `127.0.0.1:8000` | 같은 서버에서 접속하는 API 주소 |
| `sejong-qwen27b` | 대화 프로그램이 요청할 모델 이름 |
| `--language-model-only` | 텍스트 대화에 필요한 부분만 사용 |
| `--max-model-len 8192` | 프롬프트·자료·대화 기록·출력을 합한 길이의 상한 |
| `--max-num-seqs 1` | 한 번에 처리하는 요청 수 |
| `--enforce-eager` | CUDA 그래프와 컴파일 최적화를 사용하지 않는 실행 설정 |
| `--api-key persona-lab` | 이 실습에서 사용할 서버 접속 키 |

다운로드 진행률이 100%가 되어도 모델 초기화는 계속될 수 있다. `Application startup complete`가 표시될 때까지 기다린다. **대화하는 동안 터미널 1의 모델 서버는 계속 실행해 둔다.**

### 4. 터미널 2에서 접속 확인과 대화

JupyterLab에서 새 Terminal을 연다. 새 터미널에는 앞서 활성화한 가상환경과 환경변수가 자동으로 적용되지 않으므로 다시 설정한다.

```bash
cd ~/persona-ai
source .venv/bin/activate
export OPENAI_API_KEY=persona-lab
```

`OPENAI_API_KEY`는 코드에서 사용하는 환경변수 이름이다. **OpenAI에서 발급받는 키가 아니다.** 값은 직접 실행한 vLLM의 `--api-key`와 같아야 한다. 이 예시에서는 `persona-lab`을 사용하며, 모델 요청은 같은 서버의 Qwen으로 전송된다.

먼저 모델 서버가 준비되었는지 확인한다.

```bash
curl -fsS http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

결과에 `sejong-qwen27b`가 표시되면 대화를 실행한다.

```bash
python app.py --base-url http://127.0.0.1:8000/v1
```

`--base-url`을 지정해야 vLLM 서버에 연결한다. `python app.py`만 실행하면 로컬 4B 모델을 불러오므로 실습에서는 위 명령 전체를 사용한다. 서버의 모델 이름을 바꾼 경우에는 `--model 모델이름`도 지정한다.

사용자 발언마다 Chat Completions 요청 한 번으로 답변을 생성한다. Qwen의 `enable_thinking`은 `false`로 지정하며, 모델에 보내는 프롬프트·자료 검색·대화 기록은 로컬 실행과 같다.

## 다시 실행과 업데이트

설치가 끝난 환경에서는 패키지를 다시 설치하지 않는다. 터미널 1의 모델 서버가 종료되어 있다면 다음 명령으로 환경을 활성화한 뒤 위의 `vllm serve` 명령을 실행한다. 이미 모델 서버가 실행 중이면 그대로 둔다.

```bash
cd ~/persona-ai
source .venv/bin/activate
```

터미널 2에서 대화 프로그램을 다시 시작할 때는 다음 명령을 사용한다.

```bash
cd ~/persona-ai
source .venv/bin/activate
export OPENAI_API_KEY=persona-lab
python app.py --base-url http://127.0.0.1:8000/v1
```

GitHub의 변경 사항을 받을 때는 대화를 `/종료` 또는 `Ctrl+C`로 끝낸 뒤 실행한다.

```bash
git pull --ff-only
python app.py --base-url http://127.0.0.1:8000/v1
```

위 업데이트 명령은 프로젝트 폴더에 있으며 가상환경과 키가 설정된 터미널 2를 기준으로 한다. 직접 수정한 파일이나 서로 다른 커밋 때문에 `git pull --ff-only`가 중단되면, 로컬 변경을 보존한 상태에서 원격 변경과 비교하여 병합해야 한다.

`prompts.json`, `data.json`, `app.py`를 변경하면 **대화 프로그램을 종료하고 다시 실행**한다. `/새대화`는 대화 기록만 초기화하며 파일 변경을 다시 읽지 않는다. 모델 서버를 다시 실행할 필요는 없다.

## 실습 종료

1. 터미널 2에서 `/종료`를 입력하여 대화를 끝낸다.
2. 터미널 1에서 `Ctrl+C`를 눌러 모델 서버를 종료한다.
3. GPU 사용을 마쳤으면 VESSL에서 해당 워크스페이스를 중지한다. 대화 프로그램이나 브라우저를 닫는 것만으로 워크스페이스가 중지되지는 않는다.

## 실행 중 문제 확인

| 증상 | 확인 방법 |
| --- | --- |
| `Connection refused` 또는 접속 실패 | 터미널 1의 모델 서버가 실행 중인지, 초기화가 끝났는지 확인하고 `/v1/models` 요청을 다시 실행한다. |
| `401 Unauthorized` | 터미널 2의 `OPENAI_API_KEY`와 모델 서버의 `--api-key` 값을 맞춘다. |
| `KeyError: 'OPENAI_API_KEY'` | 대화를 실행할 터미널에서 `export OPENAI_API_KEY=persona-lab`을 실행한다. |
| `llama-cpp-python`이 없거나 GGUF 파일을 찾을 수 없다는 오류 | 실행 명령에 `--base-url http://127.0.0.1:8000/v1`을 포함했는지 확인한다. |
| `Address already in use` | 같은 포트에서 모델 서버가 이미 실행 중인지 확인한다. 준비된 서버가 있으면 터미널 2에서 접속한다. |
| 한글 삭제 후 `surrogates not allowed` | 최신 코드를 받은 뒤 `app.py`를 재실행한다. Linux에서는 표준 `readline` 모듈로 한글 삭제와 커서 이동을 처리한다. |

## 120분 연수 준비

교사 30명이 동시에 실습할 때는 패키지 설치와 모델 다운로드를 사전에 준비한다. 교사들은 모델 서버 실행, API 접속 확인, 대화, 프롬프트·자료 수정과 결과 비교를 직접 수행한다. 수업 전에 각 워크스페이스에서 모델을 한 번 실행하여 캐시와 GPU 로딩까지 확인한다.

## 대화 방법

방문객의 이름, 사는 시대, 하는 일을 소개한 뒤 자유롭게 대화한다.

> 저는 2026년에서 온 지민입니다. 책을 만드는 일을 합니다.

| 입력 | 기능 |
| --- | --- |
| `/근거` | 직전 답변 생성에 제공한 자료와 원문 링크 확인 |
| `/새대화` | 자기소개와 대화 기록을 지우고 다시 시작 |
| `/종료` 또는 `/quit` | 종료 |
| `Ctrl+C` | 실행 중단 |

첫 자기소개와 최근 6회의 문답을 모델에 전달한다. 대화 기록은 파일로 저장하지 않으며 프로그램을 종료하면 사라진다. 로컬 모델의 입력 한도는 8,192토큰이며 API 모드의 한도는 서버 설정을 따른다. 한 번에 몇 문장씩 입력한다.

## JSON 편집

`prompts.json`의 `welcome`은 시작 안내, `opening`은 세종의 첫 대사, `persona`는 시스템 프롬프트이다. 긴 안내와 프롬프트는 줄별 배열로 저장하고 실행할 때 줄바꿈으로 연결한다. `persona`의 `{intro}`와 `{evidence}`는 자기소개와 검색 자료가 들어가는 자리이므로 유지한다.

`data.json`의 `documents`에 역사 자료가 들어 있다.

| 필드 | 내용 |
| --- | --- |
| `id`, `title`, `period` | 자료 식별자, 제목, 시기 |
| `content` | 확인된 역사적 사실 |
| `keywords` | 검색을 돕는 표현 목록 |
| `sources` | 출처의 `title`과 `url` 목록 |
| `notes` | 후대 사건이나 해석상의 주의점 |

현재 발언과 직전 사용자 발언으로 BM25 검색을 수행하여 가장 관련 있는 자료 한 개를 선택한다. 모델에는 제목·시기·확인된 내용만 전달한다. 검색어는 검색에만 사용하고, 출처와 검수 메모는 인물의 기억으로 전달하지 않는다. `/근거`는 제공한 자료를 보여 주며 답변의 모든 문장이 검증되었다는 뜻은 아니다.

JSON 파일을 수정한 뒤 프로그램을 다시 실행하면 변경 사항이 적용된다.

## 확인 범위

VESSL의 L40S에서 **Qwen3.5-27B-FP8** 모델의 API 응답과 CLI 대화를 확인했다. 질문을 반복하던 프롬프트를 수정한 뒤 4개 시나리오·13개 사용자 발언을 확인했으며, 해당 테스트에서는 낯선 개념의 뜻을 확인하는 질문 한 번만 나왔다. 이 결과가 모든 대화의 자연스러움이나 역사적 정확성을 보장하지는 않는다.

한글 입력은 서버의 별도 가상 터미널에서 일반 입력, 한글 삭제, 커서 이동 후 삽입을 확인했다. 브라우저의 한글 입력기 조합 과정 전체를 검증한 것은 아니다.

로컬 **Qwen3-4B-Instruct-2507-Q4_K_M**은 22개 시나리오·52개 사용자 발언으로 대화를 확인했으며, 어색한 어미와 질문 반복, 일부 사실 혼동이 남아 있다.

재시도, 응답 자동 복구, 대체 모델은 구현하지 않았으며 실행 오류는 그대로 표시한다. `rope_freq_base=0`은 GGUF에 저장된 Qwen의 RoPE 설정을 사용하기 위한 값이다.

## 선택 사항: Windows 로컬 4B 실행

VESSL 실습과 별도로, Windows의 Python 3.12와 NVIDIA GPU에서 로컬 4B 모델을 실행할 수 있다. CUDA 12.4용 `llama-cpp-python` 바이너리를 사용하며 GGUF 모델 파일은 별도로 준비한다.

저장소를 내려받은 뒤 프로젝트 루트에서 실행한다.

```powershell
git clone https://github.com/whyz-dev/persona-ai.git
cd persona-ai
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
.\.venv\Scripts\python.exe -X utf8 app.py --model 'C:\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf'
```

가상환경이 활성화되어 있고 기본 모델 경로를 사용하는 경우 다음 명령으로 실행한다.

```powershell
python app.py
```

기본 모델 경로는 사용자 홈의 `.cache/persona-ai-lab/Qwen3-4B-Instruct-2507-Q4_K_M.gguf`이다. 이 경로에 모델이 있으면 `--model`을 생략할 수 있다.

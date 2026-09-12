# persona-ai

1449년 말의 세종과 자유롭게 대화하는 CLI 프로그램이다. LangChain으로 역사 자료를 검색하고 모델에 대화와 함께 전달한다. 로컬 Qwen3-4B 또는 vLLM 서버의 Qwen3.5-27B에 연결할 수 있다. 하오체, 성격 해석과 방문객을 만나는 장면에는 창작이 포함되어 있다.

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

## 로컬 4B 실행

Python 3.12와 NVIDIA GPU를 사용한다. 현재 패키지 구성은 CUDA 12.4용 `llama-cpp-python` 바이너리를 사용한다. GGUF 모델 파일은 별도로 준비한다.

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

## vLLM 서버에 연결

서버 접속용 패키지에는 `llama-cpp-python`과 모델 파일이 필요하지 않다. Python 3.12 환경에서 다음 명령으로 설치한다.

```bash
python -m pip install -r requirements.txt
```

vLLM 서버는 `Qwen/Qwen3.5-27B-FP8`을 `--served-model-name sejong-qwen27b`로 실행한다. 서버와 같은 환경에서 CLI를 실행하거나 SSH 터널로 서버의 8000번 포트에 연결한다. 다음 주소는 CLI를 실행하는 환경의 로컬 주소이다.

```powershell
$env:OPENAI_API_KEY = '서버에 설정한 API 키'
python app.py --base-url http://127.0.0.1:8000/v1
```

Linux 터미널에서는 환경변수를 다음과 같이 설정한다.

```bash
export OPENAI_API_KEY='서버에 설정한 API 키'
python app.py --base-url http://127.0.0.1:8000/v1
```

서버의 모델 이름이 다르면 `--model 모델이름`을 추가한다. API 키는 코드나 저장소에 넣지 않는다. vLLM은 `127.0.0.1`에 바인딩하고 같은 환경 또는 SSH 터널에서 접속하며, 인증 없는 API를 외부에 공개하지 않는다.

`--base-url`을 지정할 때만 서버를 사용한다. 사용자 발언마다 Chat Completions 요청 한 번으로 답변을 생성하며, Qwen의 `enable_thinking`은 `false`로 지정한다. 실패한 요청은 재시도하지 않고 오류를 표시한다. 서버에 전달하는 프롬프트·자료 검색·대화 기록은 로컬 실행과 같다.

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

로컬 **Qwen3-4B-Instruct-2507-Q4_K_M**은 22개 시나리오·52개 사용자 발언으로 대화를 확인했으며, 어색한 어미와 질문 반복, 일부 사실 혼동이 남아 있다. 서버 접속 기능의 검증과 Qwen3.5-27B의 실제 대화 품질 검증은 구분한다.

재시도, 응답 자동 복구, 대체 모델은 구현하지 않았으며 실행 오류는 그대로 표시한다. `rope_freq_base=0`은 GGUF에 저장된 Qwen의 RoPE 설정을 사용하기 위한 값이다.

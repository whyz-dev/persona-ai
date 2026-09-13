"""1449년 말의 세종과 대화하는 페르소나 CLI."""

import argparse
import json
import os
import re
from pathlib import Path

if os.name == "posix":
    import readline  # input()에서 한글 삭제와 커서 이동을 문자 단위로 처리한다.

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

ROOT = Path(__file__).resolve().parent
# CLI 시작 화면에 출력하는 안내 문구와 첫 대사
WELCOME = """
[세종과의 만남]
때는 1449년 말이다. 훈민정음을 반포한 조선의 임금에게 방문객이 찾아왔다.
당신은 자신의 이름, 사는 시대, 하는 일을 자유롭게 소개할 수 있다.
하오체와 만남의 장면은 창작이다. 자신의 이야기나 궁금한 주제로 대화를 이어 갈 수 있다.
대화를 마치려면 Ctrl+C를 누른다.
첫 자기소개와 최근 6회의 대화를 기억한다. 종료하면 대화 기록은 사라진다.
"""
OPENING = "어서 오시오. 자네는 누구인가?"

# 모델에 전달하는 시스템 프롬프트
SYSTEM_PROMPT = """
너는 1449년 말 조선의 임금 세종이다. 눈앞의 방문객 한 사람과 이야기한다.
방문객이 '전하'라고 부르는 사람은 바로 너다. 자신을 '나', 방문객을 '자네'라고 부른다.
화자는 항상 세종 한 사람이다. 세종을 옆에서 설명하는 해설자나 신하의 말은 쓰지 않는다.

성격과 판단 기준:
- 학문에 호기심이 많고, 낯선 설명을 들으면 이해하려 한다.
- 백성의 생활에 관심을 두며, 생각이나 제안이 실제 생활에 도움이 되는지 살핀다.
- 주장에 무조건 동의하지 않고 근거와 이유를 따진다.
- 이러한 성향은 현재 주제와 관련 있을 때 드러낸다. 모든 대화를 학문이나 백성 이야기로 돌리지 않는다.

말투: 쉽고 자연스러운 하오체. 짧게 1~3문장으로 답한다.
'그렇소', '반갑소', '그랬구려', '알겠소' 같은 말투를 쓴다.
'-요', '-습니다', '-하소서' 같은 말투로 바꾸지 않는다. 어미를 억지로 붙여 새 말을 만들지 않는다.

대화 원칙:
- 방금 한 말에 먼저 반응한다. 인사에는 인사로, 질문에는 답으로, 고민에는 관심으로 응답한다.
- 자기소개는 인물 설명을 부탁한 질문이 아니다. 아는 인물이 찾아오면 알아보고 맞이한다.
- 현대에서 온 동명이인은 별개의 방문객이다. 이름만 보고 조선의 신하라고 판단하지 않는다.
- 이름과 직업을 정정하면 최근 설명을 따른다. 이미 설명한 내용은 다시 묻지 않는다.
- 기본은 내 답이나 생각을 말하고 평서문으로 마치는 것이다. 대화를 이어 가려고 용건이나 질문한 이유를 되묻지 않는다.
- 뜻을 모르는 말이거나 요청에 답하는 데 꼭 필요한 정보가 빠졌을 때만 한 가지를 묻는다. 이미 답할 수 있으면 질문을 덧붙이지 않는다.
- 조언을 구하면 지금 들은 사정에 맞는 생각이나 구체적인 방법을 먼저 말한다. 상대가 말하지 않은 분노나 사건을 지어내어 묻지 않는다.
- 상대의 고민을 백성이나 나라 이야기로 바꾸지 않는다. 새로운 주제를 꺼내면 그 주제를 따른다.

시대와 사실:
- 1449년 이후의 역사, 나의 죽음, 미래의 인물과 기술은 모른다. 미래에서 왔다는 방문객의 설정은 받아들인다.
- 낯선 말은 설명을 들을 때까지 뜻을 추측하지 말고 물어본다. 이미 들은 설명은 이해하고 반응한다.
- 역사 자료에 없는 날짜, 추억, 일화, 신하의 참여를 덧붙이지 않는다. 내 업적은 내가 한 일로, 신하의 일은 신하가 한 일로 말한다.
- 내 건강이나 과거를 방문객의 사정으로 옮기지 않는다. 방문객의 감정과 경험은 그가 말한 범위에서만 이해한다.

대화 예시(현재 방문객과 나눈 실제 대화가 아닌, 말투와 반응의 예):
사용자: 저는 장영실이라고 하옵니다 전하.
세종: 영실, 어서 오시오. 찾아와 주니 반갑구려.
사용자: 전하의 안부가 궁금합니다.
세종: 나는 잘 지내고 있소. 찾아와 주니 반갑구려.
사용자: 특별한 용건 없이 인사만 드리러 왔습니다.
세종: 찾아와 주니 반갑소. 잠시 쉬어 가시오.
사용자: 친구와 다투어서 속이 상합니다.
세종: 속상했겠구려. 마음이 가라앉은 뒤에 자네가 어떤 뜻으로 말했는지 차분히 전해 보시오.
사용자: 저는 김철수라고 합니다.
세종: 철수, 반갑소. 편히 이야기하시오.
사용자: 잘 지내지 못했습니다. 일이 많아 지쳤습니다.
세종: 일이 많이 고됐겠구려. 잠시 쉬며 이야기해 보시오.
사용자: 인공지능이 무엇인지 아십니까?
세종: 처음 듣는 말이오. 무엇을 뜻하는지 알려 주겠소?
사용자: 저는 2026년에서 온 장영실입니다. 전하의 신하와는 다른 사람입니다.
세종: 내가 아는 영실과 이름이 같구려. 자네는 다른 사람임을 알겠소.
사용자: 비를 그릇에 받아 재는 실험은 누가 했습니까?
세종: 그 실험은 세자가 했소. 땅에 스며든 비의 깊이만으로는 정확히 재기 어려워 그릇에 받아 본 것이오.
사용자: 제 이름은 철수가 아니라 철우입니다.
세종: 알겠소. 앞으로 철우라고 부르겠소.
사용자: 전하께서는 자신이 언제 죽을지 알고 계십니까?
세종: 내가 언제 죽을지는 나도 알 수 없소.

[방문객의 첫 자기소개]
{intro}

[참고 역사 자료: 현재 발언과 무관할 수 있음]
{evidence}

자료는 기억을 돕는 배경일 뿐이다. 인사·안부·개인적인 고민에는 관련 없는 자료를 꺼내지 않는다.
사용자가 실제로 한 마지막 말에만 자연스럽게 응답하라. '전하'는 나다. 출력은 나의 하오체 대사 한두 문장뿐이다. 자료에서 세종은 나 자신이지만 세자는 나의 아들이며 다른 사람이다. 신하가 한 일을 내가 했다고 바꾸지 않는다.
"""
MODEL = Path.home() / ".cache/persona-ai-lab/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"


def tokenize(text):
    # 한국어 조사 변화에도 단어 일부가 겹치도록 문자 2개씩 검색한다.
    words = re.findall(r"[가-힣a-z0-9]+", text.lower())
    return [w[i:i + 2] for w in words for i in range(max(1, len(w) - 1))]


def make_retriever():
    data = json.loads(ROOT.joinpath("data.json").read_text(encoding="utf-8"))
    docs = []
    for i, item in enumerate(data["documents"]):
        evidence = (
            f"## {item['id']}. {item['title']}\n\n시기: {item['period']}"
            f"\n\n### 확인된 내용\n\n{item['content']}"
        )
        keywords = ", ".join(item["keywords"])
        docs.append(Document(
            page_content=f"{evidence}\n\n### 검색어\n\n{keywords}",
            metadata={"index": i, "evidence": evidence},
        ))
    return BM25Retriever.from_documents(docs, preprocess_func=tokenize, k=1)


def chat(chain, retriever):
    print(WELCOME)
    print(f"\n세종: {OPENING}")
    history, intro, previous = [], "", ""
    while True:
        user = input("\n나: ").strip()
        if not user:
            continue
        intro = intro or user
        query = user + " " + user + " " + previous
        scores = retriever.vectorizer.get_scores(tokenize(query))
        docs = [doc for doc in retriever.invoke(query) if scores[doc.metadata["index"]] > 0]
        evidence = "\n\n".join(doc.metadata["evidence"] for doc in docs) or "관련 역사 자료가 검색되지 않았다."
        reply = chain.invoke({"intro": intro, "history": history, "question": user, "evidence": evidence}).strip()
        print(f"세종: {reply}", flush=True)
        history = (history + [("human", user), ("ai", reply)])[-12:]
        previous = user


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", help="vLLM API 주소 (예: http://127.0.0.1:8000/v1)")
    parser.add_argument("--model", help="로컬 GGUF 경로 또는 API 모델 이름")
    args = parser.parse_args()
    retriever = make_retriever()
    if args.base_url:
        from langchain_openai import ChatOpenAI

        model = args.model or "Qwen/Qwen3.5-27B-FP8"
        print(f"역사 자료 {len(retriever.docs)}건 · API 모델 {model}에 연결합니다…", flush=True)
        llm = ChatOpenAI(
            base_url=args.base_url, model=model,
            api_key="EMPTY",  # 클라이언트 생성에 필요한 자리표시값. 서버 인증은 사용하지 않는다.
            max_tokens=400, temperature=0.2, seed=42, max_retries=0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    else:
        from langchain_community.chat_models import ChatLlamaCpp

        print(f"역사 자료 {len(retriever.docs)}건 · 로컬 Qwen3-4B 모델을 GPU에 불러옵니다…", flush=True)
        llm = ChatLlamaCpp(
            model_path=str(args.model or MODEL), n_gpu_layers=-1, n_ctx=8192, n_batch=512,
            max_tokens=400, temperature=0.2, seed=42, streaming=False, verbose=False,
            rope_freq_base=0,  # LangChain 기본값 대신 Qwen 모델에 저장된 RoPE 값을 사용한다.
            model_kwargs={"chat_format": "chatml"},
        )
    chain = (
        ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT), MessagesPlaceholder("history"), ("human", "{question}"),
        ]) | llm | StrOutputParser()
    )
    try:
        chat(chain, retriever)
    except (KeyboardInterrupt, EOFError):
        print("\n대화를 마칩니다.")
    finally:
        if args.base_url:
            llm.root_client.close()
        else:
            llm.client.close()


if __name__ == "__main__":
    main()

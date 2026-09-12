"""1449년 말의 세종과 대화하는 로컬 4B 페르소나 CLI."""

import argparse
import json
import re
from pathlib import Path

from langchain_community.chat_models import ChatLlamaCpp
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

ROOT = Path(__file__).resolve().parent
PROMPTS = json.loads(ROOT.joinpath("prompts.json").read_text(encoding="utf-8"))
WELCOME = "\n".join(PROMPTS["welcome"])
OPENING = PROMPTS["opening"]
PERSONA = "\n".join(PROMPTS["persona"])
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
        sources = "\n".join(f"- [{s['title']}]({s['url']})" for s in item["sources"])
        docs.append(Document(
            page_content=f"{evidence}\n\n### 검색어\n\n{keywords}",
            metadata={"index": i, "evidence": evidence, "sources": sources},
        ))
    return BM25Retriever.from_documents(docs, preprocess_func=tokenize, k=1)


def chat(chain, retriever):
    print(WELCOME)
    print(f"\n세종: {OPENING}")
    history, intro, previous, last_docs = [], "", "", []
    while True:
        user = input("\n나: ").strip()
        if user in ("/종료", "/quit"):
            break
        if not user:
            continue
        if user == "/새대화":
            history, intro, previous, last_docs = [], "", "", []
            print(f"\n세종: {OPENING}")
            continue
        if user == "/근거":
            for doc in last_docs:
                print(f"\n{doc.metadata['evidence']}\n\n{doc.metadata['sources']}")
            if not last_docs:
                print("이번 응답에 검색된 자료가 없습니다.")
            continue
        intro = intro or user
        query = user + " " + user + " " + previous
        scores = retriever.vectorizer.get_scores(tokenize(query))
        last_docs = [doc for doc in retriever.invoke(query) if scores[doc.metadata["index"]] > 0]
        evidence = "\n\n".join(doc.metadata["evidence"] for doc in last_docs) or "관련 역사 자료가 검색되지 않았다."
        reply = chain.invoke({"intro": intro, "history": history, "question": user, "evidence": evidence}).strip()
        print(f"세종: {reply}", flush=True)
        history = (history + [("human", user), ("ai", reply)])[-12:]
        previous = user


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL, help="로컬 Qwen3-4B-Instruct GGUF 경로")
    args = parser.parse_args()
    retriever = make_retriever()
    print(f"역사 자료 {len(retriever.docs)}건 · 로컬 Qwen3-4B 모델을 GPU에 불러옵니다…", flush=True)
    llm = ChatLlamaCpp(
        model_path=str(args.model), n_gpu_layers=-1, n_ctx=8192, n_batch=512,
        max_tokens=400, temperature=0.2, seed=42, streaming=False, verbose=False,
        rope_freq_base=0,  # LangChain 기본값 대신 Qwen 모델에 저장된 RoPE 값을 사용한다.
        model_kwargs={"chat_format": "chatml"},
    )
    chain = (
        ChatPromptTemplate.from_messages([
            ("system", PERSONA), MessagesPlaceholder("history"), ("human", "{question}"),
        ]) | llm | StrOutputParser()
    )
    try:
        chat(chain, retriever)
    except (KeyboardInterrupt, EOFError):
        print("\n대화를 마칩니다.")
    finally:
        llm.client.close()


if __name__ == "__main__":
    main()

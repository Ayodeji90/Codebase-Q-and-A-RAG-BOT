"""Run a code-aware QA flow: retrieve precise code chunks and synthesize an answer
that cites filenames and line ranges. This script is opinionated (temperature=0)
and forces the model to only use retrieved context.

Usage:
  export OPENAI_API_KEY=...
  export WEAVIATE_URL=...
  export WEAVIATE_API_KEY=...
  export WEAVIATE_INDEX_NAME=...
  python -m backend.code_qa "How does ingestion work?"
"""
from __future__ import annotations

import os
import sys
from typing import List

from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
from langchain.schema import Document
from langchain_openai import ChatOpenAI

from backend.code_retriever import detect_intent, make_code_retriever


def format_context(docs: List[Document]) -> str:
    parts = []
    for i, d in enumerate(docs):
        md = d.metadata or {}
        filename = md.get("filename") or md.get("path")
        start = md.get("line_start")
        end = md.get("line_end")
        header = f"### {filename} [{start}-{end}]" if start and end else f"### {filename}"
        parts.append(header)
        parts.append("```\n" + d.page_content + "\n```")
    return "\n\n".join(parts)


PROMPT = """
You are a code assistant. Use ONLY the provided context to answer the question. Do NOT invent facts or speculate.

Instructions:
- Answer concisely and precisely.
- Always cite where your information comes from using the format [filename:start-end] when possible.
- If you can't answer from the provided context, say "I don't know" or ask for more info.

Context:
{context}

Question:
{question}

Answer (include citations inline):
"""


def run_query(question: str):
    intent = detect_intent(question)
    retriever = make_code_retriever(intent=intent)
    docs = retriever.get_relevant_documents(question)

    if not docs:
        print("No relevant context found in the index. Try reindexing or broaden the query.")
        return

    context = format_context(docs)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) if os.environ.get("OPENAI_API_KEY") else ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    prompt = PromptTemplate(input_variables=["context", "question"], template=PROMPT)
    chain = LLMChain(llm=llm, prompt=prompt)
    answer = chain.run({"context": context, "question": question})

    print("\n--- Answer ---\n")
    print(answer)
    print("\n--- Retrieved sources ---\n")
    for d in docs:
        md = d.metadata or {}
        print(f"{md.get('filename')}:{md.get('line_start')}-{md.get('line_end')} -> {md.get('path')}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m backend.code_qa \"Your question\"")
        sys.exit(1)
    question = sys.argv[1]
    run_query(question)

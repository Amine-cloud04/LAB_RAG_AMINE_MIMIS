from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import BaseModel, Field


EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:1b")
VECTORDB_DIR = Path(os.getenv("VECTORDB_DIR", "vectordb"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "ipcc_ar6")

PROMPT = ChatPromptTemplate.from_template(
    "You are a careful climate science assistant. Answer only from the provided "
    "IPCC context. If the answer is not supported by the context, say you do not "
    "know. Keep the answer concise and cite source names and pages when useful.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)

app = FastAPI(title="RAG Ollama IPCC Lab", version="1.0.0")


class QueryIn(BaseModel):
    question: str = Field(..., min_length=3)
    k: int = Field(default=4, ge=1, le=10)


class SourceOut(BaseModel):
    source: str | None = None
    page: int | None = None
    chunk: int | None = None
    preview: str


class QueryOut(BaseModel):
    answer: str
    sources: list[SourceOut]


def get_vectordb() -> Chroma:
    if not VECTORDB_DIR.exists():
        raise HTTPException(
            status_code=503,
            detail="Vector database not found. Run `python ingest.py` and `python embeddings.py` first.",
        )

    return Chroma(
        persist_directory=str(VECTORDB_DIR),
        collection_name=COLLECTION_NAME,
        embedding_function=OllamaEmbeddings(model=EMBED_MODEL),
    )


def format_context(docs) -> str:
    blocks = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown source")
        page = doc.metadata.get("page", "?")
        blocks.append(f"[{source}, page {page}]\n{doc.page_content}")
    return "\n\n".join(blocks)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "embedding_model": EMBED_MODEL,
        "chat_model": CHAT_MODEL,
        "vectordb": str(VECTORDB_DIR),
    }


@app.post("/ask", response_model=QueryOut)
def ask(query: QueryIn) -> QueryOut:
    vectordb = get_vectordb()
    docs = vectordb.similarity_search(query.question, k=query.k)
    if not docs:
        return QueryOut(answer="I do not know.", sources=[])

    llm = ChatOllama(model=CHAT_MODEL, temperature=0.0)
    messages = PROMPT.format_messages(context=format_context(docs), question=query.question)
    response = llm.invoke(messages)

    sources = [
        SourceOut(
            source=doc.metadata.get("source"),
            page=doc.metadata.get("page"),
            chunk=doc.metadata.get("chunk"),
            preview=doc.page_content[:240],
        )
        for doc in docs
    ]
    return QueryOut(answer=response.content, sources=sources)


@app.post("/search", response_model=list[SourceOut])
def search(query: QueryIn) -> list[SourceOut]:
    vectordb = get_vectordb()
    docs = vectordb.similarity_search(query.question, k=query.k)
    return [
        SourceOut(
            source=doc.metadata.get("source"),
            page=doc.metadata.get("page"),
            chunk=doc.metadata.get("chunk"),
            preview=doc.page_content[:240],
        )
        for doc in docs
    ]

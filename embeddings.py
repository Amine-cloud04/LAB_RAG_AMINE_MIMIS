from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


DEFAULT_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def load_chunks(chunks_path: Path) -> list[Document]:
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"{chunks_path} not found. Run `python ingest.py` before embeddings."
        )

    items = json.loads(chunks_path.read_text(encoding="utf-8"))
    return [
        Document(page_content=item["page_content"], metadata=item.get("metadata", {}))
        for item in items
    ]


def embed_and_store(
    chunks_path: Path = Path("chunks/ipcc_ar6_chunks.json"),
    persist_directory: Path = Path("vectordb"),
    collection_name: str = "ipcc_ar6",
    model: str = DEFAULT_EMBED_MODEL,
) -> Chroma:
    documents = load_chunks(chunks_path)
    if not documents:
        raise ValueError("No chunks loaded; cannot build an empty vector database.")

    embeddings = OllamaEmbeddings(model=model)
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(persist_directory),
        collection_name=collection_name,
    )
    print(
        f"Persisted {len(documents)} chunks to {persist_directory} "
        f"using embedding model {model!r}."
    )
    return vectordb


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Chroma vector DB.")
    parser.add_argument("--chunks-path", default=Path("chunks/ipcc_ar6_chunks.json"), type=Path)
    parser.add_argument("--persist-directory", default=Path("vectordb"), type=Path)
    parser.add_argument("--collection-name", default="ipcc_ar6")
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    args = parser.parse_args()

    embed_and_store(
        chunks_path=args.chunks_path,
        persist_directory=args.persist_directory,
        collection_name=args.collection_name,
        model=args.model,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def extract_pdf_pages(pdf_path: Path) -> list[Document]:
    reader = PdfReader(str(pdf_path))
    documents: list[Document] = []

    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={"source": pdf_path.name, "page": page_index},
            )
        )

    return documents


def load_and_split(
    data_dir: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    pdf_paths = sorted(data_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {data_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    for pdf_path in pdf_paths:
        pages = extract_pdf_pages(pdf_path)
        pdf_chunks = splitter.split_documents(pages)
        for chunk_index, chunk in enumerate(pdf_chunks, start=1):
            chunk.metadata["chunk"] = chunk_index
        chunks.extend(pdf_chunks)
        print(f"{pdf_path.name}: {len(pages)} pages -> {len(pdf_chunks)} chunks")

    return chunks


def write_chunks(chunks: list[Document], chunks_dir: Path) -> Path:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    output_path = chunks_dir / "ipcc_ar6_chunks.json"
    payload = [
        {"page_content": chunk.page_content, "metadata": chunk.metadata}
        for chunk in chunks
    ]
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and chunk IPCC AR6 PDFs.")
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--chunks-dir", default="chunks", type=Path)
    parser.add_argument("--chunk-size", default=1000, type=int)
    parser.add_argument("--chunk-overlap", default=200, type=int)
    args = parser.parse_args()

    chunks = load_and_split(args.data_dir, args.chunk_size, args.chunk_overlap)
    output_path = write_chunks(chunks, args.chunks_dir)
    print(f"Saved {len(chunks)} chunks to {output_path}")


if __name__ == "__main__":
    main()

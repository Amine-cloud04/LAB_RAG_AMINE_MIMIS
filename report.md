# RAG Ollama IPCC Lab Report

## Design Choices

The ingest pipeline uses `pypdf` for PDF extraction because it works cleanly on Windows without the extra system dependencies required by `unstructured`. Text is split with LangChain's `RecursiveCharacterTextSplitter`.

- Chunk size: 1000 characters
- Chunk overlap: 200 characters
- Embedding model: `nomic-embed-text`
- Vector store: local persisted Chroma database in `vectordb/`
- Retriever: similarity search with `k=4` by default
- Chat model: `llama3.2:1b` by default, configurable through `OLLAMA_CHAT_MODEL`

## Example Queries To Run

1. What does AR6 say about human influence on observed warming?
2. What are the main risks from continued greenhouse gas emissions?
3. What mitigation pathways are described for limiting warming?

## Evaluation Notes

The API returns source metadata with PDF filename, page number, chunk number, and a text preview. Use these sources to check whether each answer is grounded in the retrieved IPCC context.

Local Ollama model pulls hung on this machine during setup, so the final model outputs should be generated after `ollama pull nomic-embed-text` and `ollama pull llama3.2:1b` complete successfully.

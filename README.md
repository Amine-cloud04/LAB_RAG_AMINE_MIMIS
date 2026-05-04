# RAG Ollama IPCC Lab

Retrieval-Augmented Generation app for IPCC AR6 PDFs using local Ollama models, LangChain, Chroma, FastAPI, and Streamlit.

## Setup

Use the existing `safran` environment:

```powershell
& "C:\Users\lenovo\anaconda3\envs\safran\python.exe" -m pip install -r requirements.txt
```

Install or pull the local Ollama models:

```powershell
ollama pull nomic-embed-text
ollama pull llama3.2:1b
ollama ls
```

The IPCC PDFs are already placed in `data/`.

## Build The Index

```powershell
& "C:\Users\lenovo\anaconda3\envs\safran\python.exe" ingest.py
& "C:\Users\lenovo\anaconda3\envs\safran\python.exe" embeddings.py
```

## Run

Backend:

```powershell
& "C:\Users\lenovo\anaconda3\envs\safran\python.exe" -m uvicorn app:app --reload --port 8000
```

Frontend, in another terminal:

```powershell
& "C:\Users\lenovo\anaconda3\envs\safran\python.exe" -m streamlit run ui_streamlit.py
```

## API

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/ask -ContentType "application/json" -Body '{"question":"What does AR6 say about human influence on warming?","k":4}'
```

## Configuration

The defaults are intentionally small for student laptops:

- `OLLAMA_EMBED_MODEL=nomic-embed-text`
- `OLLAMA_CHAT_MODEL=llama3.2:1b`
- `VECTORDB_DIR=vectordb`
- `CHROMA_COLLECTION=ipcc_ar6`

Set these environment variables before running `embeddings.py` or `app.py` to use different local models.

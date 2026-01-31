# Codebase Q&A RAG Bot

This repository contains a code-focused Retrieval-Augmented Generation (RAG) system: a Codebase Q&A bot that indexes a software repository, retrieves precise code chunks, and synthesizes answers that cite exact files and line ranges.

This README documents the Codebase Q&A workflow, how to run ingestion, verify metadata, run a local QA query, and suggestions for testing and deployment.

## Goals

- Enable developers to ask questions about a codebase and get precise, verifiable answers that reference specific files and line ranges.
- Avoid hallucinations by restricting answers to retrieved code chunks and requiring inline citations.
- Provide a reproducible local workflow built on top of the existing LangChain/Weaviate tooling.

## What this repo provides (Codebase Q&A features)

- `backend/ingest_codebase.py` — ingest a local repository into a Weaviate index using OpenAI embeddings and LangChain's `SQLRecordManager`.
- `backend/code_ingest_utils.py` — read files while preserving formatting, detect language, split into logical chunks (functions/classes/top-level sections), and emit `Document` objects with rich metadata.
- `backend/verify_weaviate_metadata.py` — query the Weaviate index and print sample objects to confirm metadata was saved.
- `backend/code_retriever.py` — retriever factory that performs simple intent detection and returns a Weaviate-backed retriever tuned for precision.
- `backend/code_qa.py` — CLI script that retrieves code chunks for a question and synthesizes answers constrained to retrieved context with inline citations.
- `.env.example` — example environment variables for local development.

## Architecture (focused)

- Ingestion: Read files from a codebase, chunk by logical boundaries, compute embeddings, persist vectors and structured metadata in Weaviate, and register records via the SQLRecordManager.
- Retrieval: Intent-aware retriever that adjusts search parameters (k) based on the question type (file location vs explanation).
- Synthesis: LLM prompt enforces usage of retrieved context only and requires inline citations in the form `[filename:start-end]`.

## Quickstart (local)

Prerequisites

- Python 3.11+
- Node.js + Yarn (if you want to run the frontend UI)
- A Weaviate instance (Weaviate Cloud or self-hosted)
- A Postgres-compatible database (Supabase recommended) for the record manager
- OpenAI API key (or another embedding/LLM provider — scripts use `langchain_openai` by default)

Setup

1. Clone the repo and create a Python venv:

```bash
git clone <this-repo>
cd chat-langchain
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

2. Copy and edit `.env.example` and export your values, or export manually:

```bash
cp .env.example .env
export WEAVIATE_URL="https://<your-weaviate-cluster>"
export WEAVIATE_API_KEY="<your-weaviate-api-key>"
export RECORD_MANAGER_DB_URL="postgresql://user:pass@host:5432/db"
export OPENAI_API_KEY="sk-..."
export CODEBASE_PATH="/path/to/your/repo"  # optional; defaults to repo root
export WEAVIATE_INDEX_NAME="langchain_code"
```

3. Ingest your codebase into Weaviate:

```bash
python -m backend.ingest_codebase
```

What ingestion does

- Scans files in `CODEBASE_PATH` (only allowed extensions; excludes `node_modules`, `venv`, `.git`, `dist`, `build`, `__pycache__`).
- Reads each file preserving indentation and newlines.
- Splits files into logical chunks (functions/classes/top-level) using heuristics in `backend/code_ingest_utils.py`.
- Each chunk is stored as a vector with metadata: `path`, `filename`, `language`, `chunk_type`, `line_start`, `line_end`.

Verify metadata

```bash
export WEAVIATE_INDEX_NAME="langchain_code"
python -m backend.verify_weaviate_metadata
```

This prints a few sample objects and their metadata so you can confirm fields exist and are queryable.

Run a local QA query

```bash
python -m backend.code_qa "Where is ingestion implemented?"
```

- The script will detect the intent (`file`, `how`, `why`, `explain`), retrieve precise chunks using `backend/code_retriever.py`, construct a context with file headers, and call the LLM with a strict prompt requiring evidence-only answers and inline citations.

Design and implementation notes

- Chunking by logical boundaries avoids splitting a function in the middle; however the heuristics are conservative and may need tuning for some languages.
- Metadata is stored as attributes in Weaviate so retrieval can filter and the UI can display file paths and line ranges.
- Retrieval tuning is implemented in `backend/code_retriever.py` via `get_search_kwargs_for_intent` — adjust `k` and other parameters there.
- The QA synth prompt is intentionally strict to minimize hallucination; modify `backend/code_qa.py` `PROMPT` if you need different formatting or more verbose answers.

Frontend (optional, minimal)

- The repo's frontend app is a Next.js app under `frontend/`. To integrate the Codebase Q&A, surface inline citations returned by `code_qa` style responses and make source snippets expandable.
- Minimal change: when rendering an answer, show a list of cited files with clickable links that open the repository at the specified lines (if hosted on GitHub/Gitlab).

Testing & validation suggestions

- Phase 7: Build a small validation suite with a curated set of questions and expected file citations. Run ingestion on a small sample repo and assert that `code_qa` returns citations that reference the expected files.
- Add unit tests for `backend/code_ingest_utils.py` to validate chunk boundaries for sample code snippets.
- Optionally add an integration test that uses a local FAISS index (in-memory) so CI tests don't require Weaviate.

Security & privacy

- The ingestion stores code snippets and metadata in the vector store. Be careful with private keys and sensitive repositories. Use encrypted databases and restrict access to the Weaviate cluster.
- When using OpenAI or other hosted LLMs, be mindful of data sent to the provider.

Next steps (recommended)

1. Phase 6: Frontend integration — show file citations and expose expandable context snippets.
2. Phase 7: Add automated validation and smoke tests that assert citation correctness.
3. Optional: FAISS fallback for fully-local CI runs and developer experiments.

Contact / contributions

If you want help tuning chunking heuristics, retrieval parameters, or the synthesis prompt, open an issue or submit a PR with proposed changes and tests.

License

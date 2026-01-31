# 🦜️🔗 Chat LangChain

This repo is an implementation of a chatbot specifically focused on question answering over the [LangChain documentation](https://python.langchain.com/).
Built with [LangChain](https://github.com/langchain-ai/langchain/), [LangGraph](https://github.com/langchain-ai/langgraph/), and [Next.js](https://nextjs.org).

Deployed version: [chat.langchain.com](https://chat.langchain.com)

> Looking for the JS version? Click [here](https://github.com/langchain-ai/chat-langchainjs).

The app leverages LangChain and LangGraph's streaming support and async API to update the page in real time for multiple users.

## Running locally

This project is now deployed using [LangGraph Cloud](https://langchain-ai.github.io/langgraph/cloud/), which means you won't be able to run it locally (or without a LangGraph Cloud account). If you want to run it WITHOUT LangGraph Cloud, please use the code and documentation from this [branch](https://github.com/langchain-ai/chat-langchain/tree/langserve).

> [!NOTE]
> This [branch](https://github.com/langchain-ai/chat-langchain/tree/langserve) **does not** have the same set of features.

Codebase Q&A (local ingestion)
--------------------------------

This repository now includes tooling to index a local codebase for Codebase Q&A (RAG adapted for source code). The ingestion preserves file structure and formatting, splits by logical code boundaries (functions/classes), and stores rich metadata so answers can cite exact files and line ranges.

Files added
- `.env.example` — example environment variables for local development (Weaviate, Supabase/RecordManager, OpenAI key, etc.).
- `backend/ingest_codebase.py` — script to ingest a local repository (CODEBASE_PATH) into Weaviate using the repo's existing embedding and record manager patterns.
- `backend/code_ingest_utils.py` — utilities that read files preserving indentation, detect language, split files into logical chunks (functions/classes) and emit LangChain `Document` objects with metadata (path, filename, language, chunk_type, line_start, line_end).
- `backend/verify_weaviate_metadata.py` — quick verification utility that queries the Weaviate index to ensure metadata fields were saved and prints sample objects.

Quick usage (local ingestion)
1. Install Python dependencies and the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

2. Copy and fill `.env.example` or export the required environment variables:

Required environment variables for ingestion:
- WEAVIATE_URL — Weaviate cluster URL
- WEAVIATE_API_KEY — Weaviate API key
- RECORD_MANAGER_DB_URL — Postgres/Supabase connection string (for LangChain SQLRecordManager)
- OPENAI_API_KEY — embeddings / LLM calls
- CODEBASE_PATH — path to the repository to index (optional; defaults to repo root)
- WEAVIATE_INDEX_NAME — optional index name

3. Run the ingestion script to index your repository:

```bash
export WEAVIATE_URL="https://<your-weaviate-cluster>"
export WEAVIATE_API_KEY="<your-key>"
export RECORD_MANAGER_DB_URL="postgresql://user:pass@host:port/db"
export OPENAI_API_KEY="sk-..."
export CODEBASE_PATH="/path/to/your/repo"
python -m backend.ingest_codebase
```

Notes on ingestion behavior
- The ingestion script preserves file formatting and stores each file as structured chunks. It detects language from file extensions and splits by logical boundaries (functions, classes, or top-level sections), then further splits large blocks into manageable line ranges.
- Each stored chunk includes metadata fields: `path`, `filename`, `language`, `chunk_type`, `line_start`, `line_end`, and the text content. These fields are stored as Weaviate attributes and are queryable.

Verify metadata in Weaviate
--------------------------------
After ingestion, you can verify that metadata fields were persisted with the included verification script:

```bash
export WEAVIATE_INDEX_NAME="langchain"  # or your index name
python -m backend.verify_weaviate_metadata
```

The verifier prints sample objects returned from Weaviate; each object should show the fields listed above and a `text` field with the chunk content.

Why this helps
- Chunking by logical code boundaries (not token counts) reduces the risk of cutting code mid-function.
- Rich metadata allows the retriever to filter and the synthesizer to cite exact files and line ranges in responses.

Next steps (recommended)
- Retrieval tuning: reduce `k` and prefer precision for code queries; implement intent-aware retriever strategies (file-level vs explanation queries).
- Synthesis constraints: create prompts that require answers to reference file names and line ranges and to avoid unsupported speculation.
- UI: expose file path and a link to the source in response cards so users can verify answers quickly.

Phase 4 & 5 — Retrieval tuning and answer synthesis (what we added)
-----------------------------------------------------------------
We've implemented Phase 4 (retrieval tuning) and Phase 5 (answer synthesis constraints) with two new modules:

- `backend/code_retriever.py` — a retriever factory tuned for code queries. It performs simple intent detection (file / how / why / explain) and returns a Weaviate-backed retriever with search kwargs optimized for precision (smaller `k` for file-location queries, slightly larger for explanatory queries).

- `backend/code_qa.py` — a small CLI QA runner that:
	- detects intent from the user question,
	- retrieves precise code chunks from Weaviate,
	- synthesizes an answer using an LLM with a strict prompt that requires the model to only use the retrieved context and to cite filenames and line ranges.

Quick usage (retrieval & QA)
1. Ensure environment variables are set (Weaviate + OpenAI):

```bash
export WEAVIATE_URL="https://<your-weaviate-cluster>"
export WEAVIATE_API_KEY="<your-key>"
export WEAVIATE_INDEX_NAME="langchain"
export OPENAI_API_KEY="sk-..."
```

2. Run a sample query using the CLI QA script (one-liner question):

```bash
python -m backend.code_qa "Where is ingestion implemented?"
```

What the QA script does
- Uses `code_retriever.detect_intent()` to classify the question into intents like `file`, `how`, `why`, or `explain`.
- Builds search kwargs tuned for that intent (for example, `k=3` for `file` queries to favor precision).
- Retrieves the top chunks and constructs a context with explicit headers `### filename [start-end]` followed by the chunk content.
- Sends the context + question to the LLM with a prompt that instructs the model to only answer from the context and to cite the source(s) inline as `[filename:start-end]`.

Why this reduces hallucinations
- By reducing `k` and enforcing a prompt that forbids adding external facts, we narrow the model's knowledge to retrieved evidence and require citations.
- Intent-aware retrieval adapts precision vs. coverage: `file` queries return fewer, highly-relevant chunks; `how`/`why` queries allow more chunks so explanatory connections can be made.

Customization points
- Tuning `k` and other Weaviate search parameters in `backend/code_retriever.py` (function `get_search_kwargs_for_intent`) directly affects precision vs recall.
- Prompt template in `backend/code_qa.py` (variable `PROMPT`) is intentionally strict; you can tweak wording to suit your team's style or to require different citation formats.

Next recommended steps after Phase 4 & 5
- Frontend: show inline citations and make source code expandable (Phase 6).
- Validation: build a question set and automated tests that assert the model cites the correct files (Phase 7).


If you need a fully self-hosted experience without Weaviate, consider adding a FAISS/Chroma fallback index and a small FastAPI wrapper to serve queries locally (the `langserve` branch is another option with fewer features).

## 📚 Technical description

There are two components: ingestion and question-answering.

Ingestion has the following steps:

1. Pull html from documentation site as well as the Github Codebase
2. Load html with LangChain's [RecursiveURLLoader](https://python.langchain.com/docs/integrations/document_loaders/recursive_url) and [SitemapLoader](https://python.langchain.com/docs/integrations/document_loaders/sitemap)
3. Split documents with LangChain's [RecursiveCharacterTextSplitter](https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html)
4. Create a vectorstore of embeddings, using LangChain's [Weaviate vectorstore wrapper](https://python.langchain.com/docs/integrations/vectorstores/weaviate) (with OpenAI's embeddings).

Question-Answering has the following steps:

1. Given the chat history and new user input, determine what a standalone question would be using an LLM.
2. Given that standalone question, look up relevant documents from the vectorstore.
3. Pass the standalone question and relevant documents to the model to generate and stream the final answer.
4. Generate a trace URL for the current chat session, as well as the endpoint to collect feedback.

## Documentation

Looking to use or modify this Use Case Accelerant for your own needs? We've added a few docs to aid with this:

- **[Concepts](./CONCEPTS.md)**: A conceptual overview of the different components of Chat LangChain. Goes over features like ingestion, vector stores, query analysis, etc.
- **[Modify](./MODIFY.md)**: A guide on how to modify Chat LangChain for your own needs. Covers the frontend, backend and everything in between.
- **[LangSmith](./LANGSMITH.md)**: A guide on adding robustness to your application using LangSmith. Covers observability, evaluations, and feedback.
- **[Production](./PRODUCTION.md)**: Documentation on preparing your application for production usage. Explains different security considerations, and more.
- **[Deployment](./DEPLOYMENT.md)**: How to deploy your application to production. Covers setting up production databases, deploying the frontend, and more.

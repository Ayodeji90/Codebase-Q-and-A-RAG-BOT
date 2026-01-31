"""Ingest a local codebase (directory) into Weaviate for code Q&A.

This script mirrors the pattern used in `backend/ingest.py` but reads files
from a local directory (codebase) instead of from sitemaps. It splits files,
computes embeddings (OpenAI) and writes to the configured Weaviate index.

Usage:
  export WEAVIATE_URL=...
  export WEAVIATE_API_KEY=...
  export RECORD_MANAGER_DB_URL=...
  export OPENAI_API_KEY=...
  export CODEBASE_PATH=/path/to/your/repo   # defaults to repo root
  python -m backend.ingest_codebase
"""

import logging
import os
from pathlib import Path
from typing import List

import weaviate
from langchain.indexes import SQLRecordManager, index
from langchain.document_loaders import TextLoader
from langchain_weaviate import WeaviateVectorStore

from backend.embeddings import get_embeddings_model
from backend.constants import WEAVIATE_DOCS_INDEX_NAME
from backend.code_ingest_utils import create_documents_from_file
from typing import Iterator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


WEAVIATE_URL = os.environ.get("WEAVIATE_URL")
WEAVIATE_API_KEY = os.environ.get("WEAVIATE_API_KEY")
RECORD_MANAGER_DB_URL = os.environ.get("RECORD_MANAGER_DB_URL")
CODEBASE_PATH = os.environ.get("CODEBASE_PATH", ".")



def iter_files(path: str) -> Iterator[str]:
    """Yield file paths to include for ingestion.

    Inclusion is based on file extensions and we exclude common dependency/build folders.
    """
    include_exts = {
        ".py",
        ".pyi",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".json",
        ".md",
        ".toml",
        ".yaml",
        ".yml",
        ".cfg",
        ".ini",
        ".txt",
    }
    exclude_dirs = {"node_modules", "venv", ".venv", "dist", "build", "__pycache__", ".git"}

    p = Path(path)
    for fp in p.rglob("*"):
        if fp.is_file():
            if any(part in exclude_dirs for part in fp.parts):
                continue
            if fp.suffix.lower() in include_exts:
                yield str(fp)


def ingest_codebase():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embedding = get_embeddings_model()

    with weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=weaviate.classes.init.Auth.api_key(WEAVIATE_API_KEY),
        skip_init_checks=True,
    ) as weaviate_client:
        index_name = os.environ.get("WEAVIATE_INDEX_NAME", WEAVIATE_DOCS_INDEX_NAME)
        vectorstore = WeaviateVectorStore(
            client=weaviate_client,
            index_name=index_name,
            text_key="text",
            embedding=embedding,
            # store rich metadata so retrieval can filter and cite exact locations
            attributes=[
                "source",
                "path",
                "filename",
                "language",
                "chunk_type",
                "line_start",
                "line_end",
            ],
        )

        record_manager = SQLRecordManager(f"weaviate/{index_name}", db_url=RECORD_MANAGER_DB_URL)
        record_manager.create_schema()

        logger.info(f"Scanning files from {CODEBASE_PATH}...")
        docs_transformed = []
        file_count = 0
        for file_path in iter_files(CODEBASE_PATH):
            file_count += 1
            file_docs = create_documents_from_file(file_path)
            # ensure a 'source' metadata for record manager dedupe
            for d in file_docs:
                if "source" not in d.metadata:
                    d.metadata["source"] = d.metadata.get("path", file_path)
            docs_transformed.extend(file_docs)

        logger.info(f"Found {file_count} files and produced {len(docs_transformed)} chunks")
        docs_transformed = [d for d in docs_transformed if len(d.page_content) > 10]

        # Ensure minimal metadata for indexing
        for doc in docs_transformed:
            if "source" not in doc.metadata:
                doc.metadata["source"] = str(doc.metadata.get("source", ""))
            # store file path if present
            if "path" not in doc.metadata:
                # DirectoryLoader/TextLoader usually sets `source` to filepath
                doc.metadata["path"] = doc.metadata.get("source", "")

        logger.info("Indexing into Weaviate...")
        indexing_stats = index(
            docs_transformed,
            record_manager,
            vectorstore,
            cleanup="full",
            source_id_key="source",
            force_update=(os.environ.get("FORCE_UPDATE") or "false").lower() == "true",
        )
        logger.info(f"Indexing stats: {indexing_stats}")


if __name__ == "__main__":
    missing = []
    for env in ("WEAVIATE_URL", "WEAVIATE_API_KEY", "RECORD_MANAGER_DB_URL", "OPENAI_API_KEY"):
        if not os.environ.get(env):
            missing.append(env)
    if missing:
        logger.warning(
            "Missing environment variables: %s. Check .env.example or export them before running.",
            ", ".join(missing),
        )

    ingest_codebase()

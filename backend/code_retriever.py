"""Retriever factory for code Q&A with simple intent biasing and precision-focused search kwargs."""
from __future__ import annotations

import os
import re
from contextlib import contextmanager
from typing import Iterator

import weaviate
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_weaviate import WeaviateVectorStore

from backend.embeddings import get_embeddings_model
from backend.configuration import BaseConfiguration
from backend.constants import WEAVIATE_DOCS_INDEX_NAME


def detect_intent(query: str) -> str:
    q = query.strip().lower()
    if re.match(r"^(where|which file|what file|where is)", q):
        return "file"
    if re.match(r"^(how|how do|how can|how to)", q):
        return "how"
    if re.match(r"^(why|explain why|what is the reason)", q):
        return "why"
    return "explain"


@contextmanager
def make_weaviate_client() -> Iterator[weaviate.Client]:
    url = os.environ.get("WEAVIATE_URL")
    api_key = os.environ.get("WEAVIATE_API_KEY")
    if not url or not api_key:
        raise RuntimeError("WEAVIATE_URL and WEAVIATE_API_KEY must be set in environment")

    with weaviate.connect_to_weaviate_cloud(
        cluster_url=url,
        auth_credentials=weaviate.classes.init.Auth.api_key(api_key),
        skip_init_checks=True,
    ) as client:
        yield client


def get_search_kwargs_for_intent(intent: str) -> dict:
    """Return search kwargs tuned for intent.

    - file: very precise, low k
    - how/why/explain: slightly higher k to gather explanatory context
    """
    if intent == "file":
        return {"k": 3, "return_uuids": True}
    if intent in ("how", "why"):
        return {"k": 6, "return_uuids": True}
    return {"k": 5, "return_uuids": True}


def make_code_retriever(index_name: str | None = None, intent: str | None = None) -> BaseRetriever:
    """Create a Weaviate-backed retriever tuned for code queries.

    Args:
        index_name: optional override for the weaviate index name
        intent: optional; if not provided it will be detected from query text by caller
    Returns:
        BaseRetriever
    """
    index = index_name or os.environ.get("WEAVIATE_INDEX_NAME") or WEAVIATE_DOCS_INDEX_NAME
    embedding_model = get_embeddings_model()

    # we'll create and return a retriever object configured with search_kwargs
    client_ctx = make_weaviate_client()
    client = client_ctx.__enter__()

    store = WeaviateVectorStore(
        client=client,
        index_name=index,
        text_key="text",
        embedding=embedding_model,
        attributes=["source", "path", "filename", "language", "chunk_type", "line_start", "line_end"],
    )

    if intent is None:
        search_kwargs = get_search_kwargs_for_intent("explain")
    else:
        search_kwargs = get_search_kwargs_for_intent(intent)

    retriever = store.as_retriever(search_kwargs=search_kwargs)

    # We do not close the weaviate client here because the retriever may use it; rely on process exit.
    return retriever

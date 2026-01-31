"""Simple verification utility to confirm metadata fields exist in the Weaviate index.

This script connects to the configured Weaviate instance and runs a small
GraphQL GET query for the configured index name, requesting the expected
metadata fields. It prints the first few results to stdout.

Usage:
  export WEAVIATE_URL=...
  export WEAVIATE_API_KEY=...
  export WEAVIATE_INDEX_NAME=langchain
  python -m backend.verify_weaviate_metadata
"""

import os
import logging
import json

import weaviate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    url = os.environ.get("WEAVIATE_URL")
    api_key = os.environ.get("WEAVIATE_API_KEY")
    index_name = os.environ.get("WEAVIATE_INDEX_NAME")

    if not url or not api_key or not index_name:
        logger.error(
            "Please set WEAVIATE_URL, WEAVIATE_API_KEY and WEAVIATE_INDEX_NAME environment variables."
        )
        return

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=url, auth_credentials=weaviate.classes.init.Auth.api_key(api_key), skip_init_checks=True
    )

    # Fields we expect to be present on the objects
    fields = [
        "source",
        "path",
        "filename",
        "language",
        "chunk_type",
        "line_start",
        "line_end",
        "text",
    ]

    try:
        # Use the GraphQL GET API to fetch a few objects
        res = client.query.get(index_name, fields).with_limit(5).do()
    except Exception as e:
        logger.error("Query failed: %s", e)
        return

    # print nicely
    hits = res.get("data", {}).get(index_name, [])
    if not hits:
        logger.info("No objects returned. Is the index populated?")
        # attempt a fallback to aggregate count
        try:
            count = client.collections.get(index_name).aggregate.over_all().total_count
            logger.info("Index %s has %s objects", index_name, count)
        except Exception:
            pass
        return

    print("Sample objects and their metadata:\n")
    for obj in hits:
        # each obj is a dict of requested fields
        print(json.dumps(obj, indent=2, ensure_ascii=False))
        print("---")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Upload the chunked evidence corpus to a Hugging Face dataset repo."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from huggingface_hub import HfApi


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = (
    PROJECT_ROOT / "data" / "vie" / "processed" / "corpus_v1_chunked.json"
)
DEFAULT_REPO_ID = "Loctran123/vietnamese-evidence-corpus-chunked"
DEFAULT_PATH_IN_REPO = "data/corpus_v1_chunked.json"
ENV_FILE = PROJECT_ROOT / ".env"
TOKEN_VARIABLE = "HUGGING_FACE_HUB_TOKEN"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
LOGGER = logging.getLogger("hf_chunked_uploader")


def load_hf_token() -> str | None:
    """Load a Hugging Face write token without logging its value."""

    token = os.getenv(TOKEN_VARIABLE)
    if token:
        return token.strip()

    if not ENV_FILE.is_file():
        return None

    for line in ENV_FILE.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() == TOKEN_VARIABLE:
            token = value.strip().strip('"').strip("'")
            if token:
                return token
    return None


def dataset_card(repo_id: str, path_in_repo: str) -> str:
    """Return a compact dataset card for the uploaded chunked corpus."""

    return f"""---
language:
- vi
- en
license: other
size_categories:
- 10K<n<100K
task_categories:
- text-retrieval
- question-answering
tags:
- fact-checking
- rag
- retrieval
- bge-m3
- vietnamese
pretty_name: Vietnamese Evidence Corpus - Chunked
---

# Vietnamese Evidence Corpus - Chunked

Chunked evidence corpus prepared for multilingual information retrieval,
retrieval-augmented generation, and fact-checking experiments.

## Statistics

- 47,679 chunks from 13,572 source documents
- 38,603 Vietnamese chunks and 9,076 English chunks
- Maximum chunk length: 512 BGE-M3 tokenizer tokens

## Main fields

- `chunk_id`, `doc_id`, `chunk_index`
- `token_start`, `token_end`, `token_count`
- `title`, `text`, `summary`
- `source`, `source_type`, `domain`, `document_type`
- `language`, `country`, `publish_date`, `crawl_date`, `url`, `author`
- `metadata`, `quality`

For embedding, use `title + "\\n\\n" + text` as the embedding input and keep
the remaining fields as filter/citation metadata.

## Loading

```python
from datasets import load_dataset

dataset = load_dataset(
    "{repo_id}",
    data_files="{path_in_repo}",
    split="train",
)
```

The repository does not grant additional rights over the original source
articles. Users are responsible for following the terms of each source.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload the chunked evidence corpus to Hugging Face."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--path-in-repo", default=DEFAULT_PATH_IN_REPO)
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private dataset repository (default: public).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    corpus_path = args.corpus.resolve()
    if not corpus_path.is_file():
        LOGGER.error("Corpus file not found: %s", corpus_path)
        return 2

    token = load_hf_token()
    if not token:
        LOGGER.error(
            "%s was not found in the environment or %s",
            TOKEN_VARIABLE,
            ENV_FILE,
        )
        return 2

    api = HfApi(token=token)
    try:
        account = api.whoami()
        LOGGER.info("Authenticated as %s", account.get("name", "<unknown>"))
        repo_url = api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )
        LOGGER.info("Dataset repository ready: %s", repo_url)

        LOGGER.info(
            "Uploading %s (%.2f MiB) as %s",
            corpus_path,
            corpus_path.stat().st_size / (1024 * 1024),
            args.path_in_repo,
        )
        api.upload_file(
            path_or_fileobj=corpus_path,
            path_in_repo=args.path_in_repo,
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message="Upload chunked evidence corpus",
        )

        api.upload_file(
            path_or_fileobj=dataset_card(
                args.repo_id, args.path_in_repo
            ).encode("utf-8"),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message="Add dataset card",
        )
    except Exception:
        LOGGER.exception("Hugging Face upload failed")
        return 1

    LOGGER.info("Upload complete: https://huggingface.co/datasets/%s", args.repo_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""Build reproducible FAISS and BM25S indexes from the E5 embedding dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import bm25s
import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from tqdm.auto import tqdm


DEFAULT_EMBEDDING_REPO = os.getenv(
    "HF_EMBEDDING_REPO", "Loctran123/vietnamese-evidence-corpus-embeddings-e5-large-v2-r1"
)
DEFAULT_OUTPUT_REPO = os.getenv(
    "HF_INDEX_REPO", "Loctran123/vietnamese-evidence-retrieval-indexes-v2-r1"
)
BM25_TOKEN_PATTERN = r"(?u)\b\w+\b"
EMBEDDING_DIMENSION = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-repo", default=DEFAULT_EMBEDDING_REPO)
    parser.add_argument("--embedding-revision", default=None)
    parser.add_argument("--output-repo", default=DEFAULT_OUTPUT_REPO)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/retrieval_indexes"),
    )
    parser.add_argument("--title-repetitions", type=int, default=2)
    parser.add_argument("--private-output", action="store_true")
    return parser.parse_args()


def get_hf_token() -> str:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient

            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            token = None
    if not token:
        raise RuntimeError("Missing Kaggle secret/environment variable HF_TOKEN.")
    return token


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def build_config(args: argparse.Namespace, embedding_revision: str) -> dict[str, Any]:
    return {
        "embedding_repo": args.embedding_repo,
        "embedding_revision": embedding_revision,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "faiss_index": "IndexFlatIP",
        "similarity": "inner_product_on_l2_normalized_vectors",
        "bm25_library": "bm25s",
        "bm25_method": "lucene",
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "bm25_token_pattern": BM25_TOKEN_PATTERN,
        "bm25_stopwords": None,
        "bm25_stemmer": None,
        "deduplicate_by_content_hash": True,
        "dense_input": "title_plus_text",
        "title_repetitions": args.title_repetitions,
    }


def ensure_remote_config(
    api: HfApi,
    args: argparse.Namespace,
    token: str,
    config: dict[str, Any],
) -> tuple[bool, set[str]]:
    api.create_repo(
        repo_id=args.output_repo,
        repo_type="dataset",
        private=args.private_output,
        exist_ok=True,
        token=token,
    )
    files = set(api.list_repo_files(args.output_repo, repo_type="dataset", token=token))
    if "index_config.json" in files:
        remote_path = hf_hub_download(
            repo_id=args.output_repo,
            filename="index_config.json",
            repo_type="dataset",
            token=token,
        )
        existing = json.loads(Path(remote_path).read_text(encoding="utf-8"))
        if existing != config:
            raise RuntimeError(
                "The output repository is locked to another index configuration. "
                "Use a new --output-repo or restore the original settings.\n"
                f"Existing: {json.dumps(existing, ensure_ascii=False, indent=2)}\n"
                f"Current:  {json.dumps(config, ensure_ascii=False, indent=2)}"
            )
    else:
        payload = json.dumps(config, ensure_ascii=False, indent=2).encode("utf-8")
        api.upload_file(
            path_or_fileobj=payload,
            path_in_repo="index_config.json",
            repo_id=args.output_repo,
            repo_type="dataset",
            commit_message="Lock retrieval index configuration",
            token=token,
        )
    return "manifest.json" in files, files


def download_complete_index(args: argparse.Namespace, token: str) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.output_repo,
        repo_type="dataset",
        token=token,
        local_dir=str(args.output_dir),
        allow_patterns=[
            "faiss.index",
            "metadata.parquet",
            "bm25/*",
            "manifest.json",
            "index_config.json",
            "README.md",
        ],
    )
    print(f"Downloaded existing index to {args.output_dir}")
    return args.output_dir


def embedding_values(table: pa.Table) -> np.ndarray:
    column = table.column("embedding").combine_chunks()
    if not pa.types.is_fixed_size_list(column.type) or column.type.list_size != EMBEDDING_DIMENSION:
        raise RuntimeError(f"Unexpected embedding type: {column.type}")
    values = column.values.to_numpy(zero_copy_only=False)
    matrix = values.reshape(len(column), EMBEDDING_DIMENSION).astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, atol=2e-3):
        raise RuntimeError(
            f"Input vectors are not normalized: min_norm={norms.min():.6f}, max_norm={norms.max():.6f}"
        )
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    faiss.normalize_L2(matrix)
    return matrix


def build_faiss_and_metadata(
    args: argparse.Namespace,
    token: str,
    embedding_revision: str,
) -> tuple[faiss.Index, Path, list[str]]:
    api = HfApi(token=token)
    shard_names = sorted(
        name
        for name in api.list_repo_files(
            args.embedding_repo,
            repo_type="dataset",
            revision=embedding_revision,
            token=token,
        )
        if name.startswith("data/train-") and name.endswith(".parquet")
    )
    if not shard_names:
        raise RuntimeError(f"No embedding Parquet shards found in {args.embedding_repo}.")

    index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    metadata_path = args.output_dir / "metadata.parquet"
    seen_chunk_ids: set[str] = set()
    hash_to_row_id: dict[str, int] = {}
    canonical_records: list[dict[str, Any]] = []
    canonical_vectors: list[np.ndarray] = []

    for shard_name in tqdm(shard_names, desc="FAISS + metadata", unit="shard"):
        shard_path = hf_hub_download(
            repo_id=args.embedding_repo,
            filename=shard_name,
            repo_type="dataset",
            revision=embedding_revision,
            token=token,
        )
        table = pq.read_table(shard_path)
        required = {"chunk_id", "doc_id", "text", "embedding"}
        missing = required - set(table.column_names)
        if missing:
            raise RuntimeError(f"{shard_name} is missing columns: {sorted(missing)}")

        vectors = embedding_values(table)
        rows = table.to_pylist()
        if len(rows) != len(vectors):
            raise RuntimeError(f"{shard_name} rows ({len(rows)}) != vectors ({len(vectors)})")

        for record, vector in zip(rows, vectors):
            chunk_id = str(record.get("chunk_id", "")).strip()
            if chunk_id in seen_chunk_ids:
                raise RuntimeError(f"Duplicate chunk_id detected: {chunk_id}")
            seen_chunk_ids.add(chunk_id)

            record = dict(record)
            record.pop("embedding", None)
            title = str(record.get("title", "")).strip()
            text = str(record.get("text", "")).strip()
            content_key = str(record.get("content_hash") or " ".join(f"{title}\n\n{text}".split()).casefold())
            provenance_doc_id = str(record.get("doc_id", "")).strip()
            provenance_url = record.get("url")

            if content_key in hash_to_row_id:
                canonical = canonical_records[hash_to_row_id[content_key]]
                canonical["duplicate_chunk_ids"].append(chunk_id)
                if provenance_doc_id:
                    canonical["provenance_doc_ids"].append(provenance_doc_id)
                if provenance_url:
                    canonical["provenance_urls"].append(provenance_url)
                continue

            record["content_hash"] = content_key
            record["provenance_doc_ids"] = [provenance_doc_id] if provenance_doc_id else []
            record["provenance_urls"] = [provenance_url] if provenance_url else []
            record["duplicate_chunk_ids"] = []
            hash_to_row_id[content_key] = len(canonical_records)
            canonical_records.append(record)
            canonical_vectors.append(vector)

    if not canonical_records:
        raise RuntimeError("No canonical rows were produced after deduplication.")

    vectors = np.ascontiguousarray(np.vstack(canonical_vectors), dtype=np.float32)
    faiss.normalize_L2(vectors)
    index.add(vectors)

    metadata = pa.Table.from_pylist(canonical_records)
    row_ids = pa.array(range(len(canonical_records)), type=pa.int64())
    metadata = metadata.add_column(0, "row_id", row_ids)
    writer = pq.ParquetWriter(
        metadata_path,
        metadata.schema,
        compression="zstd",
        compression_level=6,
    )
    try:
        writer.write_table(metadata, row_group_size=min(2_000, len(metadata)))
    finally:
        writer.close()

    if index.ntotal != len(canonical_records):
        raise RuntimeError(f"FAISS rows ({index.ntotal}) != metadata rows ({len(canonical_records)}).")
    faiss.write_index(index, str(args.output_dir / "faiss.index"))
    return index, metadata_path, shard_names

def bm25_text(title: str | None, text: str | None, title_repetitions: int) -> str:
    title = (title or "").strip()
    text = (text or "").strip()
    title_part = " ".join([title] * title_repetitions)
    return f"{title_part}\n{text}".strip()


def build_bm25(metadata_path: Path, output_dir: Path, title_repetitions: int) -> None:
    columns = pq.read_table(metadata_path, columns=["title", "text"])
    titles = columns.column("title").to_pylist()
    texts = columns.column("text").to_pylist()
    corpus = [
        bm25_text(title, text, title_repetitions)
        for title, text in tqdm(zip(titles, texts), total=len(texts), desc="BM25 corpus")
    ]
    corpus_tokens = bm25s.tokenize(
        corpus,
        stopwords=None,
        stemmer=None,
        token_pattern=BM25_TOKEN_PATTERN,
        return_ids=False,
        show_progress=True,
    )
    retriever = bm25s.BM25(method="lucene", k1=1.5, b=0.75)
    retriever.index(corpus_tokens, show_progress=True)
    bm25_dir = output_dir / "bm25"
    bm25_dir.mkdir(parents=True, exist_ok=True)
    retriever.save(str(bm25_dir))


def dataset_card(config: dict[str, Any], rows: int, shards: int) -> str:
    return f"""---
language:
- vi
tags:
- information-retrieval
- faiss
- bm25
---

# Vietnamese Evidence Retrieval Indexes

Prebuilt exact dense and sparse indexes for
`{config['embedding_repo']}` at revision `{config['embedding_revision']}`.

- Rows: {rows:,}
- Source embedding shards: {shards}
- Dense: FAISS `IndexFlatIP`, 1024 dimensions
- Sparse: BM25S Lucene BM25 (`k1=1.5`, `b=0.75`)
- BM25 content: title repeated {config['title_repetitions']} times + chunk text
- Dense input: title + text
- Dense rows: deduplicated by content hash
- Vietnamese tokenization: Unicode word tokens, no stemming and no stopword removal

`row_id` in `metadata.parquet` is the shared positional identifier for both
indexes. Use the project `retrieval/hybrid_retriever.py` to query safely.
"""


def write_manifest(
    output_dir: Path,
    config: dict[str, Any],
    rows: int,
    shard_names: list[str],
) -> dict[str, Any]:
    tracked_files = [output_dir / "faiss.index", output_dir / "metadata.parquet"]
    tracked_files.extend(sorted((output_dir / "bm25").glob("*")))
    manifest = {
        "status": "complete",
        "rows": rows,
        "source_embedding_shards": shard_names,
        "config": config,
        "files": {
            str(path.relative_to(output_dir)).replace("\\", "/"): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in tracked_files
            if path.is_file()
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "index_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        dataset_card(config, rows=rows, shards=len(shard_names)),
        encoding="utf-8",
    )
    return manifest


def upload_index(api: HfApi, args: argparse.Namespace, token: str) -> None:
    api.upload_folder(
        folder_path=str(args.output_dir),
        repo_id=args.output_repo,
        repo_type="dataset",
        ignore_patterns=["manifest.json"],
        commit_message="Upload FAISS and BM25 retrieval indexes",
        token=token,
    )
    api.upload_file(
        path_or_fileobj=str(args.output_dir / "manifest.json"),
        path_in_repo="manifest.json",
        repo_id=args.output_repo,
        repo_type="dataset",
        commit_message="Mark retrieval indexes complete",
        token=token,
    )


def main() -> None:
    args = parse_args()
    if args.title_repetitions < 0:
        raise ValueError("--title-repetitions must be non-negative.")
    token = get_hf_token()
    api = HfApi(token=token)
    info = api.dataset_info(
        args.embedding_repo,
        revision=args.embedding_revision,
        token=token,
    )
    embedding_revision = info.sha
    config = build_config(args, embedding_revision)
    complete, _ = ensure_remote_config(api, args, token, config)
    if complete:
        print(f"Remote index is already complete: https://huggingface.co/datasets/{args.output_repo}")
        download_complete_index(args, token)
        return

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(
            f"{args.output_dir} is not empty and the remote index is incomplete. "
            "Use a fresh output directory or restart the Kaggle session."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "index_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    index, metadata_path, shard_names = build_faiss_and_metadata(
        args,
        token=token,
        embedding_revision=embedding_revision,
    )
    build_bm25(metadata_path, args.output_dir, args.title_repetitions)
    write_manifest(args.output_dir, config, rows=index.ntotal, shard_names=shard_names)
    upload_index(api, args, token)
    print(f"Complete: {index.ntotal:,} chunks indexed.")
    print(f"https://huggingface.co/datasets/{args.output_repo}")


if __name__ == "__main__":
    main()

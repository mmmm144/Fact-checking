"""Create normalized multilingual-e5-large passage embeddings on Kaggle.

The script streams a chunked corpus from Hugging Face (or a local JSON file),
writes deterministic Parquet shards, and optionally uploads each completed shard
to a separate Hugging Face dataset repository. Re-running the same command
resumes from the first missing remote shard.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

# Avoid importing an incompatible preinstalled Keras/TensorFlow stack on Kaggle.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


DEFAULT_SOURCE_REPO = os.getenv(
    "HF_CHUNK_REPO", "Loctran123/vietnamese-evidence-corpus-chunked-e5-v2"
)
DEFAULT_OUTPUT_REPO = os.getenv(
    "HF_EMBEDDING_REPO", "Loctran123/vietnamese-evidence-corpus-embeddings-e5-large-v2-r1"
)
DEFAULT_MODEL = "intfloat/multilingual-e5-large"
DEFAULT_COLUMNS = (
    "chunk_id",
    "doc_id",
    "chunk_index",
    "title",
    "text",
    "source",
    "source_type",
    "domain",
    "document_type",
    "language",
    "country",
    "publish_date",
    "url",
    "content_hash",
)
SHARD_RE = re.compile(r"^data/train-(\d{5})\.parquet$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", default=DEFAULT_SOURCE_REPO)
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--source-revision", default=None, help="Optional immutable HF commit SHA.")
    parser.add_argument(
        "--source-json",
        type=Path,
        help="Optional local JSON/JSONL input; overrides --source-repo.",
    )
    parser.add_argument("--output-repo", default=DEFAULT_OUTPUT_REPO)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/e5_embeddings"))
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--prefix", default="passage: ")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--rows-per-shard", type=int, default=5_000)
    parser.add_argument("--max-rows", type=int, default=None, help="Useful for a smoke test.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-upload", action="store_true", help="Only write local Parquet files.")
    parser.add_argument(
        "--private-output",
        action="store_true",
        help="Create the output Hugging Face dataset as private if it does not exist.",
    )
    return parser.parse_args()


def get_hf_token(required: bool) -> str | None:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient

            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            token = None
    if required and not token:
        raise RuntimeError(
            "Missing HF_TOKEN. Add a Kaggle secret named HF_TOKEN with write access "
            "to the output dataset, or run with --no-upload."
        )
    return token


def validate_args(args: argparse.Namespace) -> None:
    if args.rows_per_shard <= 0 or args.batch_size <= 0:
        raise ValueError("--rows-per-shard and --batch-size must be positive.")
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be positive when provided.")
    if args.max_rows is not None and not args.no_upload:
        raise ValueError("--max-rows is a smoke-test option and requires --no-upload.")
    if not args.no_upload and args.source_repo == args.output_repo:
        raise ValueError(
            "Refusing to write embeddings into the source chunk dataset. "
            "Choose a separate --output-repo."
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Enable a GPU accelerator in Kaggle.")


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "source": str(args.source_json) if args.source_json else args.source_repo,
        "source_split": args.source_split,
        "source_revision": args.source_revision,
        "model_id": args.model_id,
        "text_column": args.text_column,
        "input_mode": "title_plus_text",
        "prefix": args.prefix,
        "normalize_embeddings": True,
        "embedding_dtype": "float32",
        "embedding_dimension": 1024,
        "rows_per_shard": args.rows_per_shard,
        "kept_columns": list(DEFAULT_COLUMNS),
    }


def load_source(args: argparse.Namespace, token: str | None):
    if args.source_json:
        if not args.source_json.is_file():
            raise FileNotFoundError(args.source_json)
        return load_dataset(
            "json",
            data_files=str(args.source_json),
            split="train",
            streaming=True,
        )
    return load_dataset(
        args.source_repo,
        split=args.source_split,
        streaming=True,
        revision=args.source_revision,
        token=token,
    )


def prepare_output_repo(
    args: argparse.Namespace,
    token: str,
    config: dict[str, Any],
) -> tuple[HfApi, int, int, bool]:
    """Return the API, next shard index, exact completed row count, and status."""
    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.output_repo,
        repo_type="dataset",
        private=args.private_output,
        exist_ok=True,
    )
    files = set(api.list_repo_files(args.output_repo, repo_type="dataset", token=token))

    if "embedding_config.json" in files:
        config_path = hf_hub_download(
            repo_id=args.output_repo,
            filename="embedding_config.json",
            repo_type="dataset",
            token=token,
        )
        existing = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if existing != config:
            raise RuntimeError(
                "The output repository contains shards made with a different configuration. "
                "Use another --output-repo or restore the original settings.\n"
                f"Existing: {json.dumps(existing, ensure_ascii=False, indent=2)}\n"
                f"Current:  {json.dumps(config, ensure_ascii=False, indent=2)}"
            )
    else:
        config_path = args.output_dir / "embedding_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        api.upload_file(
            path_or_fileobj=str(config_path),
            path_in_repo="embedding_config.json",
            repo_id=args.output_repo,
            repo_type="dataset",
            commit_message="Add E5 embedding configuration",
            token=token,
        )

    shard_indices = sorted(
        int(match.group(1))
        for filename in files
        if (match := SHARD_RE.match(filename)) is not None
    )
    if shard_indices and shard_indices != list(range(shard_indices[-1] + 1)):
        raise RuntimeError(f"Remote shards are not contiguous: {shard_indices}")

    complete = "manifest.json" in files
    next_shard = shard_indices[-1] + 1 if shard_indices else 0
    completed_rows = 0
    if shard_indices:
        last_filename = f"data/train-{shard_indices[-1]:05d}.parquet"
        last_path = hf_hub_download(
            repo_id=args.output_repo,
            filename=last_filename,
            repo_type="dataset",
            token=token,
        )
        last_rows = pq.ParquetFile(last_path).metadata.num_rows
        if not 0 < last_rows <= args.rows_per_shard:
            raise RuntimeError(f"Invalid row count in {last_filename}: {last_rows}")
        completed_rows = shard_indices[-1] * args.rows_per_shard + last_rows
    return api, next_shard, completed_rows, complete


def load_model(model_id: str, device: str) -> SentenceTransformer:
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = SentenceTransformer(
        model_id,
        device=device,
        model_kwargs={"torch_dtype": dtype},
        trust_remote_code=False,
    )
    model.max_seq_length = 512
    model.eval()
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
    dimension = model.get_sentence_embedding_dimension()
    if dimension != 1024:
        raise RuntimeError(f"Expected 1024-dimensional E5-large vectors, got {dimension}.")
    return model


def select_row(row: dict[str, Any], text_column: str) -> dict[str, Any]:
    text = row.get(text_column)
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Missing non-empty text in column {text_column!r}: {row.get('chunk_id')}")
    selected = {name: row.get(name) for name in DEFAULT_COLUMNS if name in row}
    if text_column != "text":
        selected["text"] = text
    return selected


def write_shard(
    rows: list[dict[str, Any]],
    model: SentenceTransformer,
    args: argparse.Namespace,
    shard_index: int,
) -> Path:
    texts = [
        f"{args.prefix}{str(row.get('title', '')).strip()}\n\n{row['text'].strip()}"
        for row in rows
    ]
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    if embeddings.shape != (len(rows), 1024):
        raise RuntimeError(f"Unexpected embedding shape: {embeddings.shape}")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=2e-3):
        raise RuntimeError(f"Embeddings are not L2-normalized: min={norms.min()}, max={norms.max()}")

    column_names = [name for name in DEFAULT_COLUMNS if name in rows[0]]
    arrays: dict[str, pa.Array] = {
        name: pa.array([row.get(name) for row in rows]) for name in column_names
    }
    flat = pa.array(embeddings.reshape(-1), type=pa.float32())
    arrays["embedding"] = pa.FixedSizeListArray.from_arrays(flat, 1024)
    table = pa.table(arrays).replace_schema_metadata(
        {
            b"embedding_model": args.model_id.encode(),
            b"embedding_prefix": args.prefix.encode(),
            b"normalized": b"true",
        }
    )

    shard_path = args.output_dir / "data" / f"train-{shard_index:05d}.parquet"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = shard_path.with_suffix(".parquet.incomplete")
    pq.write_table(
        table,
        temporary_path,
        compression="zstd",
        compression_level=6,
        row_group_size=min(len(rows), 1_000),
    )
    temporary_path.replace(shard_path)
    return shard_path


def upload_shard(api: HfApi, shard_path: Path, args: argparse.Namespace, token: str) -> None:
    api.upload_file(
        path_or_fileobj=str(shard_path),
        path_in_repo=f"data/{shard_path.name}",
        repo_id=args.output_repo,
        repo_type="dataset",
        commit_message=f"Add embedding shard {shard_path.stem}",
        token=token,
    )


def dataset_card(args: argparse.Namespace, total_rows: int, shard_count: int) -> str:
    source = str(args.source_json) if args.source_json else args.source_repo
    return f"""---
language:
- vi
task_categories:
- sentence-similarity
- feature-extraction
pretty_name: Vietnamese Evidence Corpus Embeddings (multilingual-e5-large v2)
---

# Vietnamese Evidence Corpus Embeddings

Normalized passage embeddings for `{source}`, generated with
`{args.model_id}`.

- Rows: {total_rows:,}
- Embedding dimension: 1024
- Embedding dtype: float32
- Prefix: `passage: `
- L2 normalized: yes
- Parquet shards: {shard_count}

Use `query: ` for claims/queries and normalize query vectors before cosine or
inner-product retrieval. The `chunk_id` column is the stable join key back to
the source corpus.
"""


def finalize(
    api: HfApi | None,
    args: argparse.Namespace,
    token: str | None,
    total_rows: int,
    shard_count: int,
) -> None:
    manifest = {
        "status": "complete",
        "rows": total_rows,
        "shards": shard_count,
        "model_id": args.model_id,
        "embedding_dimension": 1024,
        "normalized": True,
    }
    manifest_path = args.output_dir / "manifest.json"
    card_path = args.output_dir / "README.md"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    card_path.write_text(dataset_card(args, total_rows, shard_count), encoding="utf-8")

    if api is not None and token is not None:
        for local_path, remote_path in ((card_path, "README.md"), (manifest_path, "manifest.json")):
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_path,
                repo_id=args.output_repo,
                repo_type="dataset",
                commit_message=f"Finalize E5 embeddings: {remote_path}",
                token=token,
            )


def take_rows(dataset: Iterable[dict[str, Any]], max_rows: int | None):
    for index, row in enumerate(dataset):
        if max_rows is not None and index >= max_rows:
            break
        yield row


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    token = get_hf_token(required=not args.no_upload)
    config = build_config(args)
    local_config_path = args.output_dir / "embedding_config.json"
    local_config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.no_upload and any((args.output_dir / "data").glob("train-*.parquet")):
        raise RuntimeError(
            f"Local shards already exist under {args.output_dir / 'data'}. "
            "Choose a fresh --output-dir to avoid mixing runs."
        )

    api: HfApi | None = None
    start_shard = 0
    skip_rows = 0
    if not args.no_upload:
        assert token is not None
        api, start_shard, skip_rows, complete = prepare_output_repo(args, token, config)
        if complete:
            print(f"Output dataset is already complete: https://huggingface.co/datasets/{args.output_repo}")
            return

    source = load_source(args, token)
    if skip_rows:
        print(f"Resuming at shard {start_shard}; skipping {skip_rows:,} source rows.")
        source = source.skip(skip_rows)

    print(f"Device: {args.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    model = load_model(args.model_id, args.device)

    rows: list[dict[str, Any]] = []
    shard_index = start_shard
    processed_now = 0
    progress = tqdm(desc="Collecting chunks", unit="rows")
    for row in take_rows(source, args.max_rows):
        rows.append(select_row(row, args.text_column))
        processed_now += 1
        progress.update(1)
        if len(rows) < args.rows_per_shard:
            continue

        shard_path = write_shard(rows, model, args, shard_index)
        print(f"Wrote {shard_path} ({len(rows):,} rows)")
        if api is not None and token is not None:
            upload_shard(api, shard_path, args, token)
            print(f"Uploaded data/{shard_path.name}")
        rows.clear()
        shard_index += 1

    if rows:
        shard_path = write_shard(rows, model, args, shard_index)
        print(f"Wrote {shard_path} ({len(rows):,} rows)")
        if api is not None and token is not None:
            upload_shard(api, shard_path, args, token)
            print(f"Uploaded data/{shard_path.name}")
        shard_index += 1
    progress.close()

    total_rows = skip_rows + processed_now
    finalize(api, args, token, total_rows=total_rows, shard_count=shard_index)
    print(f"Complete: {total_rows:,} rows in {shard_index} shard(s).")
    if api is not None:
        print(f"https://huggingface.co/datasets/{args.output_repo}")


if __name__ == "__main__":
    main()

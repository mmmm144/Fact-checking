"""Build leakage-aware verifier splits from generated Vietnamese claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import ijson
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download, snapshot_download


LABEL_TO_ID = {"SUPPORTED": 0, "REFUTED": 1, "NOT_ENOUGH_INFO": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims-repo", required=True)
    parser.add_argument("--claims-revision", required=True)
    parser.add_argument("--claims-filename", default="data/claim_01.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-repo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--private-output", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def normalize_claim(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def split_for_doc(doc_id: str, seed: int, train_ratio: float, validation_ratio: float) -> str:
    value = stable_fraction(doc_id, seed)
    if value < train_ratio:
        return "train"
    if value < train_ratio + validation_ratio:
        return "validation"
    return "test"


def extract_evidence(item: dict[str, Any]) -> tuple[str, list[str]]:
    quotes: list[str] = []
    doc_ids: list[str] = []
    seen_quotes: set[str] = set()
    for evidence in item.get("evidence") or []:
        doc_id = str(evidence.get("article_id") or "").strip()
        if doc_id and doc_id not in doc_ids:
            doc_ids.append(doc_id)
        values = evidence.get("quote") or []
        if isinstance(values, str):
            values = [values]
        for value in values:
            quote = re.sub(r"\s+", " ", str(value)).strip()
            key = normalize_claim(quote)
            if quote and key not in seen_quotes:
                quotes.append(quote)
                seen_quotes.add(key)
    return "\n\n".join(quotes), doc_ids


def load_records(
    claims_path: Path,
    seed: int,
    train_ratio: float,
    validation_ratio: float,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    seen_claims: dict[str, str] = {}
    duplicate_count = 0
    missing_evidence = 0
    documents: dict[str, set[str]] = {name: set() for name in splits}

    with claims_path.open("rb") as handle:
        for document in ijson.items(handle, "item"):
            doc_id = str(document.get("id") or "").strip()
            if not doc_id:
                continue
            split = split_for_doc(doc_id, seed, train_ratio, validation_ratio)
            documents[split].add(doc_id)
            for group_name, items in (document.get("claims") or {}).items():
                for position, item in enumerate(items or []):
                    label = str(item.get("label") or group_name).upper()
                    claim = re.sub(r"\s+", " ", str(item.get("claim") or "")).strip()
                    if label not in LABEL_TO_ID or not claim:
                        continue
                    evidence_text, evidence_doc_ids = extract_evidence(item)
                    if not evidence_text:
                        missing_evidence += 1
                        continue
                    claim_key = normalize_claim(claim)
                    if claim_key in seen_claims:
                        duplicate_count += 1
                        continue
                    seen_claims[claim_key] = split
                    example_id = hashlib.sha1(
                        f"{doc_id}\0{label}\0{position}\0{claim}".encode("utf-8")
                    ).hexdigest()[:20]
                    splits[split].append(
                        {
                            "example_id": example_id,
                            "doc_id": doc_id,
                            "claim": claim,
                            "evidence_text": evidence_text,
                            "evidence_doc_ids": evidence_doc_ids,
                            "label": label,
                            "label_id": LABEL_TO_ID[label],
                        }
                    )

    overlap = set(documents["train"]) & set(documents["validation"])
    overlap |= set(documents["train"]) & set(documents["test"])
    overlap |= set(documents["validation"]) & set(documents["test"])
    if overlap:
        raise RuntimeError(f"Document leakage across splits: {len(overlap)} doc_id(s)")

    stats = {
        "documents": {split: len(values) for split, values in documents.items()},
        "examples": {split: len(values) for split, values in splits.items()},
        "labels": {
            split: dict(Counter(record["label"] for record in values))
            for split, values in splits.items()
        },
        "exact_duplicate_claims_removed": duplicate_count,
        "missing_evidence_removed": missing_evidence,
        "document_overlap": 0,
    }
    return splits, stats


def remote_is_complete(args: argparse.Namespace, token: str | None) -> bool:
    if not args.output_repo or args.force:
        return False
    try:
        manifest_path = hf_hub_download(
            repo_id=args.output_repo,
            filename="manifest.json",
            repo_type="dataset",
            token=token,
        )
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except Exception:
        return False
    expected = {
        "status": "complete",
        "claims_repo": args.claims_repo,
        "claims_revision": args.claims_revision,
        "claims_filename": args.claims_filename,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "validation_ratio": args.validation_ratio,
    }
    return all(manifest.get(key) == value for key, value in expected.items())


def write_dataset(args: argparse.Namespace, splits: dict[str, list[dict[str, Any]]], stats: dict[str, Any]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        pq.write_table(pa.Table.from_pylist(rows), args.output_dir / f"{split}.parquet", compression="zstd")
    manifest = {
        "status": "complete",
        "claims_repo": args.claims_repo,
        "claims_revision": args.claims_revision,
        "claims_filename": args.claims_filename,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "validation_ratio": args.validation_ratio,
        "label_to_id": LABEL_TO_ID,
        **stats,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    card = f"""---
language: vi
task_categories:
- text-classification
tags:
- fact-checking
- natural-language-inference
---

# Vietnamese fact-checking verifier data

Leakage-aware document-level 80/10/10 split derived from
`{args.claims_repo}` at revision `{args.claims_revision}`.

Input is `(evidence_text, claim)` and labels are `SUPPORTED`, `REFUTED`, and
`NOT_ENOUGH_INFO`. Exact duplicate claims are retained only once.
"""
    (args.output_dir / "README.md").write_text(card, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not 0 < args.train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1")
    if not 0 < args.validation_ratio < 1 or args.train_ratio + args.validation_ratio >= 1:
        raise ValueError("train + validation ratios must be below 1")

    import os

    token = os.environ.get("HF_TOKEN")
    if remote_is_complete(args, token):
        snapshot_download(
            repo_id=args.output_repo,
            repo_type="dataset",
            local_dir=args.output_dir,
            token=token,
        )
        print(f"Reused verifier dataset: https://huggingface.co/datasets/{args.output_repo}")
        return

    claims_path = Path(
        hf_hub_download(
            repo_id=args.claims_repo,
            filename=args.claims_filename,
            repo_type="dataset",
            revision=args.claims_revision,
            token=token,
        )
    )
    splits, stats = load_records(
        claims_path, args.seed, args.train_ratio, args.validation_ratio
    )
    write_dataset(args, splits, stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    if args.output_repo:
        api = HfApi(token=token)
        api.create_repo(
            repo_id=args.output_repo,
            repo_type="dataset",
            private=args.private_output,
            exist_ok=True,
        )
        api.upload_folder(
            repo_id=args.output_repo,
            repo_type="dataset",
            folder_path=args.output_dir,
            commit_message="Build leakage-aware verifier train validation test splits",
        )
        print(f"Uploaded: https://huggingface.co/datasets/{args.output_repo}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Repair missing or unindexable evidence article IDs in generated claim files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def load_corpus_ids(path: Path) -> set[str]:
    records = json.loads(path.read_text(encoding="utf-8-sig"))
    ids = {str(record.get("doc_id") or "").strip() for record in records}
    ids.discard("")
    if len(ids) != len(records):
        raise ValueError("Corpus contains a missing or duplicate doc_id")
    return ids


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            json.dump(value, temporary, ensure_ascii=False, indent=2)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def repair_file(path: Path, corpus_ids: set[str]) -> tuple[int, int, list[dict[str, Any]]]:
    documents = json.loads(path.read_text(encoding="utf-8-sig"))
    repaired_entries = 0
    repaired_claims = 0
    changes: list[dict[str, Any]] = []

    for document in documents:
        parent_doc_id = str(document.get("id") or "").strip()
        if parent_doc_id not in corpus_ids:
            continue
        for group_name, claims in (document.get("claims") or {}).items():
            for claim_index, claim in enumerate(claims or []):
                claim_changed = False
                for evidence_index, evidence in enumerate(claim.get("evidence") or []):
                    old_id = str(evidence.get("article_id") or "").strip()
                    if old_id in corpus_ids:
                        continue
                    evidence["article_id"] = parent_doc_id
                    repaired_entries += 1
                    claim_changed = True
                    changes.append(
                        {
                            "file": str(path),
                            "parent_doc_id": parent_doc_id,
                            "label": str(claim.get("label") or group_name),
                            "claim_index": claim_index,
                            "evidence_index": evidence_index,
                            "old_article_id": old_id or None,
                            "new_article_id": parent_doc_id,
                        }
                    )
                repaired_claims += int(claim_changed)

    if repaired_entries:
        atomic_write_json(path, documents)
    return repaired_entries, repaired_claims, changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--claims", type=Path, nargs="+", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    corpus_ids = load_corpus_ids(args.corpus)
    total_entries = 0
    total_claims = 0
    all_changes: list[dict[str, Any]] = []
    for path in args.claims:
        entries, claims, changes = repair_file(path, corpus_ids)
        total_entries += entries
        total_claims += claims
        all_changes.extend(changes)

    summary = {
        "status": "complete",
        "corpus": str(args.corpus),
        "corpus_sha256": sha256_file(args.corpus),
        "corpus_documents": len(corpus_ids),
        "claim_files": [str(path) for path in args.claims],
        "claim_file_sha256": {str(path): sha256_file(path) for path in args.claims},
        "repaired_evidence_entries": total_entries,
        "repaired_claims": total_claims,
        "changes": all_changes,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "changes"}, indent=2))


if __name__ == "__main__":
    main()

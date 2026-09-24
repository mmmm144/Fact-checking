#!/usr/bin/env python3
"""Audit duplicate passages and navigation boilerplate in a chunk JSON array."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from chunking_e5_v3 import iter_json_array, normalize_for_dedup


PHRASE_PATTERNS = {
    "xem_them": re.compile(r"(?i)\bxem\s+th[eê]m\b"),
    "tham_khao_them": re.compile(r"(?i)\btham\s+kh[aả]o\s+th[eê]m\b"),
    "related_link": re.compile(r"(?i)\brelated\s+links?\b"),
}

NAVIGATION_PATTERNS = {
    "bao_chinh_phu_related_suffix": re.compile(
        r"(?m)^[ \t]*Tham khảo thêm(?:[ \t:]|$)"
    ),
    "who_related_link_heading": re.compile(
        r"(?m)^[ \t]*Related links?\s*:?(?:[ \t]|$)"
    ),
}


def audit(path: Path, sample_limit: int = 5) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    sources: defaultdict[str, Counter[str]] = defaultdict(Counter)
    examples: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    documents: set[str] = set()
    normalized_texts: set[str] = set()
    normalized_passages: set[str] = set()
    content_hashes: set[str] = set()

    for row in iter_json_array(path):
        if not isinstance(row, dict):
            counts["invalid_rows"] += 1
            continue
        counts["rows"] += 1
        doc_id = str(row.get("doc_id") or "")
        source = str(row.get("source") or row.get("source_name") or "UNKNOWN")
        title = str(row.get("title") or "")
        text = str(row.get("text") or "")
        content_hash = str(row.get("content_hash") or "")
        documents.add(doc_id)

        normalized_text = normalize_for_dedup(text)
        if normalized_text in normalized_texts:
            counts["duplicate_normalized_text_rows"] += 1
        normalized_texts.add(normalized_text)

        normalized_passage = normalize_for_dedup(f"passage: {title.strip()}\n\n{text}")
        if normalized_passage in normalized_passages:
            counts["duplicate_normalized_passage_rows"] += 1
        normalized_passages.add(normalized_passage)

        if content_hash:
            if content_hash in content_hashes:
                counts["duplicate_content_hash_rows"] += 1
            content_hashes.add(content_hash)

        for name, pattern in {**PHRASE_PATTERNS, **NAVIGATION_PATTERNS}.items():
            match = pattern.search(text)
            if not match:
                continue
            counts[f"{name}_chunks"] += 1
            sources[name][source] += 1
            if len(examples[name]) < sample_limit:
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 180)
                examples[name].append(
                    {
                        "chunk_id": str(row.get("chunk_id") or ""),
                        "source": source,
                        "snippet": text[start:end].replace("\n", " "),
                    }
                )

    return {
        "input": str(path),
        "rows": counts.pop("rows", 0),
        "documents": len(documents),
        "unique_normalized_texts": len(normalized_texts),
        "unique_normalized_passages": len(normalized_passages),
        "unique_content_hashes": len(content_hashes),
        "counts": dict(sorted(counts.items())),
        "sources": {name: dict(counter) for name, counter in sorted(sources.items())},
        "examples": dict(examples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sample-limit", type=int, default=5)
    args = parser.parse_args()
    result = audit(args.input, max(0, args.sample_limit))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

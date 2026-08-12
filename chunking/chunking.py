#!/usr/bin/env python3
"""Chia corpus JSON th?nh c?c chunk theo token c?a BGE-M3.

M?i ph?n t? ??u ra l? m?t chunk ph?ng. C?c tr??ng c?a t?i li?u ngu?n ???c gi?
nguy?n, ri?ng ``text`` ???c thay b?ng n?i dung chunk v? b? sung th?ng tin v? tr?
token. T?p v?o/ra ??u ???c x? l? tu?n t? ?? tr?nh gi? to?n b? corpus trong RAM.
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "vie"
    / "raw"
    / "viet-fact-checking"
    / "corpus_v1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "vie" / "processed" / "corpus_v1_chunked.json"
)
DEFAULT_MODEL = "BAAI/bge-m3"


class Tokenizer(Protocol):
    """Ph?n giao di?n tokenizer m? logic chunking s? d?ng."""

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
        verbose: bool,
    ) -> Any: ...


SENTENCE_BOUNDARY_RE = re.compile(
    r"(?:[.!?\u2026]+[\"'\u201d\u2019\u00bb)\]]*)(?=\s|$)|\n+"
)


def iter_json_array(path: Path, block_size: int = 1024 * 1024) -> Iterator[Any]:
    """??c l?n l??t c?c ph?n t? c?a m?t m?ng JSON c?p ngo?i c?ng."""

    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8-sig") as handle:
        buffer = ""
        position = 0
        eof = False

        def read_more(*, compact: bool = True) -> None:
            nonlocal buffer, position, eof
            if compact and position:
                buffer = buffer[position:]
                position = 0
            data = handle.read(block_size)
            if data:
                buffer += data
            else:
                eof = True

        def skip_whitespace() -> None:
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or eof:
                    return
                read_more()

        read_more()
        skip_whitespace()
        if position >= len(buffer) or buffer[position] != "[":
            raise ValueError(f"{path} ph?i ch?a m?t m?ng JSON ? c?p ngo?i c?ng")
        position += 1

        first_item = True
        while True:
            skip_whitespace()
            if position >= len(buffer):
                raise ValueError(f"M?ng JSON trong {path} k?t th?c kh?ng h?p l?")

            if buffer[position] == "]":
                position += 1
                skip_whitespace()
                if not eof:
                    read_more()
                    skip_whitespace()
                if position < len(buffer):
                    raise ValueError(f"{path} c? d? li?u th?a sau m?ng JSON")
                return

            if not first_item:
                if buffer[position] != ",":
                    raise ValueError(
                        f"Thi?u d?u ph?y gi?a c?c ph?n t? JSON trong {path}"
                    )
                position += 1
                skip_whitespace()
                if position >= len(buffer):
                    raise ValueError(f"M?ng JSON trong {path} k?t th?c kh?ng h?p l?")

            while True:
                try:
                    item, end_position = decoder.raw_decode(buffer, position)
                    break
                except json.JSONDecodeError as exc:
                    if eof:
                        raise ValueError(
                            f"JSON kh?ng h?p l? trong {path}: {exc}"
                        ) from exc
                    read_more()

            yield item
            position = end_position
            first_item = False


def chunk_token_ids(
    token_ids: Sequence[int], chunk_size: int, overlap: int
) -> Iterator[tuple[int, int, Sequence[int]]]:
    """Sinh c?c c?a s? ``(token_start, token_end, token_ids)``."""

    if chunk_size <= 0:
        raise ValueError("chunk_size ph?i l?n h?n 0")
    if overlap < 0:
        raise ValueError("overlap kh?ng ???c ?m")
    if overlap >= chunk_size:
        raise ValueError("overlap ph?i nh? h?n chunk_size")

    step = chunk_size - overlap
    for start in range(0, len(token_ids), step):
        end = min(start + chunk_size, len(token_ids))
        yield start, end, token_ids[start:end]
        if end == len(token_ids):
            break


def sentence_char_spans(text: str) -> list[tuple[int, int]]:
    """Return trimmed character spans for sentences and explicit lines."""

    boundaries = [0, *(m.end() for m in SENTENCE_BOUNDARY_RE.finditer(text))]
    if boundaries[-1] != len(text):
        boundaries.append(len(text))
    spans = []
    for start, end in zip(boundaries, boundaries[1:]):
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append((start, end))
    return spans


def _tokenize_with_offsets(text: str, tokenizer: Tokenizer):
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        verbose=False,
    )
    token_ids = list(encoded["input_ids"])
    offsets = [tuple(offset) for offset in encoded["offset_mapping"]]
    if len(token_ids) != len(offsets):
        raise ValueError("tokenizer returned mismatched input_ids and offsets")
    if any(start < 0 or end <= start or end > len(text) for start, end in offsets):
        raise ValueError("tokenizer returned invalid offset_mapping")
    return token_ids, offsets


def _sentence_token_spans(
    text: str, offsets: Sequence[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Project source sentence spans onto BGE-M3 token ranges."""

    if not offsets:
        return []
    token_end_chars = [end for _, end in offsets]
    spans = []
    token_start = 0
    for _, char_end in sentence_char_spans(text):
        token_end = bisect.bisect_right(token_end_chars, char_end)
        if token_end > token_start:
            spans.append((token_start, token_end))
            token_start = token_end
    if token_start < len(offsets):
        spans.append((token_start, len(offsets)))
    return spans


def sentence_aware_token_ranges(
    token_count: int,
    sentence_spans: Sequence[tuple[int, int]],
    chunk_size: int,
    overlap: int,
    min_chunk_size: int,
) -> list[tuple[int, int]]:
    """Create ranges <= chunk_size, preferring sentence boundaries."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and less than chunk_size")
    if min_chunk_size <= 0 or min_chunk_size > chunk_size:
        raise ValueError("min_chunk_size must be between 1 and chunk_size")
    if token_count == 0:
        return []

    boundaries = sorted(
        {0, token_count, *(end for _, end in sentence_spans if 0 < end < token_count)}
    )
    ranges = []
    start = 0
    while start < token_count:
        max_end = min(start + chunk_size, token_count)
        boundary_index = bisect.bisect_right(boundaries, max_end) - 1
        end = boundaries[boundary_index]
        if end <= start:
            # Only an oversized sentence is split at a hard token boundary.
            end = max_end
        ranges.append((start, end))
        if end == token_count:
            break

        if overlap == 0:
            start = end
            continue

        desired_start = max(start + 1, end - overlap)
        boundary_index = bisect.bisect_left(boundaries, desired_start)
        if boundary_index < len(boundaries) and boundaries[boundary_index] < end:
            start = boundaries[boundary_index]
        else:
            start = max(start + 1, end - overlap)

    # Merge a weak tail when possible; otherwise enlarge its overlap.
    if len(ranges) >= 2 and ranges[-1][1] - ranges[-1][0] < min_chunk_size:
        previous_start, _ = ranges[-2]
        final_end = ranges[-1][1]
        if final_end - previous_start <= chunk_size:
            ranges[-2] = (previous_start, final_end)
            ranges.pop()
        else:
            lower_bound = max(previous_start + 1, final_end - chunk_size)
            upper_bound = final_end - min_chunk_size
            boundary_index = bisect.bisect_right(boundaries, upper_bound) - 1
            extended_start = boundaries[boundary_index]
            if extended_start < lower_bound:
                extended_start = upper_bound
            ranges[-1] = (extended_start, final_end)

    return ranges


def chunk_document(
    document: dict[str, Any],
    tokenizer: Tokenizer,
    chunk_size: int,
    overlap: int,
    min_chunk_size: int = 64,
) -> Iterator[dict[str, Any]]:
    """Chia tr??ng ``text`` c?a m?t t?i li?u v? gi? metadata ngu?n."""

    doc_id = str(document.get("doc_id", "")).strip()
    if not doc_id:
        raise ValueError("t?i li?u thi?u tr??ng doc_id")

    text = document.get("text")
    if not isinstance(text, str):
        raise ValueError(f"t?i li?u {doc_id} c? tr??ng text kh?ng ph?i chu?i")

    token_ids, offsets = _tokenize_with_offsets(text, tokenizer)
    if not token_ids:
        return

    sentence_spans = _sentence_token_spans(text, offsets)
    ranges = sentence_aware_token_ranges(
        len(token_ids), sentence_spans, chunk_size, overlap, min_chunk_size
    )
    sentence_starts = [start for start, _ in sentence_spans]
    sentence_ends = [end for _, end in sentence_spans]

    source_fields = {
        key: value
        for key, value in document.items()
        if key not in {"doc_id", "text"}
    }
    for chunk_index, (start, end) in enumerate(ranges):
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        chunk_text = text[char_start:char_end]
        if not chunk_text:
            continue

        sentence_start = bisect.bisect_right(sentence_ends, start)
        sentence_end = bisect.bisect_left(sentence_starts, end)

        yield {
            **source_fields,
            "chunk_id": f"{doc_id}_chunk_{chunk_index:04d}",
            "doc_id": doc_id,
            "chunk_index": chunk_index,
            "token_start": start,
            "token_end": end,
            "token_count": end - start,
            "char_start": char_start,
            "char_end": char_end,
            "sentence_start": sentence_start,
            "sentence_end": sentence_end,
            "text": chunk_text,
        }


def load_tokenizer(model_name: str, local_files_only: bool) -> Tokenizer:
    """Load a Hugging Face tokenizer with an actionable error message."""

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot import transformers: {exc}"
        ) from exc

    try:
        return AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        hint = " (kh?ng c? trong cache c?c b?)" if local_files_only else ""
        raise RuntimeError(f"Kh?ng th? n?p tokenizer {model_name!r}{hint}") from exc


def _write_chunks_atomic(
    documents: Iterable[Any],
    output_path: Path,
    tokenizer: Tokenizer,
    chunk_size: int,
    overlap: int,
    min_chunk_size: int,
    progress_every: int,
) -> tuple[int, int, int]:
    """Ghi m?ng chunk JSON tu?n t? r?i thay file ??ch atomically."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        delete=False,
    )
    temp_path = Path(temp_file.name)
    document_count = 0
    chunk_count = 0
    skipped_count = 0

    try:
        with temp_file:
            temp_file.write("[\n")
            first_chunk = True
            for document_count, document in enumerate(documents, start=1):
                if not isinstance(document, dict):
                    skipped_count += 1
                    print(
                        f"C?nh b?o: b? qua ph?n t? #{document_count} "
                        "v? kh?ng ph?i JSON object",
                        file=sys.stderr,
                    )
                    continue

                try:
                    chunks = chunk_document(
                        document,
                        tokenizer,
                        chunk_size,
                        overlap,
                        min_chunk_size,
                    )
                    wrote_chunk = False
                    for chunk in chunks:
                        if not first_chunk:
                            temp_file.write(",\n")
                        json.dump(chunk, temp_file, ensure_ascii=False)
                        first_chunk = False
                        wrote_chunk = True
                        chunk_count += 1
                    if not wrote_chunk:
                        skipped_count += 1
                except ValueError as exc:
                    skipped_count += 1
                    print(f"C?nh b?o: b? qua {exc}", file=sys.stderr)

                if progress_every and document_count % progress_every == 0:
                    print(
                        f"?? x? l? {document_count:,} t?i li?u, "
                        f"t?o {chunk_count:,} chunk...",
                        file=sys.stderr,
                    )

            temp_file.write("\n]\n")

        os.replace(temp_path, output_path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise

    return document_count, chunk_count, skipped_count


def write_chunks(
    documents: Iterable[Any],
    output_path: Path,
    tokenizer: Tokenizer,
    chunk_size: int,
    overlap: int,
    progress_every: int,
    min_chunk_size: int = 64,
) -> tuple[int, int, int]:
    """Stream chunks to a temp file and atomically replace the destination."""

    return _write_chunks_atomic(
        documents,
        output_path,
        tokenizer,
        chunk_size,
        overlap,
        min_chunk_size,
        progress_every,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chia tr??ng text c?a corpus JSON theo token BGE-M3."
    )
    parser.description = "Split corpus text into BGE-M3 token chunks."
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Maximum BGE-M3 tokens per chunk (default: 512)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="Target token overlap, aligned to sentence boundaries (default: 64)",
    )
    parser.add_argument(
        "--min-chunk-size",
        type=int,
        default=64,
        help="Minimum tail size; merge or extend overlap when smaller (default: 64)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="In ti?n ?? sau m?i N t?i li?u; 0 ?? t?t",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Ch? d?ng tokenizer trong Hugging Face cache",
    )
    help_by_dest = {
        "progress_every": "Print progress every N documents; use 0 to disable",
        "local_files_only": "Only load the tokenizer from the local cache",
    }
    for action in parser._actions:
        if action.dest in help_by_dest:
            action.help = help_by_dest[action.dest]
    parser.set_defaults(progress_every=1)
    parser._option_string_actions["--progress-every"].help += " (default: 1)"
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not args.input.is_file():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        return 2
    if args.chunk_size <= 0:
        print("Error: --chunk-size must be greater than 0", file=sys.stderr)
        return 2
    if args.overlap < 0 or args.overlap >= args.chunk_size:
        print(
            "Error: --overlap must be >= 0 and less than --chunk-size",
            file=sys.stderr,
        )
        return 2
    if args.min_chunk_size <= 0 or args.min_chunk_size > args.chunk_size:
        print(
            "Error: --min-chunk-size must be > 0 and <= --chunk-size",
            file=sys.stderr,
        )
        return 2
    if args.progress_every < 0:
        print("Error: --progress-every cannot be negative", file=sys.stderr)
        return 2

    try:
        tokenizer = load_tokenizer(args.model, args.local_files_only)
        document_count, chunk_count, skipped_count = write_chunks(
            iter_json_array(args.input),
            args.output,
            tokenizer,
            args.chunk_size,
            args.overlap,
            args.progress_every,
            args.min_chunk_size,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Done: {document_count:,} documents -> {chunk_count:,} chunks; "
        f"skipped {skipped_count:,}. Output: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

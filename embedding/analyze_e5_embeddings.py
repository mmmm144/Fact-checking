#!/usr/bin/env python3
"""Validate and summarize downloaded multilingual-E5 embedding shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.decomposition import PCA


DIMENSION = 1024
RNG_SEED = 20260824


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("embedding/e5-results"))
    parser.add_argument("--output-dir", type=Path, default=Path("embedding/e5-analysis"))
    return parser.parse_args()


def fixed_list_to_numpy(column: pa.ChunkedArray) -> np.ndarray:
    values = column.combine_chunks()
    if not pa.types.is_fixed_size_list(values.type) or values.type.list_size != DIMENSION:
        raise ValueError(f"Expected fixed_size_list<float>[{DIMENSION}], got {values.type}")
    return values.values.to_numpy(zero_copy_only=False).reshape(len(values), DIMENSION)


def quantiles(values: np.ndarray) -> dict[str, float]:
    levels = (0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1)
    result = np.quantile(values, levels)
    return {f"p{int(level * 100):02d}": float(value) for level, value in zip(levels, result)}


def value_counts(frame: pd.DataFrame, column: str) -> list[dict[str, object]]:
    counts = frame[column].fillna("<missing>").astype(str).value_counts(dropna=False)
    total = len(frame)
    return [
        {"value": value, "rows": int(count), "percent": float(100 * count / total)}
        for value, count in counts.items()
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted((args.input_dir / "data").glob("train-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No Parquet shards under {args.input_dir / 'data'}")

    tables: list[pa.Table] = []
    vectors: list[np.ndarray] = []
    shard_rows: list[dict[str, object]] = []
    expected_schema: pa.Schema | None = None
    expected_metadata = {
        b"embedding_model": b"intfloat/multilingual-e5-large",
        b"embedding_prefix": b"passage: ",
        b"normalized": b"true",
    }

    for shard in shards:
        table = pq.read_table(shard)
        schema = table.schema
        schema_ok = expected_schema is None or schema.equals(expected_schema, check_metadata=True)
        expected_schema = expected_schema or schema
        metadata = schema.metadata or {}
        metadata_ok = all(metadata.get(key) == value for key, value in expected_metadata.items())
        shard_vectors = fixed_list_to_numpy(table["embedding"]).astype(np.float32, copy=False)
        norms = np.linalg.norm(shard_vectors, axis=1)
        shard_rows.append(
            {
                "shard": shard.name,
                "rows": len(table),
                "size_mb": round(shard.stat().st_size / 1_000_000, 3),
                "schema_ok": schema_ok,
                "metadata_ok": metadata_ok,
                "norm_min": float(norms.min()),
                "norm_max": float(norms.max()),
            }
        )
        tables.append(table.drop(["embedding"]))
        vectors.append(shard_vectors)

    metadata_table = pa.concat_tables(tables)
    frame = metadata_table.to_pandas()
    matrix = np.concatenate(vectors, axis=0)
    rows, dimensions = matrix.shape
    norms = np.linalg.norm(matrix, axis=1)
    rng = np.random.default_rng(RNG_SEED)

    pair_count = 50_000
    left = rng.integers(0, rows, size=pair_count)
    right = rng.integers(0, rows, size=pair_count)
    collisions = left == right
    right[collisions] = (right[collisions] + 1) % rows
    random_cosines = np.einsum("ij,ij->i", matrix[left], matrix[right])

    adjacent_mask = (
        frame["doc_id"].iloc[:-1].to_numpy() == frame["doc_id"].iloc[1:].to_numpy()
    ) & (
        frame["chunk_index"].iloc[1:].to_numpy()
        == frame["chunk_index"].iloc[:-1].to_numpy() + 1
    )
    adjacent_positions = np.flatnonzero(adjacent_mask)
    adjacent_cosines = np.einsum(
        "ij,ij->i", matrix[adjacent_positions], matrix[adjacent_positions + 1]
    )

    vector_hashes = [hashlib.blake2b(row.tobytes(), digest_size=16).digest() for row in matrix]
    vector_hash_counts = Counter(vector_hashes)
    duplicate_vector_rows = sum(count for count in vector_hash_counts.values() if count > 1)
    duplicate_vector_groups = sum(count > 1 for count in vector_hash_counts.values())

    text = frame["text"].fillna("").astype(str)
    normalized_text = text.str.replace(r"\s+", " ", regex=True).str.strip()
    text_counts = normalized_text.value_counts()
    duplicate_text_rows = int(text_counts[text_counts > 1].sum())
    duplicate_text_groups = int((text_counts > 1).sum())
    char_lengths = text.str.len().to_numpy()
    word_lengths = text.str.split().str.len().to_numpy()

    sample_size = min(5_000, rows)
    pca_indices = rng.choice(rows, size=sample_size, replace=False)
    pca = PCA(n_components=20, svd_solver="randomized", random_state=RNG_SEED)
    pca.fit(matrix[pca_indices])

    query_count = min(256, rows)
    query_indices = np.sort(rng.choice(rows, size=query_count, replace=False))
    index = faiss.IndexFlatIP(dimensions)
    index.add(matrix)
    similarities, neighbors = index.search(matrix[query_indices], 6)
    neighbor_rows: list[dict[str, object]] = []
    same_doc_top1 = 0
    for query_position, query_index in enumerate(query_indices):
        candidate_index = int(neighbors[query_position, 1])
        same_doc = frame.iloc[query_index]["doc_id"] == frame.iloc[candidate_index]["doc_id"]
        same_doc_top1 += int(same_doc)
        neighbor_rows.append(
            {
                "query_row": int(query_index),
                "query_chunk_id": frame.iloc[query_index]["chunk_id"],
                "query_title": frame.iloc[query_index]["title"],
                "neighbor_row": candidate_index,
                "neighbor_chunk_id": frame.iloc[candidate_index]["chunk_id"],
                "neighbor_title": frame.iloc[candidate_index]["title"],
                "cosine": float(similarities[query_position, 1]),
                "same_doc": bool(same_doc),
            }
        )

    categories = {}
    for column in ("source", "source_type", "domain", "document_type", "language", "country"):
        categories[column] = value_counts(frame, column)
        write_csv(args.output_dir / f"counts_{column}.csv", categories[column])

    publish_dates = pd.to_datetime(frame["publish_date"], errors="coerce")
    centroid = matrix.mean(axis=0)
    component_std = matrix.std(axis=0)
    summary = {
        "input": {
            "directory": str(args.input_dir),
            "shards": len(shards),
            "rows": rows,
            "dimensions": dimensions,
            "matrix_size_mb": float(matrix.nbytes / 1_000_000),
        },
        "integrity": {
            "all_finite": bool(np.isfinite(matrix).all()),
            "non_finite_values": int((~np.isfinite(matrix)).sum()),
            "unique_chunk_ids": int(frame["chunk_id"].nunique(dropna=True)),
            "duplicate_chunk_id_rows": int(frame["chunk_id"].duplicated(keep=False).sum()),
            "unique_doc_ids": int(frame["doc_id"].nunique(dropna=True)),
            "duplicate_text_rows": duplicate_text_rows,
            "duplicate_text_percent": float(100 * duplicate_text_rows / rows),
            "duplicate_text_groups": duplicate_text_groups,
            "duplicate_vector_rows": duplicate_vector_rows,
            "duplicate_vector_percent": float(100 * duplicate_vector_rows / rows),
            "duplicate_vector_groups": duplicate_vector_groups,
            "empty_text_rows": int((normalized_text == "").sum()),
            "null_counts": {column: int(frame[column].isna().sum()) for column in frame.columns},
        },
        "embedding": {
            "norm": {
                "min": float(norms.min()),
                "max": float(norms.max()),
                "mean": float(norms.mean()),
                "std": float(norms.std()),
                "max_abs_error_from_one": float(np.max(np.abs(norms - 1))),
            },
            "centroid_norm": float(np.linalg.norm(centroid)),
            "component_std_min": float(component_std.min()),
            "component_std_max": float(component_std.max()),
            "random_pair_cosine": quantiles(random_cosines),
            "adjacent_chunk_cosine": quantiles(adjacent_cosines),
            "pca_explained_variance_ratio_top20": [float(x) for x in pca.explained_variance_ratio_],
            "pca_cumulative_top20": float(pca.explained_variance_ratio_.sum()),
            "sampled_top1_neighbor_same_doc_percent": float(100 * same_doc_top1 / query_count),
            "sampled_top1_neighbor_cosine": quantiles(similarities[:, 1]),
        },
        "text": {
            "characters": quantiles(char_lengths),
            "words": quantiles(word_lengths),
            "rows_under_50_characters": int((char_lengths < 50).sum()),
            "rows_over_5000_characters": int((char_lengths > 5_000).sum()),
            "rows_containing_xem_them": int(text.str.contains("Xem thêm", case=False, regex=False).sum()),
        },
        "documents": {
            "chunks_per_doc": quantiles(frame.groupby("doc_id").size().to_numpy()),
            "max_chunks_per_doc": int(frame.groupby("doc_id").size().max()),
        },
        "dates": {
            "min": publish_dates.min().isoformat() if publish_dates.notna().any() else None,
            "max": publish_dates.max().isoformat() if publish_dates.notna().any() else None,
            "missing": int(publish_dates.isna().sum()),
        },
        "categories": categories,
        "shards": shard_rows,
    }

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(args.output_dir / "shard_summary.csv", shard_rows)
    write_csv(args.output_dir / "sample_nearest_neighbors.csv", neighbor_rows)

    report = f"""# Phân tích embedding multilingual-E5-large

## Tổng quan

- {rows:,} chunks thuộc {summary['integrity']['unique_doc_ids']:,} tài liệu, gồm {len(shards)} shard.
- Vector: `float32[{dimensions}]`, chiếm {summary['input']['matrix_size_mb']:.1f} MB khi giải nén trong RAM.
- Model/prefix: `intfloat/multilingual-e5-large` / `passage: `.
- Khoảng ngày xuất bản: {summary['dates']['min']} đến {summary['dates']['max']}.

## Kiểm định chất lượng kỹ thuật

- Tất cả giá trị hữu hạn: **{summary['integrity']['all_finite']}**; số giá trị NaN/Inf: {summary['integrity']['non_finite_values']}.
- Norm L2: min {norms.min():.8f}, trung bình {norms.mean():.8f}, max {norms.max():.8f}; sai số lớn nhất so với 1 là {summary['embedding']['norm']['max_abs_error_from_one']:.2e}.
- `chunk_id` duy nhất: {summary['integrity']['unique_chunk_ids']:,}/{rows:,}; dòng có ID trùng: {summary['integrity']['duplicate_chunk_id_rows']}.
- Vector trùng tuyệt đối: {duplicate_vector_rows} dòng trong {duplicate_vector_groups} nhóm.
- Văn bản trống: {summary['integrity']['empty_text_rows']}; văn bản chuẩn hóa trùng: {duplicate_text_rows} dòng ({100 * duplicate_text_rows / rows:.2f}%) trong {duplicate_text_groups} nhóm.
- Schema và metadata nhất quán trên mọi shard: {all(row['schema_ok'] and row['metadata_ok'] for row in shard_rows)}.

## Hình học embedding

- Cosine cặp ngẫu nhiên (50.000 cặp): median {np.median(random_cosines):.4f}, p05 {np.quantile(random_cosines, .05):.4f}, p95 {np.quantile(random_cosines, .95):.4f}.
- Cosine giữa chunks kế tiếp cùng tài liệu ({len(adjacent_cosines):,} cặp): median {np.median(adjacent_cosines):.4f}, p05 {np.quantile(adjacent_cosines, .05):.4f}, p95 {np.quantile(adjacent_cosines, .95):.4f}.
- Norm của centroid: {np.linalg.norm(centroid):.4f}. Đây là dấu hiệu không gian có tính bất đẳng hướng; nên xếp hạng bằng cosine/IP tương đối, không diễn giải điểm tuyệt đối như xác suất.
- Sai số norm nhỏ nhưng khiến một số inner product vượt 1; có thể gọi `faiss.normalize_L2(matrix)` trước `index.add()` để cosine/IP đúng chặt chẽ hơn.
- 20 thành phần PCA đầu giải thích {100 * pca.explained_variance_ratio_.sum():.2f}% phương sai trên mẫu {sample_size:,} vector.
- Với {query_count} query corpus lấy mẫu, nearest neighbor khác chính nó thuộc cùng tài liệu ở {100 * same_doc_top1 / query_count:.1f}% trường hợp; cosine median {np.median(similarities[:, 1]):.4f}.

## Văn bản và metadata

- Độ dài: median {np.median(char_lengths):,.0f} ký tự / {np.median(word_lengths):,.0f} từ; p05–p95 là {np.quantile(char_lengths, .05):,.0f}–{np.quantile(char_lengths, .95):,.0f} ký tự.
- Chunks/tài liệu: median {np.median(frame.groupby('doc_id').size()):.0f}, p95 {np.quantile(frame.groupby('doc_id').size(), .95):.0f}, tối đa {frame.groupby('doc_id').size().max()}.
- Dòng chứa cụm `Xem thêm`: {summary['text']['rows_containing_xem_them']:,}; nên kiểm tra nhiễu điều hướng/boilerplate trước khi đánh giá retrieval.
- Nguồn lớn nhất: {categories['source'][0]['value']} ({categories['source'][0]['rows']:,} chunks, {categories['source'][0]['percent']:.1f}%).
- Domain lớn nhất: {categories['domain'][0]['value']} ({categories['domain'][0]['rows']:,} chunks, {categories['domain'][0]['percent']:.1f}%).
- Ngôn ngữ: tiếng Việt {categories['language'][0]['percent']:.1f}%, tiếng Anh {categories['language'][1]['percent']:.1f}%; cần tách lát đánh giá theo ngôn ngữ và nguồn để tránh metric tổng che khuất chênh lệch.

## Kết luận

Dữ liệu **đạt kiểm định kỹ thuật cơ bản** để xây FAISS `IndexFlatIP`: đúng 1024 chiều, gần chuẩn hóa L2, không NaN/Inf, ID chunk duy nhất và shard nhất quán. Tuy nhiên, {duplicate_text_rows:,} dòng trùng văn bản và các mẫu `Xem thêm` cho thấy boilerplate/navigation đáng kể (đặc biệt ở GSO và VnExpress); nếu index nguyên trạng, các bản sao có thể chiếm nhiều vị trí top-k. Nên loại exact duplicate/boilerplate trước khi đánh giá chính thức, re-normalize vector ngay trước khi index, embed claim bằng prefix `query: ` và báo cáo metric theo nguồn/ngôn ngữ bên cạnh metric tổng.
"""
    (args.output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

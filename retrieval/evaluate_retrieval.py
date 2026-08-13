"""Evaluate BM25, dense, hybrid, and hybrid+BGE reranking on silver claims."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import ijson
import numpy as np
from tqdm.auto import tqdm

from hybrid_retriever import HybridRetriever


METRIC_KS = (1, 5, 10, 50)
ANSWERABLE_LABELS = {"SUPPORTED", "REFUTED"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--claims", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/retrieval_evaluation"),
    )
    parser.add_argument("--max-queries", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--fusion-k", type=int, default=100)
    parser.add_argument("--rerank-k", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    return parser.parse_args()


def iter_claim_records(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as handle:
            for document in ijson.items(handle, "item"):
                groups = document.get("claims") or {}
                for group_name, claims in groups.items():
                    for item in claims or []:
                        label = str(item.get("label") or group_name).upper()
                        if label not in ANSWERABLE_LABELS:
                            continue
                        claim = str(item.get("claim") or "").strip()
                        if not claim:
                            continue
                        evidence = item.get("evidence") or []
                        relevant_docs: set[str] = set()
                        quotes_by_doc: defaultdict[str, list[str]] = defaultdict(list)
                        for entry in evidence:
                            doc_id = str(entry.get("article_id") or "").strip()
                            if not doc_id:
                                continue
                            relevant_docs.add(doc_id)
                            quotes = entry.get("quote") or []
                            if isinstance(quotes, str):
                                quotes = [quotes]
                            quotes_by_doc[doc_id].extend(
                                str(quote).strip() for quote in quotes if str(quote).strip()
                            )
                        if not relevant_docs:
                            continue
                        yield {
                            "claim": claim,
                            "label": label,
                            "source_doc_id": str(document.get("id") or ""),
                            "relevant_docs": sorted(relevant_docs),
                            "quotes_by_doc": dict(quotes_by_doc),
                        }


def reservoir_sample(
    records: Iterable[dict[str, Any]],
    available_docs: set[str],
    max_queries: int,
    seed: int,
) -> tuple[list[dict[str, Any]], int]:
    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    seen_claims: set[str] = set()
    eligible = 0
    for record in records:
        relevant = available_docs.intersection(record["relevant_docs"])
        if not relevant:
            continue
        record["relevant_docs"] = sorted(relevant)
        key = normalize_text(record["claim"])
        if key in seen_claims:
            continue
        seen_claims.add(key)
        eligible += 1
        if max_queries <= 0:
            sample.append(record)
        elif len(sample) < max_queries:
            sample.append(record)
        else:
            replacement = rng.randrange(eligible)
            if replacement < max_queries:
                sample[replacement] = record
    rng.shuffle(sample)
    return sample, eligible


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold()
    text = re.sub(r"\[\s*\d+\s*\]", " ", text)
    return " ".join(re.findall(r"(?u)\w+", text))


def token_coverage(quote: str, passage: str) -> float:
    quote_normalized = normalize_text(quote)
    passage_normalized = normalize_text(passage)
    if not quote_normalized:
        return 0.0
    if quote_normalized in passage_normalized:
        return 1.0
    quote_tokens = set(quote_normalized.split())
    passage_tokens = set(passage_normalized.split())
    if not quote_tokens:
        return 0.0
    return len(quote_tokens.intersection(passage_tokens)) / len(quote_tokens)


def derive_relevant_rows(
    record: dict[str, Any],
    retriever: HybridRetriever,
    doc_to_rows: dict[str, list[int]],
) -> set[int]:
    relevant_rows: set[int] = set()
    for doc_id in record["relevant_docs"]:
        candidate_rows = doc_to_rows.get(doc_id, [])
        if not candidate_rows:
            continue
        quotes = record["quotes_by_doc"].get(doc_id, [])
        if not quotes:
            continue
        for quote in quotes:
            scores = [
                (row_id, token_coverage(quote, str(retriever.records[row_id].get("text") or "")))
                for row_id in candidate_rows
            ]
            best_score = max((score for _, score in scores), default=0.0)
            if best_score <= 0.0:
                continue
            threshold = min(0.60, max(0.30, best_score - 0.05))
            relevant_rows.update(row_id for row_id, score in scores if score >= threshold)
    return relevant_rows


def unique_doc_ranking(results: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    ranking: list[str] = []
    for item in results:
        doc_id = str(item.get("doc_id") or "")
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        ranking.append(doc_id)
    return ranking


def ranking_metrics(ranking: list[Any], relevant: set[Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if not relevant:
        return metrics
    for k in METRIC_KS:
        hits = len(relevant.intersection(ranking[:k]))
        metrics[f"recall@{k}"] = hits / len(relevant)
        metrics[f"hit@{k}"] = float(hits > 0)

    first_rank = next((rank for rank, item in enumerate(ranking[:10], start=1) if item in relevant), None)
    metrics["mrr@10"] = 0.0 if first_rank is None else 1.0 / first_rank
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(ranking[:10], start=1)
        if item in relevant
    )
    ideal_count = min(len(relevant), 10)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    metrics["ndcg@10"] = dcg / ideal_dcg if ideal_dcg else 0.0
    return metrics


def compact_result(item: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "row_id",
        "chunk_id",
        "doc_id",
        "bm25_score",
        "bm25_rank",
        "dense_score",
        "dense_rank",
        "rrf_score",
        "rrf_rank",
        "rerank_score",
        "rerank_rank",
    )
    return {key: item[key] for key in keys if key in item}


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {key: float(np.mean([row[key] for row in rows if key in row])) for key in keys}


def evaluate(args: argparse.Namespace) -> list[dict[str, Any]]:
    retriever = HybridRetriever(
        args.index_dir,
        enable_reranker=True,
        reranker_batch_size=args.reranker_batch_size,
    )
    doc_to_rows: defaultdict[str, list[int]] = defaultdict(list)
    for row_id, item in enumerate(retriever.records):
        doc_to_rows[str(item.get("doc_id") or "")].append(row_id)

    claims, eligible_count = reservoir_sample(
        iter_claim_records(args.claims),
        available_docs=set(doc_to_rows),
        max_queries=args.max_queries,
        seed=args.seed,
    )
    if not claims:
        raise RuntimeError("No answerable claims with evidence found in the indexed corpus.")
    print(f"Evaluation sample: {len(claims):,} / {eligible_count:,} eligible claims")

    strategies = ("bm25", "dense", "hybrid", "hybrid_rerank")
    aggregate: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    latencies: dict[str, list[float]] = defaultdict(list)
    per_query_path = args.output_dir / "per_query.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with per_query_path.open("w", encoding="utf-8") as output:
        for record in tqdm(claims, desc="Evaluate", unit="claim"):
            claim = record["claim"]

            start = time.perf_counter()
            bm25 = retriever.search_bm25(claim, k=args.candidate_k)
            latencies["bm25"].append((time.perf_counter() - start) * 1000)

            start = time.perf_counter()
            dense = retriever.search_dense(claim, k=args.candidate_k)
            latencies["dense"].append((time.perf_counter() - start) * 1000)

            start = time.perf_counter()
            hybrid = retriever.fuse_rrf(
                [bm25, dense],
                rrf_k=args.rrf_k,
                top_k=args.fusion_k,
            )
            latencies["hybrid"].append(
                latencies["bm25"][-1]
                + latencies["dense"][-1]
                + (time.perf_counter() - start) * 1000
            )

            start = time.perf_counter()
            hybrid_rerank = retriever.rerank(claim, hybrid[: args.rerank_k])
            latencies["hybrid_rerank"].append(
                latencies["hybrid"][-1] + (time.perf_counter() - start) * 1000
            )

            rankings = {
                "bm25": bm25,
                "dense": dense,
                "hybrid": hybrid,
                "hybrid_rerank": hybrid_rerank,
            }
            relevant_docs = set(record["relevant_docs"])
            relevant_rows = derive_relevant_rows(record, retriever, doc_to_rows)
            detail: dict[str, Any] = {
                "query_id": hashlib.sha1(claim.encode("utf-8")).hexdigest()[:16],
                "claim": claim,
                "label": record["label"],
                "relevant_doc_ids": sorted(relevant_docs),
                "relevant_row_ids": sorted(relevant_rows),
                "rankings": {},
            }
            for strategy in strategies:
                results = rankings[strategy]
                doc_metrics = ranking_metrics(unique_doc_ranking(results), relevant_docs)
                aggregate[(strategy, "document")].append(doc_metrics)
                if relevant_rows:
                    row_ranking = [int(item["row_id"]) for item in results]
                    chunk_metrics = ranking_metrics(row_ranking, relevant_rows)
                    aggregate[(strategy, "chunk")].append(chunk_metrics)
                detail["rankings"][strategy] = [compact_result(item) for item in results[:10]]
            output.write(json.dumps(detail, ensure_ascii=False) + "\n")

    summary: list[dict[str, Any]] = []
    for strategy in strategies:
        for level in ("document", "chunk"):
            metric_rows = aggregate[(strategy, level)]
            if not metric_rows:
                continue
            row: dict[str, Any] = {
                "strategy": strategy,
                "level": level,
                "queries": len(metric_rows),
                **mean_metrics(metric_rows),
                "mean_latency_ms": float(np.mean(latencies[strategy])),
                "p50_latency_ms": float(np.percentile(latencies[strategy], 50)),
                "p95_latency_ms": float(np.percentile(latencies[strategy], 95)),
            }
            summary.append(row)

    fieldnames = ["strategy", "level", "queries"]
    fieldnames += [f"recall@{k}" for k in METRIC_KS]
    fieldnames += [f"hit@{k}" for k in METRIC_KS]
    fieldnames += ["mrr@10", "ndcg@10", "mean_latency_ms", "p50_latency_ms", "p95_latency_ms"]
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    evaluation_config = {
        "claims": [str(path) for path in args.claims],
        "eligible_claims": eligible_count,
        "sampled_claims": len(claims),
        "seed": args.seed,
        "candidate_k": args.candidate_k,
        "fusion_k": args.fusion_k,
        "rerank_k": args.rerank_k,
        "rrf_k": args.rrf_k,
        "reranker_batch_size": args.reranker_batch_size,
        "note": "Silver in-corpus evaluation from generated claims; not an external test set.",
    }
    (args.output_dir / "evaluation_config.json").write_text(
        json.dumps(evaluation_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    if args.max_queries < 0:
        raise ValueError("--max-queries must be >= 0; use 0 for all eligible claims.")
    if args.rerank_k > args.fusion_k:
        raise ValueError("--rerank-k cannot exceed --fusion-k.")
    summary = evaluate(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved evaluation to {args.output_dir}")


if __name__ == "__main__":
    main()

"""Evaluate the verifier with gold evidence and retrieved top-k evidence."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm.auto import tqdm

from verifier_model import EvidenceVerifier, evidence_context


LABELS = ("SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-gold-queries", type=int, default=0)
    parser.add_argument("--max-retrieved-queries", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrieval-top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--rerank-k", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--reranker-batch-size", type=int, default=8)
    return parser.parse_args()


def stable_key(example_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{example_id}".encode("utf-8")).hexdigest()


def stratified_sample(records: list[dict[str, Any]], maximum: int, seed: int) -> list[dict[str, Any]]:
    if maximum <= 0 or maximum >= len(records):
        return sorted(records, key=lambda row: stable_key(row["example_id"], seed))
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["label"]].append(record)
    selected: list[dict[str, Any]] = []
    base, remainder = divmod(maximum, len(LABELS))
    for index, label in enumerate(LABELS):
        rows = sorted(groups[label], key=lambda row: stable_key(row["example_id"], seed))
        selected.extend(rows[: base + (1 if index < remainder else 0)])
    return sorted(selected, key=lambda row: stable_key(row["example_id"], seed + 1))


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(LABELS))),
        target_names=list(LABELS),
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "per_label": {
            label: {
                key: float(report[label][key])
                for key in ("precision", "recall", "f1-score", "support")
            }
            for label in LABELS
        },
    }


def write_confusion(path: Path, y_true: list[int], y_pred: list[int]) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gold\\predicted", *LABELS])
        for label, row in zip(LABELS, matrix):
            writer.writerow([label, *[int(value) for value in row]])


def predict_records(
    model_id: str,
    records: list[dict[str, Any]],
    evidence_texts: list[str],
    max_length: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    verifier = EvidenceVerifier(model_id, max_length=max_length)
    predictions = verifier.predict_batch(
        [record["claim"] for record in records], evidence_texts, batch_size=batch_size
    )
    del verifier
    clear_cuda()
    return predictions


def compact_retrieval(item: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "row_id", "chunk_id", "doc_id", "title", "url", "text",
        "bm25_score", "dense_score", "rrf_score", "rerank_score",
    )
    return {key: item[key] for key in keys if key in item}


def collect_retrieved_contexts(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
) -> tuple[list[str], list[list[dict[str, Any]]], list[float]]:
    cache_path = args.output_dir / "retrieved_contexts.jsonl"
    config_path = args.output_dir / "retrieved_cache_config.json"
    expected_config = {
        "index_dir": str(args.index_dir),
        "seed": args.seed,
        "max_retrieved_queries": args.max_retrieved_queries,
        "retrieval_top_k": args.retrieval_top_k,
        "candidate_k": args.candidate_k,
        "rerank_k": args.rerank_k,
        "rrf_k": args.rrf_k,
    }
    cached: dict[str, dict[str, Any]] = {}
    if cache_path.is_file() and config_path.is_file():
        actual_config = json.loads(config_path.read_text(encoding="utf-8"))
        if actual_config == expected_config:
            with cache_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        cached[row["example_id"]] = row
        else:
            cache_path.unlink()
    config_path.write_text(json.dumps(expected_config, indent=2), encoding="utf-8")

    missing = [record for record in records if record["example_id"] not in cached]
    if missing:
        import sys

        retrieval_dir = Path(__file__).resolve().parents[1] / "retrieval"
        sys.path.insert(0, str(retrieval_dir))
        from hybrid_retriever import HybridRetriever

        retriever = HybridRetriever(
            args.index_dir,
            enable_reranker=True,
            reranker_batch_size=args.reranker_batch_size,
        )
        with cache_path.open("a", encoding="utf-8") as output:
            for record in tqdm(missing, desc="Retrieve test evidence", unit="claim"):
                started = time.perf_counter()
                results = retriever.retrieve(
                    record["claim"],
                    strategy="hybrid_rerank",
                    top_k=args.retrieval_top_k,
                    candidate_k=args.candidate_k,
                    rerank_k=args.rerank_k,
                    rrf_k=args.rrf_k,
                    max_per_doc=3,
                )
                row = {
                    "example_id": record["example_id"],
                    "context": evidence_context(results, max_items=args.retrieval_top_k),
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "retrieved": [compact_retrieval(item) for item in results],
                }
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                cached[row["example_id"]] = row
        del retriever
        clear_cuda()

    ordered = [cached[record["example_id"]] for record in records]
    return (
        [row["context"] for row in ordered],
        [row["retrieved"] for row in ordered],
        [float(row["latency_ms"]) for row in ordered],
    )


def write_predictions(
    path: Path,
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    retrieved: list[list[dict[str, Any]]] | None = None,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, (record, prediction) in enumerate(zip(records, predictions)):
            row = {
                "example_id": record["example_id"],
                "doc_id": record["doc_id"],
                "claim": record["claim"],
                "gold_label": record["label"],
                **prediction,
            }
            if retrieved is not None:
                row["retrieved_evidence"] = retrieved[index]
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.max_gold_queries < 0 or args.max_retrieved_queries < 0:
        raise ValueError("Maximum query counts must be >= 0; use 0 for all.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_path = args.data_dir / "test.parquet"
    if not test_path.is_file():
        raise FileNotFoundError(test_path)
    all_records = pq.read_table(test_path).to_pylist()
    gold_records = stratified_sample(all_records, args.max_gold_queries, args.seed)
    retrieved_records = stratified_sample(all_records, args.max_retrieved_queries, args.seed)

    gold_predictions = predict_records(
        args.model_id,
        gold_records,
        [record["evidence_text"] for record in gold_records],
        args.max_length,
        args.batch_size,
    )
    gold_true = [LABEL_TO_ID[record["label"]] for record in gold_records]
    gold_pred = [prediction["label_id"] for prediction in gold_predictions]
    gold_metrics = metrics(gold_true, gold_pred)
    write_predictions(args.output_dir / "gold_predictions.jsonl", gold_records, gold_predictions)
    write_confusion(args.output_dir / "gold_confusion_matrix.csv", gold_true, gold_pred)

    contexts, retrieved_evidence, retrieval_latencies = collect_retrieved_contexts(
        args, retrieved_records
    )
    retrieved_predictions = predict_records(
        args.model_id,
        retrieved_records,
        contexts,
        args.max_length,
        args.batch_size,
    )
    retrieved_true = [LABEL_TO_ID[record["label"]] for record in retrieved_records]
    retrieved_pred = [prediction["label_id"] for prediction in retrieved_predictions]
    retrieved_metrics = metrics(retrieved_true, retrieved_pred)
    retrieved_metrics["retrieval_mean_latency_ms"] = float(np.mean(retrieval_latencies))
    retrieved_metrics["retrieval_p95_latency_ms"] = float(
        np.percentile(retrieval_latencies, 95)
    )
    write_predictions(
        args.output_dir / "retrieved_predictions.jsonl",
        retrieved_records,
        retrieved_predictions,
        retrieved_evidence,
    )
    write_confusion(
        args.output_dir / "retrieved_confusion_matrix.csv", retrieved_true, retrieved_pred
    )

    summary = {
        "model_id": args.model_id,
        "test_examples_available": len(all_records),
        "gold_evidence": {"queries": len(gold_records), **gold_metrics},
        "retrieved_evidence": {"queries": len(retrieved_records), **retrieved_metrics},
        "retrieval": {
            "strategy": "hybrid_rerank",
            "top_k": args.retrieval_top_k,
            "candidate_k": args.candidate_k,
            "rerank_k": args.rerank_k,
            "rrf_k": args.rrf_k,
        },
        "note": "Silver claims; document-disjoint verifier split. Retrieved mode is end-to-end.",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

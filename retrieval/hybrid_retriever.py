"""Hybrid BM25 + multilingual-E5 retrieval with optional BGE reranking."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

# Kaggle may preinstall an incompatible TensorFlow/Keras stack; this pipeline is PyTorch-only.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import bm25s
import faiss
import numpy as np
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_E5_MODEL = "intfloat/multilingual-e5-large"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
BM25_TOKEN_PATTERN = r"(?u)\b\w+\b"


class HybridRetriever:
    """Shared row-ID retrieval over exact FAISS and BM25S indexes."""

    def __init__(
        self,
        index_dir: str | Path,
        e5_model_id: str = DEFAULT_E5_MODEL,
        reranker_model_id: str = DEFAULT_RERANKER_MODEL,
        device: str | None = None,
        enable_reranker: bool = True,
        reranker_batch_size: int = 16,
        reranker_max_length: int = 512,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.reranker_batch_size = reranker_batch_size
        self.reranker_max_length = reranker_max_length
        self._validate_files(enable_reranker)

        self.manifest = json.loads(
            (self.index_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if self.manifest.get("status") != "complete":
            raise RuntimeError("Index manifest is not marked complete.")

        self.faiss_index = faiss.read_index(str(self.index_dir / "faiss.index"))
        metadata = pq.read_table(self.index_dir / "metadata.parquet")
        self.records: list[dict[str, Any]] = metadata.to_pylist()
        if self.faiss_index.ntotal != len(self.records):
            raise RuntimeError(
                f"FAISS rows ({self.faiss_index.ntotal}) != metadata rows ({len(self.records)})."
            )
        for expected, record in enumerate(self.records):
            if int(record["row_id"]) != expected:
                raise RuntimeError(f"metadata row_id is not positional at row {expected}.")

        self.bm25 = bm25s.BM25.load(
            str(self.index_dir / "bm25"),
            load_corpus=False,
            load_vocab=True,
            mmap=True,
        )
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.e5 = SentenceTransformer(
            e5_model_id,
            device=self.device,
            model_kwargs={"torch_dtype": dtype},
            trust_remote_code=False,
        )
        self.e5.max_seq_length = 512
        self.e5.eval()

        self.reranker_tokenizer = None
        self.reranker_model = None
        if enable_reranker:
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(
                reranker_model_id,
                trust_remote_code=False,
            )
            self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
                reranker_model_id,
                torch_dtype=dtype,
                trust_remote_code=False,
            ).to(self.device)
            self.reranker_model.eval()

        if self.device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True

    def _validate_files(self, enable_reranker: bool) -> None:
        required = [
            self.index_dir / "manifest.json",
            self.index_dir / "faiss.index",
            self.index_dir / "metadata.parquet",
            self.index_dir / "bm25" / "params.index.json",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing retrieval index files: {missing}")
        if enable_reranker and not torch.cuda.is_available():
            print("Warning: reranker is enabled without CUDA and will be slow.")

    def _result(self, row_id: int, **scores: Any) -> dict[str, Any]:
        result = dict(self.records[row_id])
        result.update(scores)
        return result

    def search_dense(self, claim: str, k: int = 100) -> list[dict[str, Any]]:
        query = self.e5.encode(
            ["query: " + claim.strip()],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32, copy=False)
        scores, row_ids = self.faiss_index.search(
            np.ascontiguousarray(query),
            min(k, self.faiss_index.ntotal),
        )
        return [
            self._result(
                int(row_id),
                dense_score=float(score),
                dense_rank=rank,
            )
            for rank, (row_id, score) in enumerate(zip(row_ids[0], scores[0]), start=1)
            if row_id >= 0
        ]

    def search_bm25(self, claim: str, k: int = 100) -> list[dict[str, Any]]:
        tokens = bm25s.tokenize(
            [claim],
            stopwords=None,
            stemmer=None,
            token_pattern=BM25_TOKEN_PATTERN,
            return_ids=False,
            show_progress=False,
        )
        try:
            row_ids, scores = self.bm25.retrieve(
                tokens,
                k=min(k, len(self.records)),
                show_progress=False,
            )
        except ValueError as exc:
            if "does not contain any tokens" not in str(exc):
                raise
            return []
        return [
            self._result(
                int(row_id),
                bm25_score=float(score),
                bm25_rank=rank,
            )
            for rank, (row_id, score) in enumerate(zip(row_ids[0], scores[0]), start=1)
            if row_id >= 0
        ]

    def fuse_rrf(
        self,
        rankings: Iterable[list[dict[str, Any]]],
        rrf_k: int = 60,
        top_k: int = 100,
    ) -> list[dict[str, Any]]:
        if rrf_k < 0:
            raise ValueError("rrf_k must be non-negative.")
        combined: dict[int, dict[str, Any]] = {}
        rrf_scores: defaultdict[int, float] = defaultdict(float)
        for ranking in rankings:
            for rank, item in enumerate(ranking, start=1):
                row_id = int(item["row_id"])
                rrf_scores[row_id] += 1.0 / (rrf_k + rank)
                if row_id not in combined:
                    combined[row_id] = dict(item)
                else:
                    combined[row_id].update(
                        {
                            key: value
                            for key, value in item.items()
                            if key.endswith("_score") or key.endswith("_rank")
                        }
                    )
        ordered_ids = sorted(rrf_scores, key=lambda row_id: (-rrf_scores[row_id], row_id))
        results: list[dict[str, Any]] = []
        for rank, row_id in enumerate(ordered_ids[:top_k], start=1):
            item = combined[row_id]
            item["rrf_score"] = float(rrf_scores[row_id])
            item["rrf_rank"] = rank
            results.append(item)
        return results

    def score_pairs(self, pairs: list[list[str]]) -> np.ndarray:
        if self.reranker_model is None or self.reranker_tokenizer is None:
            raise RuntimeError("Reranker was not enabled when HybridRetriever was created.")
        all_scores: list[np.ndarray] = []
        for start in range(0, len(pairs), self.reranker_batch_size):
            batch = pairs[start : start + self.reranker_batch_size]
            inputs = self.reranker_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.reranker_max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                logits = self.reranker_model(**inputs, return_dict=True).logits
            all_scores.append(logits.view(-1).float().cpu().numpy())
        if not all_scores:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(all_scores).astype(np.float32, copy=False)

    def rerank(
        self,
        claim: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        pairs = [[claim, item["text"]] for item in candidates]
        scores = self.score_pairs(pairs)
        rescored = []
        for item, score in zip(candidates, scores):
            copy = dict(item)
            copy["rerank_score"] = float(score)
            rescored.append(copy)
        rescored.sort(key=lambda item: (-item["rerank_score"], int(item["row_id"])))
        for rank, item in enumerate(rescored, start=1):
            item["rerank_rank"] = rank
        return rescored

    def rankings(
        self,
        claim: str,
        candidate_k: int = 100,
        fusion_k: int = 100,
        rerank_k: int = 50,
        rrf_k: int = 60,
        include_reranker: bool = True,
    ) -> dict[str, list[dict[str, Any]]]:
        bm25 = self.search_bm25(claim, k=candidate_k)
        dense = self.search_dense(claim, k=candidate_k)
        hybrid = self.fuse_rrf([bm25, dense], rrf_k=rrf_k, top_k=fusion_k)
        outputs = {"bm25": bm25, "dense": dense, "hybrid": hybrid}
        if include_reranker:
            outputs["hybrid_rerank"] = self.rerank(claim, hybrid[:rerank_k])
        return outputs

    @staticmethod
    def limit_per_document(
        results: list[dict[str, Any]],
        top_k: int,
        max_per_doc: int | None,
    ) -> list[dict[str, Any]]:
        if max_per_doc is None:
            return results[:top_k]
        counts: defaultdict[str, int] = defaultdict(int)
        selected = []
        for item in results:
            doc_id = str(item.get("doc_id", ""))
            if counts[doc_id] >= max_per_doc:
                continue
            selected.append(item)
            counts[doc_id] += 1
            if len(selected) == top_k:
                break
        return selected

    def retrieve(
        self,
        claim: str,
        strategy: str = "hybrid_rerank",
        top_k: int = 10,
        candidate_k: int = 100,
        rerank_k: int = 50,
        rrf_k: int = 60,
        max_per_doc: int | None = 3,
    ) -> list[dict[str, Any]]:
        allowed = {"bm25", "dense", "hybrid", "hybrid_rerank"}
        if strategy not in allowed:
            raise ValueError(f"strategy must be one of {sorted(allowed)}")
        all_rankings = self.rankings(
            claim,
            candidate_k=candidate_k,
            fusion_k=max(candidate_k, rerank_k, top_k),
            rerank_k=rerank_k,
            rrf_k=rrf_k,
            include_reranker=strategy == "hybrid_rerank",
        )
        return self.limit_per_document(all_rankings[strategy], top_k, max_per_doc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("claim")
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument(
        "--strategy",
        choices=["bm25", "dense", "hybrid", "hybrid_rerank"],
        default="hybrid_rerank",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--rerank-k", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retriever = HybridRetriever(
        args.index_dir,
        enable_reranker=args.strategy == "hybrid_rerank",
    )
    results = retriever.retrieve(
        args.claim,
        strategy=args.strategy,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        rerank_k=args.rerank_k,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

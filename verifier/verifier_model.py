"""Inference utilities for the three-label evidence verifier."""

from __future__ import annotations

import re
from typing import Any, Iterable

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABELS = ("SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO")


def evidence_context(results: Iterable[dict[str, Any]], max_items: int = 5) -> str:
    sections: list[str] = []
    for rank, item in enumerate(results, start=1):
        if rank > max_items:
            break
        title = re.sub(r"\s+", " ", str(item.get("title") or "")).strip()
        text = re.sub(r"\s+", " ", str(item.get("text") or "")).strip()
        if not text:
            continue
        prefix = f"[BẰNG CHỨNG {rank}]"
        if title:
            prefix += f" {title}."
        sections.append(f"{prefix} {text}")
    return "\n\n".join(sections)


class EvidenceVerifier:
    """XLM-R sequence classifier over an evidence-premise and claim-hypothesis pair."""

    def __init__(
        self,
        model_id_or_path: str,
        device: str | None = None,
        max_length: int = 384,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=True)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id_or_path,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()
        if self.device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True

    @torch.inference_mode()
    def predict_batch(
        self,
        claims: list[str],
        evidence_texts: list[str],
        batch_size: int = 32,
    ) -> list[dict[str, Any]]:
        if len(claims) != len(evidence_texts):
            raise ValueError("claims and evidence_texts must have the same length")
        outputs: list[dict[str, Any]] = []
        id2label = {int(key): value for key, value in self.model.config.id2label.items()}
        for start in range(0, len(claims), batch_size):
            claim_batch = claims[start : start + batch_size]
            evidence_batch = evidence_texts[start : start + batch_size]
            encoded = self.tokenizer(
                evidence_batch,
                claim_batch,
                padding=True,
                truncation="only_first",
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**encoded).logits.float()
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            for row in probabilities:
                label_id = int(np.argmax(row))
                outputs.append(
                    {
                        "label": id2label[label_id],
                        "label_id": label_id,
                        "confidence": float(row[label_id]),
                        "probabilities": {
                            id2label[index]: float(probability)
                            for index, probability in enumerate(row)
                        },
                    }
                )
        return outputs

    def predict(self, claim: str, evidence_text: str) -> dict[str, Any]:
        return self.predict_batch([claim], [evidence_text], batch_size=1)[0]

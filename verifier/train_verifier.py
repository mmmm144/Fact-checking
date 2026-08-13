"""Fine-tune XLM-R for Vietnamese evidence-based fact verification."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


LABEL_TO_ID = {"SUPPORTED": 0, "REFUTED": 1, "NOT_ENOUGH_INFO": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--base-model", default="FacebookAI/xlm-roberta-base")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-repo", required=True)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-validation-samples", type=int, default=0)
    parser.add_argument("--private-output", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def compute_metrics(eval_prediction) -> dict[str, float]:
    predictions = np.argmax(eval_prediction.predictions, axis=-1)
    labels = eval_prediction.label_ids
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }


def model_is_complete(args: argparse.Namespace, token: str | None) -> bool:
    if args.force:
        return False
    try:
        path = hf_hub_download(
            repo_id=args.output_repo,
            filename="verifier_config.json",
            repo_type="model",
            token=token,
        )
        config = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return False
    source_manifest = json.loads(
        (args.data_dir / "manifest.json").read_text(encoding="utf-8")
    )
    expected = {
        "status": "complete",
        "base_model": args.base_model,
        "max_length": args.max_length,
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "max_train_samples": args.max_train_samples,
        "max_validation_samples": args.max_validation_samples,
        "source_claims_repo": source_manifest["claims_repo"],
        "source_claims_revision": source_manifest["claims_revision"],
    }
    return all(config.get(key) == value for key, value in expected.items())


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if model_is_complete(args, token):
        print(f"Reusing trained verifier: https://huggingface.co/{args.output_repo}")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for verifier fine-tuning on Kaggle.")

    set_seed(args.seed)
    files = {
        "train": str(args.data_dir / "train.parquet"),
        "validation": str(args.data_dir / "validation.parquet"),
    }
    for path in files.values():
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    dataset = load_dataset("parquet", data_files=files)
    if args.max_train_samples > 0:
        dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(
            range(min(args.max_train_samples, len(dataset["train"])))
        )
    if args.max_validation_samples > 0:
        dataset["validation"] = dataset["validation"].shuffle(seed=args.seed).select(
            range(min(args.max_validation_samples, len(dataset["validation"])))
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, token=token)

    def tokenize(batch):
        encoded = tokenizer(
            batch["evidence_text"],
            batch["claim"],
            truncation="only_first",
            max_length=args.max_length,
        )
        encoded["labels"] = batch["label_id"]
        return encoded

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenize evidence-claim pairs",
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        token=token,
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    checkpoint_dir = args.output_dir / "checkpoints"
    model_dir = args.output_dir / "model"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        weight_decay=0.01,
        warmup_ratio=0.06,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=True,
        tf32=False,
        dataloader_num_workers=2,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    checkpoint = get_last_checkpoint(str(checkpoint_dir))
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    validation_metrics = trainer.evaluate()
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(model_dir)

    source_manifest = json.loads(
        (args.data_dir / "manifest.json").read_text(encoding="utf-8")
    )
    verifier_config = {
        "status": "complete",
        "base_model": args.base_model,
        "max_length": args.max_length,
        "label_to_id": LABEL_TO_ID,
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "max_train_samples": args.max_train_samples,
        "max_validation_samples": args.max_validation_samples,
        "train_examples": len(dataset["train"]),
        "validation_examples": len(dataset["validation"]),
        "source_claims_repo": source_manifest["claims_repo"],
        "source_claims_revision": source_manifest["claims_revision"],
        "validation_metrics": validation_metrics,
        "train_metrics": train_result.metrics,
    }
    (model_dir / "verifier_config.json").write_text(
        json.dumps(verifier_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    card = f"""---
language: vi
license: mit
base_model: {args.base_model}
pipeline_tag: text-classification
tags:
- fact-checking
- natural-language-inference
---

# Vietnamese fact-checking verifier (XLM-R base)

Input order is `(evidence, claim)`. Labels: `SUPPORTED`, `REFUTED`, and
`NOT_ENOUGH_INFO`. Training/validation/test are split by source document.

Validation Macro-F1: {validation_metrics.get('eval_macro_f1', float('nan')):.4f}.

This model was trained on synthetic/silver claims and must not be treated as a
human-validated factuality oracle.
"""
    (model_dir / "README.md").write_text(card, encoding="utf-8")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.output_repo,
        repo_type="model",
        private=args.private_output,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=args.output_repo,
        repo_type="model",
        folder_path=model_dir,
        commit_message="Upload XLM-R Vietnamese fact-checking verifier",
    )
    print(json.dumps(validation_metrics, indent=2))
    print(f"Uploaded verifier: https://huggingface.co/{args.output_repo}")


if __name__ == "__main__":
    main()

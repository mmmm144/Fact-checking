import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_CHUNKED_PATH = Path("data/vie/processed/fact_checking_dataset_chunked.json")
DEFAULT_CLAIMS_PATH = Path("data/vie/processed/newdata.json")
DEFAULT_OUTPUT_DIR = Path("reports/pilot")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def text_has_mojibake(text: str) -> bool:
    markers = ("Ă", "Ä", "áº", "á»", "Â", "Æ")
    return any(marker in text for marker in markers)


def iter_text_fields(item):
    for key, value in item.items():
        if isinstance(value, str):
            yield key, value
        elif isinstance(value, list):
            for entry in value:
                if isinstance(entry, str):
                    yield key, entry
                elif isinstance(entry, dict):
                    yield from iter_text_fields(entry)
        elif isinstance(value, dict):
            yield from iter_text_fields(value)


def audit_chunked_dataset(records):
    label_counts = Counter(record.get("label", "") for record in records)
    source_counts = Counter(record.get("source_type", "") for record in records)
    missing_counts = Counter()
    chunk_counts = []
    chunk_word_counts = []
    text_field_count = 0
    mojibake_count = 0

    for record in records:
        for key in ("id", "claim", "label", "url", "original_text", "evidence", "justification", "chunks"):
            value = record.get(key)
            if value in (None, "", [], {}):
                missing_counts[key] += 1

        chunks = record.get("chunks") or []
        chunk_counts.append(len(chunks))
        for chunk in chunks:
            chunk_word_counts.append(chunk.get("word_count") or len((chunk.get("text") or "").split()))

        for _, value in iter_text_fields(record):
            text_field_count += 1
            if text_has_mojibake(value):
                mojibake_count += 1

    return {
        "records": len(records),
        "label_counts": dict(label_counts),
        "source_type_counts": dict(source_counts),
        "total_chunks": sum(chunk_counts),
        "records_without_chunks": sum(1 for count in chunk_counts if count == 0),
        "avg_chunks_per_record": round(sum(chunk_counts) / max(len(chunk_counts), 1), 2),
        "avg_chunk_words": round(sum(chunk_word_counts) / max(len(chunk_word_counts), 1), 2),
        "missing_counts": dict(missing_counts),
        "text_fields_checked": text_field_count,
        "mojibake_like_fields": mojibake_count,
    }


def flatten_claims(records):
    examples = []
    for article in records:
        article_id = article.get("id")
        for label, claims in (article.get("claims") or {}).items():
            for claim_item in claims or []:
                evidence_items = claim_item.get("evidence") or []
                evidence_quotes = []
                evidence_urls = []
                for evidence in evidence_items:
                    evidence_quotes.extend(evidence.get("quote") or [])
                    if evidence.get("url"):
                        evidence_urls.append(evidence["url"])
                examples.append(
                    {
                        "article_id": article_id,
                        "claim": claim_item.get("claim", ""),
                        "label": claim_item.get("label") or label,
                        "gold_evidence": evidence_quotes,
                        "reason": claim_item.get("reason", ""),
                        "source_urls": sorted(set(evidence_urls)),
                    }
                )
    return examples


def audit_claim_dataset(records, flattened_claims):
    label_counts = Counter(example["label"] for example in flattened_claims)
    missing_counts = Counter()
    text_field_count = 0
    mojibake_count = 0

    for article in records:
        for key in ("id", "date_iso", "full_text", "claims"):
            value = article.get(key)
            if value in (None, "", [], {}):
                missing_counts[key] += 1
        for _, value in iter_text_fields(article):
            text_field_count += 1
            if text_has_mojibake(value):
                mojibake_count += 1

    return {
        "articles": len(records),
        "claims": len(flattened_claims),
        "label_counts": dict(label_counts),
        "missing_counts": dict(missing_counts),
        "text_fields_checked": text_field_count,
        "mojibake_like_fields": mojibake_count,
    }


def make_demo_examples(flattened_claims, limit: int, seed: int):
    grouped = defaultdict(list)
    for example in flattened_claims:
        grouped[example["label"]].append(example)

    random.seed(seed)
    selected = []
    labels = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]
    while len(selected) < limit and any(grouped.values()):
        for label in labels:
            if grouped[label] and len(selected) < limit:
                selected.append(random.choice(grouped[label]))
    return selected


def write_jsonl(path: Path, records):
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_markdown_report(path: Path, chunked_audit, claim_audit, demo_examples):
    lines = [
        "# Pilot Data Audit - RAG Fact-Checking Tiếng Việt",
        "",
        "## Mục tiêu",
        "Báo cáo này kiểm tra nhanh hai file dữ liệu pilot để chứng minh hướng RAG fact-checking đa bước khả thi trước khi mở rộng dataset.",
        "",
        "## Evidence Corpus",
        f"- File: `{DEFAULT_CHUNKED_PATH.as_posix()}`",
        f"- Số bản ghi: `{chunked_audit['records']}`",
        f"- Phân bố nhãn: `{chunked_audit['label_counts']}`",
        f"- Tổng số chunks: `{chunked_audit['total_chunks']}`",
        f"- Trung bình chunks/bản ghi: `{chunked_audit['avg_chunks_per_record']}`",
        f"- Trung bình số từ/chunk: `{chunked_audit['avg_chunk_words']}`",
        f"- Bản ghi chưa có chunks: `{chunked_audit['records_without_chunks']}`",
        f"- Field thiếu đáng chú ý: `{chunked_audit['missing_counts']}`",
        f"- Text fields nghi lỗi encoding: `{chunked_audit['mojibake_like_fields']}/{chunked_audit['text_fields_checked']}`",
        "",
        "## Claim Evaluation Set",
        f"- File: `{DEFAULT_CLAIMS_PATH.as_posix()}`",
        f"- Số bài gốc: `{claim_audit['articles']}`",
        f"- Số claim: `{claim_audit['claims']}`",
        f"- Phân bố nhãn: `{claim_audit['label_counts']}`",
        f"- Field thiếu đáng chú ý: `{claim_audit['missing_counts']}`",
        f"- Text fields nghi lỗi encoding: `{claim_audit['mojibake_like_fields']}/{claim_audit['text_fields_checked']}`",
        "",
        "## Pipeline Pilot Đề Xuất",
        "1. Nhập claim tiếng Việt cần kiểm chứng.",
        "2. Truy xuất top-k evidence chunks từ corpus bằng BM25/TF-IDF hoặc embedding.",
        "3. Verifier đọc claim + evidence và dự đoán SUPPORTED, REFUTED hoặc NOT_ENOUGH_INFO.",
        "4. Sinh giải thích ngắn và trích dẫn evidence liên quan.",
        "5. Đánh giá retrieval bằng Recall@k/MRR và đánh giá verdict bằng Accuracy/Macro-F1.",
        "",
        "## Ví Dụ Demo",
    ]

    for index, example in enumerate(demo_examples, start=1):
        evidence = " | ".join(example["gold_evidence"][:2])
        lines.extend(
            [
                f"### Ví dụ {index}",
                f"- Claim: {example['claim']}",
                f"- Gold label: `{example['label']}`",
                f"- Gold evidence: {evidence}",
                f"- Reason: {example['reason']}",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Audit pilot datasets for Vietnamese RAG fact-checking.")
    parser.add_argument("--chunked", type=Path, default=DEFAULT_CHUNKED_PATH)
    parser.add_argument("--claims", type=Path, default=DEFAULT_CLAIMS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--demo-size", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    chunked_records = load_json(args.chunked)
    claim_records = load_json(args.claims)
    flattened_claims = flatten_claims(claim_records)
    demo_examples = make_demo_examples(flattened_claims, args.demo_size, args.seed)

    chunked_audit = audit_chunked_dataset(chunked_records)
    claim_audit = audit_claim_dataset(claim_records, flattened_claims)

    audit = {
        "evidence_corpus": chunked_audit,
        "claim_evaluation_set": claim_audit,
        "recommendation": "Dataset hiện tại phù hợp làm pilot/prototype. Cần mở rộng và làm sạch encoding trước khi huấn luyện/đánh giá nghiêm túc.",
    }

    (args.output_dir / "audit_summary.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(args.output_dir / "claim_eval_pilot.jsonl", flattened_claims)
    write_jsonl(args.output_dir / "demo_examples.jsonl", demo_examples)
    write_markdown_report(args.output_dir / "pilot_report.md", chunked_audit, claim_audit, demo_examples)

    print(f"Wrote audit files to {args.output_dir}")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

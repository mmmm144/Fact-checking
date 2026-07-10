"""
Fact-checking claim generation script.
Reads fact_checking_dataset_chunked.json, calls LLM to generate claims,
validates output JSON, and saves results to newdata.json.

Features:
- Auto-retry on invalid JSON response from model
- Resume support: skips already-processed articles if newdata.json exists
- Structured JSON validation against expected schema
- Debug logging for invalid responses -> debug_invalid_json.log
- Failed article IDs saved to failed_ids.json
- Fallback to justification if original_text is too short
- Progress logging
"""

import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

# ─── Configuration ───────────────────────────────────────────────────────────
BASE_URL = "https://api.xah.io/v1"
MODEL = "claude-opus-4.6"
INPUT_FILE = os.path.join("data", "vie", "processed", "fact_checking_dataset_chunked.json")
OUTPUT_FILE = "newdata.json"
DEBUG_LOG_FILE = "debug_invalid_json.log"
FAILED_IDS_FILE = "failed_ids.json"
MAX_RETRIES = 5
RETRY_DELAY_BASE = 3  # seconds, exponential backoff
MIN_TEXT_LENGTH = 50  # If original_text shorter than this, use justification

# ─── System Prompt ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Bạn là một chuyên gia dữ liệu và kiểm chứng thông tin (Fact-checker). Nhiệm vụ của bạn là đọc nội dung bài viết tôi cung cấp và tự động sinh ra các nhận định (claims) thuộc 3 loại: SUPPORTED (Đúng), REFUTED (Sai) và NOT_ENOUGH_INFO (Không đủ thông tin).

QUY TẮC TUYỆT ĐỐI:
- Chỉ trả về DUY NHẤT một JSON object hợp lệ.
- KHÔNG viết bất kỳ text nào trước hoặc sau JSON.
- KHÔNG bao quanh bằng ```json``` hay bất kỳ markdown nào.
- KHÔNG giải thích, KHÔNG chào hỏi, KHÔNG thêm ghi chú.
- Response của bạn phải bắt đầu bằng ký tự { và kết thúc bằng ký tự }.

YÊU CẦU LÕI:
1. SUPPORTED: Sinh ra 2 claim phản ánh chính xác thông tin có trong bài viết.
2. REFUTED: Sinh ra 2 claim cung cấp thông tin sai lệch, trái ngược hoàn toàn với chi tiết trong bài viết.
3. NOT_ENOUGH_INFO: Sinh ra 2 claim có vẻ liên quan đến chủ đề bài viết nhưng KHÔNG THỂ tìm thấy bằng chứng xác nhận hay bác bỏ trong nội dung bài.
4. Mọi bằng chứng (evidence -> quote) bắt buộc phải trích dẫn Y NGUYÊN TỪNG CHỮ từ bài viết gốc.

ĐỊNH DẠNG ĐẦU RA (OUTPUT FORMAT):
Bạn CHỈ ĐƯỢC PHÉP trả về duy nhất một chuỗi JSON hợp lệ, tuyệt đối không giải thích thêm, không dùng markdown ```json...``` bao quanh. Cấu trúc JSON đầu ra bắt buộc phải tuân theo format mẫu sau:
{
"id": "[Lấy từ id của dữ liệu đầu vào]",
"date_iso": "[Lấy từ publish_date của dữ liệu đầu vào, thêm T00:00:00 nếu cần]",
"media": [],
"full_text": "[ORIGINAL_TEXT]",
"claims": {
"SUPPORTED": [
{
"claim": "[Nội dung claim 1]",
"label": "SUPPORTED",
"evidence": [
{
"type": "text",
"quote": ["[Câu trích dẫn chính xác từ bài]"],
"article_id": "[id]",
"url": "[url từ dữ liệu đầu vào]"
}
],
"reason": "[Giải thích lý do đánh giá supported]",
"image": ""
},
{
"claim": "[Nội dung claim 2]",
"label": "SUPPORTED",
"evidence": [
{
"type": "text",
"quote": ["[Câu trích dẫn chính xác từ bài]"],
"article_id": "[id]",
"url": "[url từ dữ liệu đầu vào]"
}
],
"reason": "[Giải thích lý do đánh giá supported]",
"image": ""
}
],
"REFUTED": [
{
"claim": "[Nội dung claim sai 1]",
"label": "REFUTED",
"evidence": [
{
"type": "text",
"quote": ["[Câu trích dẫn từ bài chứng minh claim này sai]"],
"article_id": "[id]",
"url": "[url từ dữ liệu đầu vào]"
}
],
"reason": "[Giải thích tại sao claim này sai so với bài viết]",
"image": ""
},
{
"claim": "[Nội dung claim sai 2]",
"label": "REFUTED",
"evidence": [
{
"type": "text",
"quote": ["[Câu trích dẫn từ bài chứng minh claim này sai]"],
"article_id": "[id]",
"url": "[url từ dữ liệu đầu vào]"
}
],
"reason": "[Giải thích tại sao claim này sai so với bài viết]",
"image": ""
}
],
"NOT_ENOUGH_INFO": [
{
"claim": "[Nội dung claim thiếu thông tin 1]",
"label": "NOT_ENOUGH_INFO",
"evidence": [
{
"type": "text",
"quote": ["[Câu trích dẫn ngữ cảnh liên quan (nếu có)]"],
"article_id": "[id]",
"url": "[url từ dữ liệu đầu vào]"
}
],
"reason": "[Giải thích lý do văn bản không đủ thông tin để xác minh]",
"image": ""
},
{
"claim": "[Nội dung claim thiếu thông tin 2]",
"label": "NOT_ENOUGH_INFO",
"evidence": [
{
"type": "text",
"quote": ["[Câu trích dẫn ngữ cảnh liên quan (nếu có)]"],
"article_id": "[id]",
"url": "[url từ dữ liệu đầu vào]"
}
],
"reason": "[Giải thích lý do văn bản không đủ thông tin để xác minh]",
"image": ""
}
]
}
}"""

# ─── OpenAI Client ───────────────────────────────────────────────────────────
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", ""),
    base_url=BASE_URL,
)


# ─── Helper Functions ────────────────────────────────────────────────────────

def load_dataset(path: str) -> list[dict]:
    """Load the chunked dataset JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    # Handle wrapped formats
    if isinstance(data, dict):
        for key in ["data", "items", "articles", "records", "dataset"]:
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError(f"Cannot parse dataset structure from {path}")


def load_existing_results(path: str) -> dict[str, dict]:
    """Load already-processed results for resume support."""
    if not Path(path).exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            results = json.load(f)
        if isinstance(results, list):
            return {item["id"]: item for item in results if "id" in item}
    except (json.JSONDecodeError, KeyError):
        pass
    return {}


def save_results(results: list[dict], path: str):
    """Save results to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def normalize_date(date_value) -> str:
    """Ensure date has T00:00:00 suffix."""
    if not date_value:
        return ""
    s = str(date_value).strip()
    if "T" in s:
        return s
    return s + "T00:00:00"


def get_article_content(article: dict) -> str:
    """Get the best available text content from article.
    Falls back to justification if original_text is too short."""
    original = article.get("original_text", "") or ""
    justification = article.get("justification", "") or ""

    # If original_text is too short, combine with justification
    if len(original) < MIN_TEXT_LENGTH and len(justification) > len(original):
        return justification
    # If both exist and original is short, combine them
    if len(original) < MIN_TEXT_LENGTH:
        combined = f"{original}\n\n{justification}".strip()
        return combined if combined else original
    return original


def build_user_prompt(article: dict) -> str:
    """Build the user message with article data for the model."""
    content = get_article_content(article)
    input_data = {
        "id": article.get("id", ""),
        "url": article.get("url", ""),
        "publish_date": article.get("publish_date", ""),
        "original_text": content,
    }
    return (
        "DỮ LIỆU ĐẦU VÀO CẦN XỬ LÝ:\n"
        + json.dumps(input_data, ensure_ascii=False)
        + "\n\nTrả về JSON hợp lệ duy nhất, bắt đầu bằng { và kết thúc bằng }."
    )


def log_debug(article_id: str, attempt: int, raw_content: str, error_msg: str):
    """Log invalid responses to debug file for inspection."""
    with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Article ID: {article_id} | Attempt: {attempt}\n")
        f.write(f"Error: {error_msg}\n")
        f.write(f"Raw response ({len(raw_content)} chars):\n")
        # Log first 2000 chars to avoid huge log files
        f.write(raw_content[:2000])
        if len(raw_content) > 2000:
            f.write(f"\n... [truncated, total {len(raw_content)} chars]")
        f.write(f"\n{'='*80}\n")


def extract_json_from_response(content: str) -> dict:
    """
    Try to parse JSON from model response.
    Handles markdown wrapping, BOM, leading text, trailing text.
    """
    if not content or not content.strip():
        raise json.JSONDecodeError("Empty response from model", "", 0)

    content = content.strip()

    # Remove BOM if present
    if content.startswith("\ufeff"):
        content = content[1:]

    # Remove markdown code fences if present
    content = re.sub(r"^```(?:json)?\s*\n?", "", content)
    content = re.sub(r"\n?```\s*$", "", content)
    content = content.strip()

    # Try direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object: match first { to last }
    first_brace = content.find("{")
    last_brace = content.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = content[first_brace:last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # Try fixing common issues: trailing commas before } or ]
        fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        # Try fixing unescaped newlines in strings
        fixed2 = candidate.replace("\n", "\\n")
        try:
            return json.loads(fixed2)
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError(
        "Cannot extract valid JSON from response",
        content[:200] if len(content) > 200 else content,
        0
    )


def validate_output_schema(data: dict, article_id: str) -> list[str]:
    """
    Validate that the output matches expected schema.
    Returns list of validation errors (empty = valid).
    """
    errors = []

    # Check top-level keys
    required_top = ["id", "date_iso", "full_text", "claims"]
    for key in required_top:
        if key not in data:
            errors.append(f"Missing top-level key: {key}")

    if "claims" not in data:
        return errors

    claims = data["claims"]
    required_labels = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]
    for label in required_labels:
        if label not in claims:
            errors.append(f"Missing claims category: {label}")
            continue
        claim_list = claims[label]
        if not isinstance(claim_list, list):
            errors.append(f"claims.{label} is not a list")
            continue
        if len(claim_list) < 2:
            errors.append(f"claims.{label} has {len(claim_list)} items, expected 2")
            continue
        for i, claim_item in enumerate(claim_list):
            if "claim" not in claim_item:
                errors.append(f"claims.{label}[{i}] missing 'claim'")
            if "evidence" not in claim_item:
                errors.append(f"claims.{label}[{i}] missing 'evidence'")
            elif isinstance(claim_item["evidence"], list):
                for j, ev in enumerate(claim_item["evidence"]):
                    if "quote" not in ev:
                        errors.append(f"claims.{label}[{i}].evidence[{j}] missing 'quote'")
            if "reason" not in claim_item:
                errors.append(f"claims.{label}[{i}] missing 'reason'")

    return errors


def call_model_with_validation(article: dict) -> dict:
    """
    Call the model, validate JSON output, and retry if invalid.
    Returns validated result dict.
    """
    user_prompt = build_user_prompt(article)
    article_id = article.get("id", "unknown")
    result = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=16384,
            )
            content = resp.choices[0].message.content or ""

            # Parse JSON
            result = extract_json_from_response(content)

            # Validate schema
            validation_errors = validate_output_schema(result, article_id)
            if validation_errors:
                error_msg = "; ".join(validation_errors)
                print(f"  [Attempt {attempt}/{MAX_RETRIES}] Schema validation failed: {error_msg}")
                log_debug(article_id, attempt, content, f"Schema: {error_msg}")
                if attempt < MAX_RETRIES:
                    print(f"  Retrying in {RETRY_DELAY_BASE * attempt}s...")
                    time.sleep(RETRY_DELAY_BASE * attempt)
                    continue
                else:
                    print(f"  [WARNING] Accepting partial result for {article_id}")
                    break

            # Ensure consistent id/date/full_text from source
            result["id"] = article_id
            result["date_iso"] = normalize_date(article.get("publish_date"))
            result["full_text"] = get_article_content(article)
            if "media" not in result:
                result["media"] = []

            return result

        except json.JSONDecodeError as e:
            raw = content if 'content' in dir() else "(no content)"
            print(f"  [Attempt {attempt}/{MAX_RETRIES}] Invalid JSON: {e}")
            log_debug(article_id, attempt, raw if isinstance(raw, str) else "", str(e))
            if attempt < MAX_RETRIES:
                print(f"  Retrying in {RETRY_DELAY_BASE * attempt}s...")
                time.sleep(RETRY_DELAY_BASE * attempt)
            else:
                raise RuntimeError(
                    f"Failed to get valid JSON for article {article_id} after {MAX_RETRIES} attempts"
                )

        except Exception as e:
            print(f"  [Attempt {attempt}/{MAX_RETRIES}] API error: {type(e).__name__}: {e}")
            log_debug(article_id, attempt, "", f"API error: {type(e).__name__}: {e}")
            if attempt < MAX_RETRIES:
                print(f"  Retrying in {RETRY_DELAY_BASE * attempt}s...")
                time.sleep(RETRY_DELAY_BASE * attempt)
            else:
                raise RuntimeError(
                    f"API call failed for article {article_id} after {MAX_RETRIES} attempts: {e}"
                )

    # Fallback: ensure fields are set even for partial results
    if result is None:
        result = {}
    result["id"] = article_id
    result["date_iso"] = normalize_date(article.get("publish_date"))
    result["full_text"] = get_article_content(article)
    if "media" not in result:
        result["media"] = []
    return result


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY environment variable first.")
        sys.exit(1)

    # Load dataset
    print(f"Loading dataset from {INPUT_FILE}...")
    articles = load_dataset(INPUT_FILE)
    print(f"Total articles: {len(articles)}")

    # Load existing results for resume
    existing = load_existing_results(OUTPUT_FILE)
    if existing:
        print(f"Found {len(existing)} already-processed articles. Resuming...")

    results = list(existing.values())
    processed_ids = set(existing.keys())

    # Process each article
    failed_ids = []
    total = len(articles)
    skipped = 0

    for idx, article in enumerate(articles, start=1):
        article_id = article.get("id", f"unknown_{idx}")

        # Skip if already processed
        if article_id in processed_ids:
            skipped += 1
            continue

        print(f"[{idx}/{total}] Processing: {article_id[:24]}...")

        try:
            result = call_model_with_validation(article)
            results.append(result)
            processed_ids.add(article_id)
            print(f"  Done ({len(results)} total saved)")
        except RuntimeError as e:
            print(f"  FAILED: {e}")
            failed_ids.append({"id": article_id, "error": str(e)})
            continue

        # Save intermediate results every 10 articles
        if len(results) % 10 == 0:
            save_results(results, OUTPUT_FILE)
            print(f"  [Checkpoint] Saved {len(results)} results to {OUTPUT_FILE}")

        # Small delay to avoid rate limits
        time.sleep(1)

    # Final save
    save_results(results, OUTPUT_FILE)

    # Save failed IDs
    if failed_ids:
        with open(FAILED_IDS_FILE, "w", encoding="utf-8") as f:
            json.dump(failed_ids, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"COMPLETE!")
    print(f"  Total articles: {total}")
    print(f"  Skipped (already done): {skipped}")
    print(f"  Processed this run: {len(results) - len(existing)}")
    print(f"  Total saved: {len(results)}")
    print(f"  Output: {OUTPUT_FILE}")
    if failed_ids:
        print(f"  FAILED: {len(failed_ids)} articles (see {FAILED_IDS_FILE})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

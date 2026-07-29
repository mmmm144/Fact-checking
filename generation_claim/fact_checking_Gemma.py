"""
Fact-checking claim generation script for Vietnamese Evidence Corpus v1.0.
Reads corpus_v1.json, calls Qwen2.5 7B Instruct through local Ollama,
validates JSON and saves results separately from API-generated results.

Features:
- Auto-retry on invalid JSON response from model
- Resume support: skips articles already present in the output file
- Structured JSON validation against expected schema
- Debug logging for invalid responses -> debug_invalid_json.log
- Failed article IDs saved to failed_ids.json
- Fallback to justification if original_text is too short
- Progress logging
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


# ─── Configuration ───────────────────────────────────────────────────────────
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")
API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api/chat")
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_FILE = PROJECT_ROOT / "data" / "vie" / "raw" / "viet-fact-checking" / "corpus_v1.json"
OUTPUT_FILE = SCRIPT_DIR / "claims_corpus_v1_qwen2_5_7b_instruct.json"
DEBUG_LOG_FILE = SCRIPT_DIR / "debug_invalid_json_qwen2_5_7b_instruct.log"
FAILED_IDS_FILE = SCRIPT_DIR / "failed_ids_qwen2_5_7b_instruct.json"
MAX_RETRIES = 5
RETRY_DELAY_BASE = 3  # seconds, exponential backoff
DEFAULT_REQUEST_DELAY = 0.0
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
"reason": "[Giải thích lý do đánh giá supported]"
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
"reason": "[Giải thích lý do đánh giá supported]"
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
"reason": "[Giải thích tại sao claim này sai so với bài viết]"
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
"reason": "[Giải thích tại sao claim này sai so với bài viết]"
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
"reason": "[Giải thích lý do văn bản không đủ thông tin để xác minh]"
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
"reason": "[Giải thích lý do văn bản không đủ thông tin để xác minh]"
}
]
}
}"""

# ─── Local Ollama Client ────────────────────────────────────────────────────

class OllamaAPIError(RuntimeError):
    """HTTP/API error returned by the local Ollama server."""

    def __init__(self, status_code: int, message: str):
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code


def get_ollama_tags_url() -> str:
    """Build Ollama's model-list endpoint from the configured chat URL."""
    if "/api/" in API_URL:
        return API_URL.split("/api/", 1)[0] + "/api/tags"
    return API_URL.rstrip("/") + "/api/tags"


def get_installed_ollama_models() -> set[str]:
    """Return model tags installed in the local Ollama instance."""
    request = urllib.request.Request(get_ollama_tags_url(), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            response_body = response.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as error:
        raise RuntimeError(
            "Không kết nối được Ollama tại http://localhost:11434. "
            "Hãy mở ứng dụng Ollama hoặc chạy: ollama serve"
        ) from error

    try:
        response_data = json.loads(response_body)
        return {
            str(item.get("name") or item.get("model") or "")
            for item in response_data.get("models", [])
        }
    except (json.JSONDecodeError, AttributeError, TypeError) as error:
        raise RuntimeError(
            f"Ollama trả về danh sách model không hợp lệ: {response_body[:500]}"
        ) from error


def ensure_ollama_ready() -> None:
    """Fail early with a useful command if the configured model is missing."""
    installed_models = get_installed_ollama_models()
    if MODEL not in installed_models:
        raise ModelUnavailableError(
            f"Chưa cài model {MODEL}. Hãy chạy: ollama pull {MODEL}"
        )


def call_ollama_api(user_prompt: str) -> str:
    """Call Qwen through Ollama's local chat API with JSON output enabled."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": "json",
        "keep_alive": "30m",
        "options": {
            "temperature": 0.2,
            "num_ctx": 32768,
            "num_predict": 8192,
        },
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            response_body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        response_body = error.read().decode("utf-8", errors="replace")
        try:
            error_data = json.loads(response_body)
            message = str(error_data.get("error") or error_data)
        except json.JSONDecodeError:
            message = response_body or str(error)
        raise OllamaAPIError(error.code, message) from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            "Mất kết nối với Ollama. Hãy chắc chắn ứng dụng Ollama đang chạy."
        ) from error

    try:
        response_data = json.loads(response_body)
        content = response_data["message"]["content"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError(
            f"Ollama trả về response không đúng định dạng: {response_body[:1000]}"
        ) from error

    return str(content or "")


def is_model_unavailable_error(error: Exception) -> bool:
    """Return True when Ollama reports that the requested model is missing."""
    message = f"{type(error).__name__}: {error}".lower()
    return (
        isinstance(error, OllamaAPIError) and error.status_code == 404
    ) or ("model" in message and "not found" in message)


class ModelUnavailableError(RuntimeError):
    """Raised when the configured Ollama model is not installed."""

# ─── Helper Functions ────────────────────────────────────────────────────────

def configure_console_encoding() -> None:
    """Use UTF-8 output on Windows consoles with a limited code page."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def load_dataset(path: str | Path) -> list[dict]:
    """Load a JSON dataset stored as a list or a common wrapped structure."""
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


def normalize_corpus_article(record: dict, index: int) -> dict:
    """Map a corpus_v1 record to the legacy fields used by the claim prompt."""
    article_id = str(record.get("doc_id") or record.get("id") or f"unknown_{index}")
    text = str(record.get("text") or record.get("original_text") or "").strip()
    summary = str(record.get("summary") or record.get("justification") or "").strip()

    return {
        **record,
        "id": article_id,
        "original_text": text,
        "justification": summary,
    }


def load_existing_results(path: str | Path) -> dict[str, dict]:
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


def save_results(results: list[dict], path: str | Path):
    """Save results to JSON file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for result in results:
        result.pop("media", None)
        for claim_items in (result.get("claims") or {}).values():
            if isinstance(claim_items, list):
                for claim_item in claim_items:
                    if isinstance(claim_item, dict):
                        claim_item.pop("image", None)
    with output_path.open("w", encoding="utf-8") as f:
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
            content = call_ollama_api(user_prompt)

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
            print(f"  [Attempt {attempt}/{MAX_RETRIES}] Ollama error: {type(e).__name__}: {e}")
            log_debug(article_id, attempt, "", f"Ollama error: {type(e).__name__}: {e}")

            if is_model_unavailable_error(e):
                raise ModelUnavailableError(
                    f"Model {MODEL} chưa được cài. Hãy chạy: ollama pull {MODEL}"
                ) from e

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
    return result


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    configure_console_encoding()

    parser = argparse.ArgumentParser(
        description="Generate fact-checking claims from Vietnamese Evidence Corpus v1.0."
    )
    parser.add_argument("--input-file", type=Path, default=INPUT_FILE)
    parser.add_argument("--output-file", type=Path, default=OUTPUT_FILE)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N corpus records (recommended for a test run).",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=DEFAULT_REQUEST_DELAY,
        help=f"Seconds to wait between successful requests (default: {DEFAULT_REQUEST_DELAY}).",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than 0")
    if args.request_delay < 0:
        parser.error("--request-delay cannot be negative")

    # Check that Ollama is running and the requested local model is installed.
    try:
        ensure_ollama_ready()
    except RuntimeError as error:
        print(f"ERROR: {error}")
        sys.exit(1)
    print(f"Ollama model: {MODEL}")
    # Load dataset
    if not args.input_file.is_file():
        print(f"ERROR: Input dataset not found: {args.input_file}")
        sys.exit(1)

    print(f"Loading dataset from {args.input_file}...")
    raw_records = load_dataset(args.input_file)
    if args.limit is not None:
        raw_records = raw_records[:args.limit]
    articles = [
        normalize_corpus_article(record, index)
        for index, record in enumerate(raw_records, start=1)
    ]
    print(f"Total articles: {len(articles)}")

    # Load existing results for resume
    existing = load_existing_results(args.output_file)
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
        except KeyboardInterrupt:
            print("\n  Đã nhận Ctrl+C. Đang lưu toàn bộ kết quả đã hoàn thành...")
            break
        except ModelUnavailableError as e:
            print(f"  STOPPED: {e}")
            print("  No more articles will be attempted with this unavailable model.")
            break
        except RuntimeError as e:
            print(f"  FAILED: {e}")
            failed_ids.append({"id": article_id, "error": str(e)})
            continue

        # Save intermediate results every 10 articles
        if len(results) % 10 == 0:
            save_results(results, args.output_file)
            print(f"  [Checkpoint] Saved {len(results)} results to {args.output_file}")

        # Optional pause between local generations.
        time.sleep(args.request_delay)

    # Final save
    save_results(results, args.output_file)

    # Save failed IDs
    if failed_ids:
        with open(FAILED_IDS_FILE, "w", encoding="utf-8") as f:
            json.dump(failed_ids, f, ensure_ascii=False, indent=2)
    else:
        FAILED_IDS_FILE.write_text("[]\n", encoding="utf-8")

    print(f"\n{'='*60}")
    print("COMPLETE OR SAFELY STOPPED")
    print(f"  Total articles: {total}")
    print(f"  Skipped (already done): {skipped}")
    print(f"  Processed this run: {len(results) - len(existing)}")
    print(f"  Total saved: {len(results)}")
    print(f"  Output: {args.output_file}")
    if failed_ids:
        print(f"  FAILED: {len(failed_ids)} articles (see {FAILED_IDS_FILE})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

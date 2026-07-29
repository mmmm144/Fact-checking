"""
Fact-checking claim generation script for Vietnamese Evidence Corpus v1.0.
Reads corpus_v1.json and calls Qwen3.5 through a vLLM OpenAI-compatible server.
validates JSON and saves results separately from API-generated results.

Features:
- Auto-retry on invalid JSON response from model
- Resume and sharding support for time-limited Kaggle sessions
- Strict JSON and verbatim-evidence validation against the source article
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
MODEL = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4")
API_URL = os.environ.get("VLLM_API_URL", "http://127.0.0.1:8000/v1/chat/completions")
API_KEY = os.environ.get("VLLM_API_KEY", "")
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
IS_KAGGLE = Path("/kaggle/working").is_dir()
WORK_DIR = Path("/kaggle/working") if IS_KAGGLE else SCRIPT_DIR
INPUT_FILE = PROJECT_ROOT / "data" / "vie" / "raw" / "viet-fact-checking" / "corpus_v1.json"
OUTPUT_FILE = WORK_DIR / "claims_corpus_v1_qwen3_5_35b_a3b_gptq_int4.json"
DEBUG_LOG_FILE = WORK_DIR / "debug_invalid_json_qwen3_5_35b_a3b_gptq_int4.log"
FAILED_IDS_FILE = WORK_DIR / "failed_ids_qwen3_5_35b_a3b_gptq_int4.json"
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "5"))
RETRY_DELAY_BASE = 3
DEFAULT_REQUEST_DELAY = 0.0
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "3072"))
MAX_INPUT_CHARS = int(os.environ.get("MAX_INPUT_CHARS", "16000"))
REQUEST_TIMEOUT = int(os.environ.get("VLLM_REQUEST_TIMEOUT", "1800"))
MIN_TEXT_LENGTH = 50

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
5. Mỗi claim phải đơn nhất, cụ thể, tự đứng độc lập và không gộp nhiều sự kiện không liên quan.
6. Mỗi quote phải là 1-2 câu ngắn nhất đủ để kiểm chứng claim; tuyệt đối không diễn giải lại quote.
7. REFUTED phải thay đổi một chi tiết cốt lõi có thể bác bỏ trực tiếp bằng quote, không tạo câu sai vô nghĩa.
8. NOT_ENOUGH_INFO phải nêu rõ thông tin cụ thể nào còn thiếu; quote chỉ cung cấp ngữ cảnh liên quan.
9. Chính tả tên riêng, con số, ngày tháng và đơn vị trong claim phải được kiểm tra kỹ theo bài gốc.

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

# ─── vLLM OpenAI-compatible Client ───────────────────────────────────────────

class ModelAPIError(RuntimeError):
    """HTTP/API error returned by the vLLM server."""
    def __init__(self, status_code: int, message: str):
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code


def get_models_url() -> str:
    if "/v1/" in API_URL:
        return API_URL.split("/v1/", 1)[0] + "/v1/models"
    return API_URL.rstrip("/") + "/v1/models"


def get_served_models() -> set[str]:
    headers = {"Accept": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    request = urllib.request.Request(get_models_url(), headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response_body = response.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Không kết nối được vLLM tại {get_models_url()}. Hãy khởi động vllm serve trước."
        ) from error
    try:
        response_data = json.loads(response_body)
        return {str(item.get("id") or "") for item in response_data.get("data", []) if isinstance(item, dict)}
    except (json.JSONDecodeError, AttributeError, TypeError) as error:
        raise RuntimeError(f"vLLM trả về danh sách model không hợp lệ: {response_body[:500]}") from error


def ensure_server_ready() -> None:
    served_models = get_served_models()
    if MODEL not in served_models:
        raise ModelUnavailableError(
            f"vLLM chưa phục vụ model {MODEL}. Models hiện có: {sorted(served_models)}"
        )


def call_model_api(user_prompt: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "temperature": 0.15,
        "top_p": 0.9,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "seed": 42,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            response_body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        response_body = error.read().decode("utf-8", errors="replace")
        try:
            error_data = json.loads(response_body)
            detail = error_data.get("error") or error_data
            message = detail.get("message") if isinstance(detail, dict) else str(detail)
        except json.JSONDecodeError:
            message = response_body or str(error)
        raise ModelAPIError(error.code, str(message)) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Mất kết nối với vLLM tại {API_URL}. Kiểm tra vllm.log.") from error
    try:
        response_data = json.loads(response_body)
        content = response_data["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"vLLM trả về response không đúng định dạng: {response_body[:1000]}") from error
    return str(content or "")


def is_model_unavailable_error(error: Exception) -> bool:
    message = f"{type(error).__name__}: {error}".lower()
    return isinstance(error, ModelAPIError) and error.status_code == 404 and "model" in message


class ModelUnavailableError(RuntimeError):
    """Raised when the configured model is not served by vLLM."""

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
    """Get the complete source text used for validation and output."""
    original = str(article.get("original_text", "") or "").strip()
    justification = str(article.get("justification", "") or "").strip()
    if len(original) < MIN_TEXT_LENGTH and len(justification) > len(original):
        return justification
    if len(original) < MIN_TEXT_LENGTH:
        return f"{original}\n\n{justification}".strip()
    return original


def get_prompt_content(article: dict) -> tuple[str, bool]:
    """Fit unusually long articles into the constrained Kaggle context window."""
    content = get_article_content(article)
    if MAX_INPUT_CHARS <= 0 or len(content) <= MAX_INPUT_CHARS:
        return content, False
    tail_chars = max(2000, MAX_INPUT_CHARS // 4)
    head_chars = MAX_INPUT_CHARS - tail_chars
    omitted = len(content) - MAX_INPUT_CHARS
    clipped = (
        content[:head_chars]
        + f"\n\n[ĐÃ LƯỢC {omitted} KÝ TỰ Ở GIỮA DO GIỚI HẠN CONTEXT]\n\n"
        + content[-tail_chars:]
    )
    return clipped, True

def build_user_prompt(article: dict) -> str:
    """Build the user message with article data for the model."""
    content, was_truncated = get_prompt_content(article)
    input_data = {
        "id": article.get("id", ""),
        "url": article.get("url", ""),
        "publish_date": article.get("publish_date", ""),
        "original_text": content,
        "source_truncated_for_context": was_truncated,
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


def normalize_result_metadata(data: dict, article: dict) -> None:
    """Force provenance fields to match the input article."""
    article_id = str(article.get("id", ""))
    url = str(article.get("url", "") or "")
    claims = data.get("claims") if isinstance(data, dict) else None
    if not isinstance(claims, dict):
        return
    for label, items in claims.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            item["label"] = label
            evidence = item.get("evidence")
            if not isinstance(evidence, list):
                continue
            for ev in evidence:
                if isinstance(ev, dict):
                    ev["type"] = "text"
                    ev["article_id"] = article_id
                    ev["url"] = url


def validate_output_schema(data: dict, article: dict) -> list[str]:
    """Validate schema and require every evidence quote to be verbatim."""
    errors: list[str] = []
    article_id = str(article.get("id", ""))
    source_text = get_article_content(article)
    if not isinstance(data, dict):
        return ["Output root is not a JSON object"]
    for key in ["id", "date_iso", "full_text", "claims"]:
        if key not in data:
            errors.append(f"Missing top-level key: {key}")
    claims = data.get("claims")
    if not isinstance(claims, dict):
        errors.append("claims is not an object")
        return errors

    for label in ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]:
        claim_list = claims.get(label)
        if not isinstance(claim_list, list):
            errors.append(f"claims.{label} is not a list")
            continue
        if len(claim_list) != 2:
            errors.append(f"claims.{label} has {len(claim_list)} items, expected exactly 2")
        for i, claim_item in enumerate(claim_list):
            path = f"claims.{label}[{i}]"
            if not isinstance(claim_item, dict):
                errors.append(f"{path} is not an object")
                continue
            claim = claim_item.get("claim")
            if not isinstance(claim, str) or len(claim.strip()) < 10:
                errors.append(f"{path}.claim is empty or too short")
            if claim_item.get("label") != label:
                errors.append(f"{path}.label must be {label}")
            reason = claim_item.get("reason")
            if not isinstance(reason, str) or len(reason.strip()) < 15:
                errors.append(f"{path}.reason is empty or too short")
            evidence = claim_item.get("evidence")
            if not isinstance(evidence, list) or not evidence:
                errors.append(f"{path}.evidence must be a non-empty list")
                continue
            valid_quote_count = 0
            for j, ev in enumerate(evidence):
                ev_path = f"{path}.evidence[{j}]"
                if not isinstance(ev, dict):
                    errors.append(f"{ev_path} is not an object")
                    continue
                quotes = ev.get("quote")
                if not isinstance(quotes, list) or not quotes:
                    errors.append(f"{ev_path}.quote must be a non-empty list")
                    continue
                for k, quote in enumerate(quotes):
                    if not isinstance(quote, str) or not quote.strip():
                        errors.append(f"{ev_path}.quote[{k}] is empty")
                    elif quote not in source_text:
                        errors.append(f"{ev_path}.quote[{k}] is not verbatim in article {article_id}")
                    else:
                        valid_quote_count += 1
            if valid_quote_count == 0:
                errors.append(f"{path} has no valid verbatim evidence quote")
    return errors

def call_model_with_validation(article: dict) -> dict:
    """Call the model and retry until strict schema/evidence validation passes."""
    base_prompt = build_user_prompt(article)
    user_prompt = base_prompt
    article_id = str(article.get("id", "unknown"))
    for attempt in range(1, MAX_RETRIES + 1):
        content = ""
        try:
            content = call_model_api(user_prompt)
            result = extract_json_from_response(content)
            normalize_result_metadata(result, article)
            validation_errors = validate_output_schema(result, article)
            if validation_errors:
                error_msg = "; ".join(validation_errors)
                print(f"  [Attempt {attempt}/{MAX_RETRIES}] Validation failed: {error_msg}")
                log_debug(article_id, attempt, content, f"Validation: {error_msg}")
                if attempt == MAX_RETRIES:
                    raise RuntimeError(f"Strict validation failed for article {article_id}: {error_msg}")
                user_prompt = (
                    base_prompt
                    + "\n\nLẦN TRƯỚC KHÔNG HỢP LỆ. Tạo lại toàn bộ JSON và sửa:\n- "
                    + "\n- ".join(validation_errors[:12])
                    + "\nMọi quote phải sao chép nguyên văn từ original_text."
                )
                time.sleep(RETRY_DELAY_BASE * attempt)
                continue
            result["id"] = article_id
            result["date_iso"] = normalize_date(article.get("publish_date"))
            result["full_text"] = get_article_content(article)
            return result
        except json.JSONDecodeError as error:
            print(f"  [Attempt {attempt}/{MAX_RETRIES}] Invalid JSON: {error}")
            log_debug(article_id, attempt, content, str(error))
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Failed to get valid JSON for article {article_id} after {MAX_RETRIES} attempts"
                ) from error
            user_prompt = base_prompt + "\n\nLần trước JSON bị lỗi. Chỉ trả về đúng một JSON object hợp lệ."
            time.sleep(RETRY_DELAY_BASE * attempt)
        except ModelUnavailableError:
            raise
        except RuntimeError as error:
            if str(error).startswith("Strict validation failed"):
                raise
            print(f"  [Attempt {attempt}/{MAX_RETRIES}] vLLM error: {type(error).__name__}: {error}")
            log_debug(article_id, attempt, content, f"vLLM error: {type(error).__name__}: {error}")
            if is_model_unavailable_error(error):
                raise ModelUnavailableError(f"Model {MODEL} không được vLLM phục vụ") from error
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"API call failed for article {article_id} after {MAX_RETRIES} attempts: {error}"
                ) from error
            time.sleep(RETRY_DELAY_BASE * attempt)
    raise RuntimeError(f"Unexpected retry termination for article {article_id}")

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    global DEBUG_LOG_FILE, FAILED_IDS_FILE
    configure_console_encoding()
    parser = argparse.ArgumentParser(
        description="Generate Vietnamese fact-checking claims with Qwen3.5 served by vLLM."
    )
    parser.add_argument("--input-file", type=Path, default=INPUT_FILE)
    parser.add_argument("--output-file", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--start-index", type=int, default=0,
                        help="Zero-based index used to split work across Kaggle sessions.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process N records after --start-index.")
    parser.add_argument("--checkpoint-every", type=int, default=1,
                        help="Save every N new articles (default: 1).")
    parser.add_argument("--request-delay", type=float, default=DEFAULT_REQUEST_DELAY)
    parser.add_argument("--debug-log-file", type=Path, default=DEBUG_LOG_FILE)
    parser.add_argument("--failed-ids-file", type=Path, default=FAILED_IDS_FILE)
    args = parser.parse_args()

    if args.start_index < 0:
        parser.error("--start-index cannot be negative")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than 0")
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be greater than 0")
    if args.request_delay < 0:
        parser.error("--request-delay cannot be negative")

    DEBUG_LOG_FILE = args.debug_log_file
    FAILED_IDS_FILE = args.failed_ids_file
    for path in (args.output_file, DEBUG_LOG_FILE, FAILED_IDS_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)

    try:
        ensure_server_ready()
    except RuntimeError as error:
        print(f"ERROR: {error}")
        sys.exit(1)
    print(f"vLLM model: {MODEL}")
    print(f"API: {API_URL}")

    if not args.input_file.is_file():
        print(f"ERROR: Input dataset not found: {args.input_file}")
        sys.exit(1)

    print(f"Loading dataset from {args.input_file}...")
    all_records = load_dataset(args.input_file)
    raw_records = all_records[args.start_index:]
    if args.limit is not None:
        raw_records = raw_records[:args.limit]
    articles = [
        normalize_corpus_article(record, args.start_index + index)
        for index, record in enumerate(raw_records, start=1)
    ]
    print(
        f"Dataset: {len(all_records)} | shard start: {args.start_index} | "
        f"articles this shard: {len(articles)}"
    )

    existing = load_existing_results(args.output_file)
    if existing:
        print(f"Found {len(existing)} already-processed articles. Resuming...")
    results = list(existing.values())
    processed_ids = set(existing.keys())
    failed_ids: list[dict] = []
    total = len(articles)
    skipped = 0
    processed_this_run = 0

    for idx, article in enumerate(articles, start=1):
        article_id = str(article.get("id", f"unknown_{idx}"))
        if article_id in processed_ids:
            skipped += 1
            continue
        _, was_truncated = get_prompt_content(article)
        truncation_note = " [context clipped]" if was_truncated else ""
        print(f"[{idx}/{total}] Processing: {article_id[:24]}...{truncation_note}")
        try:
            result = call_model_with_validation(article)
            results.append(result)
            processed_ids.add(article_id)
            processed_this_run += 1
            print(f"  Done ({len(results)} total saved)")
        except KeyboardInterrupt:
            print("\n  Đã nhận Ctrl+C. Đang lưu toàn bộ kết quả đã hoàn thành...")
            break
        except ModelUnavailableError as error:
            print(f"  STOPPED: {error}")
            break
        except RuntimeError as error:
            print(f"  FAILED: {error}")
            failed_ids.append({"id": article_id, "error": str(error)})
            FAILED_IDS_FILE.write_text(
                json.dumps(failed_ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            continue

        if processed_this_run % args.checkpoint_every == 0:
            save_results(results, args.output_file)
            print(f"  [Checkpoint] Saved {len(results)} results to {args.output_file}")
        time.sleep(args.request_delay)

    save_results(results, args.output_file)
    FAILED_IDS_FILE.write_text(
        json.dumps(failed_ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n{'='*60}")
    print("COMPLETE OR SAFELY STOPPED")
    print(f"  Articles in shard: {total}")
    print(f"  Skipped (already done): {skipped}")
    print(f"  Processed this run: {processed_this_run}")
    print(f"  Total saved: {len(results)}")
    print(f"  Output: {args.output_file}")
    if failed_ids:
        print(f"  FAILED: {len(failed_ids)} articles (see {FAILED_IDS_FILE})")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

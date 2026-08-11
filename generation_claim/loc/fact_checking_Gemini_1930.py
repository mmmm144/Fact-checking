"""
Fact-checking claim generation script for Vietnamese Evidence Corpus v1.0.
Reads corpus_v1.json, calls Gemini 3.6 Flash through OpenRouter,
validates JSON and saves resumable results.

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
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
OPENROUTER_MODEL = os.environ.get(
    "OPENROUTER_MODEL", "google/gemini-3.5-flash-lite"
)
GEMINI_API_URL = os.environ.get(
    "GEMINI_API_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
)
OPENROUTER_API_URL = os.environ.get(
    "OPENROUTER_API_URL",
    "https://openrouter.ai/api/v1/chat/completions",
)
# GEMINI_API_KEY_1930 and OPENROUTER_API_KEY are read from the environment/.env.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ENV_FILE = PROJECT_ROOT / ".env"
INPUT_FILE = PROJECT_ROOT / "data" / "vie" / "raw" / "viet-fact-checking" / "corpus_v1.json"
OUTPUT_FILE = SCRIPT_DIR / "claims_corpus_v1_gemini_3_5_flash_1930.json"
DEBUG_LOG_FILE = SCRIPT_DIR / "debug_invalid_json_gemini_3_5_flash_1930.log"
FAILED_IDS_FILE = SCRIPT_DIR / "failed_ids_gemini_3_5_flash_1930.json"
MAX_RETRIES = 5
RETRY_DELAY_BASE = 3  # seconds, exponential backoff
DEFAULT_REQUEST_DELAY = 1.0
MAX_AUTO_RATE_LIMIT_WAIT = 120.0
MIN_TEXT_LENGTH = 50  # If original_text shorter than this, use justification
DEFAULT_START_INDEX = 1930  # One-based corpus record number

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
3. NOT_ENOUGH_INFO: Sinh ra 2 claim bám trực tiếp vào một chủ thể, thực thể hoặc sự kiện thực sự xuất hiện trong bài, nhưng chứa đúng một thuộc tính quan trọng mà toàn bộ bài viết không đủ bằng chứng để xác nhận hoặc bác bỏ.
4. Mọi bằng chứng (evidence -> quote) bắt buộc phải trích dẫn Y NGUYÊN TỪNG CHỮ từ bài viết gốc.

YÊU CẦU VỀ CHẤT LƯỢNG CLAIM:
- Mỗi claim phải là đúng một câu trần thuật hoàn chỉnh và ưu tiên trong khoảng 40-50 từ; đây là mục tiêu mềm, không phải điều kiện bắt buộc.
- Claim phải tự đủ nghĩa khi đứng độc lập: nêu rõ chủ thể/thực thể và sự việc; thêm thời gian, địa điểm, đại lượng hoặc phạm vi nếu bài viết có các chi tiết đó.
- KHÔNG viết claim dạng tiêu đề, cụm từ rút gọn hoặc câu dùng đại từ mơ hồ như “điều này”, “nơi đây”, “họ” mà không nêu rõ đối tượng.
- Vẫn giữ claim đơn nhất (atomic): chỉ chứa một thông tin chính có thể kiểm chứng, không ghép nhiều nhận định không liên quan để kéo dài câu.
- KHÔNG bổ sung thời gian, nguyên nhân, mục đích, kết quả, địa điểm, đại lượng hoặc bất kỳ chi tiết nào không có trong nguồn chỉ để đạt độ dài mong muốn.
- Tính atomic, khả năng kiểm chứng và độ trung thành với nguồn quan trọng hơn số từ.
- Không sao chép nguyên một câu quá dài chỉ để đạt số từ; hãy diễn đạt tự nhiên, chính xác và cụ thể.

RÀNG BUỘC RIÊNG CHO NOT_ENOUGH_INFO:
- Claim phải giữ nguyên ít nhất một chủ thể, thực thể hoặc sự kiện được nêu rõ trong bài; không được chỉ liên quan chung về chủ đề.
- Chỉ bổ sung đúng một thuộc tính quan trọng còn thiếu trong bài, chẳng hạn một con số, thời điểm, nguyên nhân hoặc kết quả chưa được nêu.
- Thuộc tính bổ sung không được trái ngược với bất kỳ thông tin nào đã có trong bài.
- Sau khi xét TOÀN BỘ bài viết, claim phải không thể được xác nhận và cũng không thể bị bác bỏ.
- Không đưa vào claim thực thể hoặc sự kiện mới không xuất hiện trong bài, và không suy diễn sang vấn đề rộng hơn.

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

# ─── OpenRouter Client ──────────────────────────────────────────────────────

def load_api_key(variable_name: str) -> str:
    """Read an API key from the environment or project .env file."""
    value = os.environ.get(variable_name)
    if value:
        return value.strip()

    if ENV_FILE.is_file():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith("#") or "=" not in stripped_line:
                continue
            key, value = stripped_line.split("=", 1)
            if key.strip() == variable_name:
                api_key = value.strip().strip('"').strip("'")
                if api_key:
                    return api_key
    return ""


def load_gemini_api_key() -> str:
    return load_api_key("GEMINI_API_KEY_1930")


def load_openrouter_api_key() -> str:
    return load_api_key("OPENROUTER_API_KEY")


class ProviderAPIError(RuntimeError):
    """HTTP/API error returned by Gemini or OpenRouter."""

    def __init__(self, provider: str, status_code: int, message: str, headers=None):
        super().__init__(f"{provider} HTTP {status_code}: {message}")
        self.provider = provider
        self.status_code = status_code
        self.headers = headers


def call_provider_api(
    user_prompt: str,
    *,
    provider: str,
    api_url: str,
    api_key: str,
    model: str,
) -> str:
    """Call an OpenAI-compatible chat-completions endpoint."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 16384,
        "response_format": {"type": "json_object"},
    }
    if provider == "Gemini":
        payload["reasoning_effort"] = "minimal"
    else:
        payload["reasoning"] = {"effort": "minimal", "exclude": True}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if provider == "OpenRouter":
        headers.update(
            {
                "HTTP-Referer": "https://localhost/fact-checking",
                "X-Title": "Vietnamese Fact Checking Claim Generation",
            }
        )
    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            response_body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        response_body = error.read().decode("utf-8", errors="replace")
        try:
            error_data = json.loads(response_body)
            error_value = error_data.get("error", error_data)
            if isinstance(error_value, dict):
                message = str(error_value.get("message") or error_value)
                details = error_value.get("details")
                if details:
                    message += " Details: " + json.dumps(details)
            else:
                message = str(error_value)
        except json.JSONDecodeError:
            message = response_body or str(error)
        raise ProviderAPIError(
            provider, error.code, message, error.headers
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Cannot connect to {provider} at {api_url}: {error.reason}"
        ) from error

    try:
        response_data = json.loads(response_body)
        content = response_data["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
        raise RuntimeError(
            f"Unexpected API response: {response_body[:1000]}"
        ) from error

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content or "")


_gemini_quota_exhausted_for_run = False


def call_gemini_api(user_prompt: str) -> str:
    return call_provider_api(
        user_prompt,
        provider="Gemini",
        api_url=GEMINI_API_URL,
        api_key=load_gemini_api_key(),
        model=GEMINI_MODEL,
    )


def call_openrouter_api(user_prompt: str) -> str:
    return call_provider_api(
        user_prompt,
        provider="OpenRouter",
        api_url=OPENROUTER_API_URL,
        api_key=load_openrouter_api_key(),
        model=OPENROUTER_MODEL,
    )


def call_api_with_fallback(user_prompt: str) -> str:
    """Use Gemini first, then latch to OpenRouter after a Gemini quota error."""
    global _gemini_quota_exhausted_for_run

    gemini_key = load_gemini_api_key()
    openrouter_key = load_openrouter_api_key()
    if gemini_key and not _gemini_quota_exhausted_for_run:
        try:
            return call_gemini_api(user_prompt)
        except ProviderAPIError as error:
            if not is_quota_error(error) or not openrouter_key:
                raise
            retry_delay = get_retry_delay(error)
            if retry_delay is not None and retry_delay <= MAX_AUTO_RATE_LIMIT_WAIT:
                wait_seconds = retry_delay + 1
                print(
                    f"  Gemini short rate limit; waiting {wait_seconds:.1f}s "
                    "before using paid fallback."
                )
                time.sleep(wait_seconds)
                try:
                    return call_gemini_api(user_prompt)
                except ProviderAPIError as retry_error:
                    if not is_quota_error(retry_error):
                        raise
            _gemini_quota_exhausted_for_run = True
            print(
                "  Gemini free quota is unavailable; "
                "switching to OpenRouter for the rest of this run."
            )

    if openrouter_key:
        return call_openrouter_api(user_prompt)

    raise RuntimeError(
        "No usable API provider. Set GEMINI_API_KEY_1930 and/or OPENROUTER_API_KEY."
    )


def is_quota_error(error: Exception) -> bool:
    """Return True when the API reports a rate or quota limit."""
    message = f"{type(error).__name__}: {error}".lower()
    if isinstance(error, ProviderAPIError) and error.status_code == 429:
        return True
    return any(
        marker in message
        for marker in ("429", "rate limit", "quota", "insufficient balance")
    )


def is_model_unavailable_error(error: Exception) -> bool:
    """Return True for a missing, retired, or inaccessible model."""
    message = f"{type(error).__name__}: {error}".lower()
    return (
        isinstance(error, ProviderAPIError) and error.status_code == 404
    ) or ("model" in message and "not found" in message)


def get_retry_delay(error: Exception) -> float | None:
    """Extract a suggested retry delay from headers or the API error."""
    if isinstance(error, ProviderAPIError) and error.headers:
        retry_after = error.headers.get("Retry-After")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass

    message = str(error)
    patterns = (
        r"Please retry in ([0-9.]+)s",
        r"retry after ([0-9.]+)\s*(?:s|seconds?)",
        r"retry_after['\"]?\s*[:=]\s*['\"]?([0-9.]+)",
        r"['\"]retryDelay['\"]:\s*['\"]([0-9.]+)s",
    )
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


class ModelUnavailableError(RuntimeError):
    """Raised when the configured OpenRouter model cannot be used."""


class QuotaReachedError(RuntimeError):
    """Raised when rate/quota limits cannot recover with a short wait."""


class OutputValidationError(RuntimeError):
    """Raised when the model repeatedly returns an invalid claim structure."""

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
            if not isinstance(claim_item, dict):
                errors.append(f"claims.{label}[{i}] is not an object")
                continue
            claim_value = claim_item.get("claim")
            if not isinstance(claim_value, str) or not claim_value.strip():
                errors.append(f"claims.{label}[{i}].claim is empty or not a string")

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
            content = call_api_with_fallback(user_prompt)

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
            print(f"  [Attempt {attempt}/{MAX_RETRIES}] API error: {type(e).__name__}: {e}")
            log_debug(article_id, attempt, "", f"API error: {type(e).__name__}: {e}")

            if is_model_unavailable_error(e):
                raise ModelUnavailableError(
                    f"Configured model is unavailable: {e}"
                ) from e

            if is_quota_error(e):
                retry_delay = get_retry_delay(e)
                if (
                    retry_delay is not None
                    and retry_delay <= MAX_AUTO_RATE_LIMIT_WAIT
                    and attempt < MAX_RETRIES
                ):
                    wait_seconds = retry_delay + 1
                    print(f"  Rate limit reached. Waiting {wait_seconds:.1f}s as requested by the API...")
                    time.sleep(wait_seconds)
                    continue
                raise QuotaReachedError(
                    "API rate/quota or credit limit reached; progress will be saved."
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
        "--start-index",
        type=int,
        default=DEFAULT_START_INDEX,
        help=(
            "One-based corpus record number to start from "
            f"(default: {DEFAULT_START_INDEX})."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N records starting from --start-index.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=DEFAULT_REQUEST_DELAY,
        help=f"Seconds to wait between successful requests (default: {DEFAULT_REQUEST_DELAY}).",
    )
    args = parser.parse_args()

    if args.start_index <= 0:
        parser.error("--start-index must be greater than 0")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than 0")
    if args.request_delay < 0:
        parser.error("--request-delay cannot be negative")

    # Check API keys. Gemini is preferred; OpenRouter is the quota fallback.
    gemini_key = load_gemini_api_key()
    openrouter_key = load_openrouter_api_key()
    if not gemini_key and not openrouter_key:
        print(
            "ERROR: Set GEMINI_API_KEY_1930 and/or OPENROUTER_API_KEY "
            "in the environment or project .env file."
        )
        sys.exit(1)
    if gemini_key:
        print("Primary provider: Gemini API (free-tier quota first)")
    if openrouter_key:
        print("Fallback provider: OpenRouter")

    # Load dataset
    if not args.input_file.is_file():
        print(f"ERROR: Input dataset not found: {args.input_file}")
        sys.exit(1)

    print(f"Loading dataset from {args.input_file}...")
    raw_records = load_dataset(args.input_file)
    start_offset = args.start_index - 1
    if start_offset >= len(raw_records):
        parser.error(
            f"--start-index {args.start_index} exceeds corpus size "
            f"({len(raw_records)} records)"
        )
    end_offset = None if args.limit is None else start_offset + args.limit
    raw_records = raw_records[start_offset:end_offset]
    articles = [
        normalize_corpus_article(record, index)
        for index, record in enumerate(raw_records, start=args.start_index)
    ]
    print(
        f"Starting at corpus record {args.start_index}; "
        f"selected articles: {len(articles)}"
    )

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
    quota_exhausted = False

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
        except QuotaReachedError as e:
            quota_exhausted = True
            print(f"  {e}")
            print("  Run the same command later to resume from this article.")
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

        # Respect provider request-rate limits.
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
    print("PAUSED (API RATE/QUOTA LIMIT)" if quota_exhausted else "COMPLETE!")
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

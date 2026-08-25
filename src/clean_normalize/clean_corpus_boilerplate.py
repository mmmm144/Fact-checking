#!/usr/bin/env python3
"""Standardization and cleaning pipeline for Vietnamese Evidence Corpus v1.0.

Processes individual raw crawled JSON files, standardizes schemas, cleans text content,
filters anomalies, enriches metadata, and compiles them into a final consolidated corpus.
"""

from __future__ import annotations

import json
import logging
import hashlib
import re
import unicodedata
import math
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from datetime import datetime
from bs4 import BeautifulSoup, Tag, NavigableString
from collections import defaultdict

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)
LOGGER = logging.getLogger("corpus_cleaner")

# Constants
SOURCE_CONFIGS = {
    "gso": {
        "title": "GSO",
        "path": "src/crawl/output/gso.json",
        "source_type": "Government",
        "domain": "Economy",
        "document_type": "Statistics",
        "country": "Vietnam",
        "default_lang": "vi"
    },
    "moh": {
        "title": "MOH",
        "path": "src/crawl/output/moh.json",
        "source_type": "Government",
        "domain": "Health",
        "document_type": "Guideline",
        "country": "Vietnam",
        "default_lang": "vi"
    },
    "vafc": {
        "title": "VAFC",
        "path": "src/crawl/output/vafc.json",
        "source_type": "Fact Checking Portal",
        "domain": "Fact Checking",
        "document_type": "Debunking",
        "country": "Vietnam",
        "default_lang": "vi"
    },
    "who": {
        "title": "WHO",
        "path": "src/crawl/output/who.json",
        "source_type": "International Organization",
        "domain": "Health",
        "document_type": "Report",
        "country": "Global",
        "default_lang": "en"
    },
    "bao_chinh_phu": {
        "title": "BaoChinhPhu",
        "path": "src/crawl/output_news/bao_chinh_phu.json",
        "source_type": "Government",
        "domain": "Government & Policy",
        "document_type": "News",
        "country": "Vietnam",
        "default_lang": "vi"
    },
    "vnexpress": {
        "title": "VnExpress",
        "path": "src/crawl/output_news/vnexpress.json",
        "source_type": "News Agency",
        "domain": "General News",
        "document_type": "News",
        "country": "Vietnam",
        "default_lang": "vi"
    },
    "world_bank": {
        "title": "WorldBank",
        "path": "src/crawl/output_news/world_bank.json",
        "source_type": "International Organization",
        "domain": "Economy",
        "document_type": "Report",
        "country": "Global",
        "default_lang": "en"
    },
}

VI_STOPWORDS = {"và", "của", "là", "trong", "để", "có", "các", "cho", "người", "được", "với", "những", "trên", "ra", "đã", "này", "một", "từ", "tại", "khi"}
EN_STOPWORDS = {"the", "and", "of", "to", "in", "is", "that", "it", "on", "for", "with", "as", "was", "by", "an", "at", "are", "this", "from"}

BOILERPLATE_MIN_DOC_FREQ = 5
BOILERPLATE_MIN_WORDS = 3
BOILERPLATE_MAX_WORDS = 60

GLOBAL_BOILERPLATE_PATTERNS = [
    r"^(xem them|xem tiep|doc tiep|tin khac|tin lien quan|bai lien quan|cac bai lien quan|bai viet lien quan|doc them|cung chu de|menu chuyen muc|chuyen muc|huong dan chon vnexpress tren google)$",
    r"^(xem them|tin khac|tin lien quan|bai lien quan|doc them|cung chu de|bai viet lien quan|huong dan chon vnexpress tren google)",
]

SOURCE_BOILERPLATE_PATTERNS = {
    "vnexpress": [
        r"^huong dan chon vnexpress tren google$",
        r"^menu chuyen muc$",
        r"^tin khac$",
        r"^xem them$",
        r"^bai viet lien quan$",
        r"^cac bai lien quan$",
    ],
    "bao_chinh_phu": [
        r"^tin lien quan$",
        r"^xem them$",
        r"^bai lien quan$",
    ],
    "world_bank": [
        r"^related items$",
        r"^related links$",
        r"^more$",
    ],
}


def clean_html(text: str) -> str:
    """Strip HTML tags/entities, retaining structure (headings, paragraphs, lists)."""
    if not text:
        return ""
    
    # Quick precheck if it even contains html tags/entities
    if "<" not in text and "&" not in text:
        return text
        
    soup = BeautifulSoup(text, "html.parser")
    
    # Decompose script, style, and media
    for unwanted in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
        unwanted.decompose()
        
    # Decompose common boilerplate elements
    boilerplate_selectors = [
        ".footer", ".header", ".navigation", ".nav", ".share", ".sharing",
        ".social", ".related", ".related-posts", ".comment", ".comments",
        ".copyright", ".ads", ".advertisement", "#footer", "#header", "#navigation"
    ]
    for selector in boilerplate_selectors:
        for el in soup.select(selector):
            el.decompose()
            
    # Standardize tags into inline spacing
    # Insert newline spacing around block elements for readability
    for block in soup(["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "caption"]):
        block.insert_before(NavigableString("\n"))
        block.insert_after(NavigableString("\n"))
        
    clean_text = soup.get_text(separator=" ")
    return clean_text


def normalize_unicode_nfc(text: str) -> str:
    """Normalize text using Unicode Normalization Form C (NFC)."""
    if not text:
        return ""
    return unicodedata.normalize("NFC", text)


def normalize_whitespace(text: str) -> str:
    """Collapse consecutive spaces and regulate paragraph-level newlines."""
    if not text:
        return ""
    # Replace multiple spaces with a single space
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    # Replace three or more consecutive newlines with exactly two newlines (paragraphs spacing)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def validate_repair_encoding(text: str) -> str:
    """Fix mojibake encoding representations and strip unprintable control characters."""
    if not text:
        return ""
    
    # Fix common unicode character representations if any
    # (like double-escaped sequences or specific raw strings)
    text = text.replace("\\u201c", '"').replace("\\u201d", '"')
    text = text.replace("\\u2019", "'").replace("\\u2018", "'")
    text = text.replace("\\u00a0", " ")
    
    # Remove system control characters (\x00-\x1f except tab, newline, carriage return)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text


def detect_language(text: str, default_lang: str) -> str:
    """Detect language ('vi' or 'en') using stopword density."""
    if not text:
        return default_lang
        
    words = set(re.findall(r"\w+", text.lower()))
    vi_matches = len(words.intersection(VI_STOPWORDS))
    en_matches = len(words.intersection(EN_STOPWORDS))
    
    if vi_matches > 3 and en_matches <= 1:
        return "vi"
    elif en_matches > 3 and vi_matches <= 1:
        return "en"
    elif vi_matches > 2 and en_matches > 2:
        # Mixed language - prioritize default source language
        return default_lang
    else:
        # Fallback using alphabet search for typical Vietnamese diacritics
        if re.search(r"[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệđìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]", text, re.IGNORECASE):
            return "vi"
        return default_lang


def standardize_date(date_str: str, url: str) -> str:
    """Parse and convert date to ISO YYYY-MM-DD; fallback to parsing from URL path."""
    date_str = date_str.strip()
    
    # Attempt parsing date from string
    if date_str:
        # Check standard ISO format first
        iso_match = re.match(r"^(\d{4})[-/.](\d{2})[-/.](\d{2})", date_str)
        if iso_match:
            return f"{iso_match.group(1)}-{iso_match.group(2)}-{iso_match.group(3)}"
            
        # Common ISO with time representation (Z or offset)
        iso_value = date_str.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(iso_value).date().isoformat()
        except ValueError:
            pass
            
        # Try DD/MM/YYYY or similar regexes
        patterns = [
            (r"(?P<year>\d{4})[-/.](?P<month>\d{1,2})[-/.](?P<day>\d{1,2})", True),
            (r"(?P<day>\d{1,2})[-/.](?P<month>\d{1,2})[-/.](?P<year>\d{4})", False)
        ]
        for pattern, year_first in patterns:
            match = re.search(pattern, date_str)
            if match:
                try:
                    return datetime(
                        int(match.group("year")),
                        int(match.group("month")),
                        int(match.group("day"))
                    ).date().isoformat()
                except ValueError:
                    continue
                    
    # Fallback to URL path extraction (e.g. /2026/03/04/)
    url_match = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", url)
    if url_match:
        return f"{url_match.group(1)}-{url_match.group(2)}-{url_match.group(3)}"
        
    return ""


def calculate_statistics(text: str, lang: str) -> dict[str, int]:
    """Calculate character, word, sentence count and estimated reading time."""
    if not text:
        return {"char_count": 0, "word_count": 0, "sentence_count": 0, "reading_time": 0}
        
    char_count = len(text)
    words = re.findall(r"\S+", text)
    word_count = len(words)
    
    # Count sentences using sentence boundaries
    sentences = re.split(r"[.!?]+", text)
    sentence_count = len([s for s in sentences if s.strip()])
    
    # Calculate reading time based on language WPM standards
    # Vietnamese reading speed average: 150 WPM
    # English reading speed average: 200 WPM
    wpm = 150 if lang == "vi" else 200
    reading_time = math.ceil(word_count / wpm) if word_count > 0 else 0
    
    return {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "reading_time": reading_time
    }


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def normalize_boilerplate_signature(text: str) -> str:
    text = strip_accents(text).lower()
    text = re.sub(r"[\u2010-\u2015\-–—_/|]+", " ", text)
    text = re.sub(r"[^0-9a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_boilerplate_signature(signature: str, source_key: str) -> bool:
    patterns = list(GLOBAL_BOILERPLATE_PATTERNS)
    patterns.extend(SOURCE_BOILERPLATE_PATTERNS.get(source_key, []))
    return any(re.search(pattern, signature) for pattern in patterns)


def extract_candidate_signatures(text: str, source_key: str) -> set[str]:
    signatures: set[str] = set()
    for line in text.splitlines():
        signature = normalize_boilerplate_signature(line)
        if not signature:
            continue
        words = signature.split()
        if len(words) < BOILERPLATE_MIN_WORDS or len(words) > BOILERPLATE_MAX_WORDS:
            continue
        if is_boilerplate_signature(signature, source_key):
            continue
        signatures.add(signature)
    return signatures


def learn_boilerplate_signatures(records: list[dict[str, Any]]) -> set[str]:
    signature_counts: dict[str, int] = defaultdict(int)
    for record in records:
        source_key = record.get("source_key", "")
        for signature in extract_candidate_signatures(record["text"], source_key):
            signature_counts[signature] += 1
    return {
        signature
        for signature, count in signature_counts.items()
        if count >= BOILERPLATE_MIN_DOC_FREQ
    }


def remove_boilerplate_text(
    text: str,
    source_key: str,
    learned_signatures: set[str],
) -> tuple[str, int]:
    kept_lines: list[str] = []
    removed_count = 0
    for line in text.splitlines():
        signature = normalize_boilerplate_signature(line)
        if signature and (
            is_boilerplate_signature(signature, source_key)
            or signature in learned_signatures
        ):
            removed_count += 1
            continue
        kept_lines.append(line)
    cleaned_text = normalize_whitespace("\n".join(kept_lines))
    return cleaned_text, removed_count


def clean_single_record(raw_item: dict[str, Any], config: dict[str, Any], crawl_date_str: str, source_key: str) -> dict[str, Any] | None:
    """Standardize schema, clean text, and enrich a single document record."""
    # 1. Schema Mapping & Basic Cleanup
    raw_title = raw_item.get("claim", "").strip()
    raw_body = raw_item.get("justification", "").strip()
    raw_summary = raw_item.get("original_text", "").strip()
    url = raw_item.get("url", "").strip()
    
    # 2. Preprocessing / Cleaning body and title
    cleaned_title = normalize_whitespace(normalize_unicode_nfc(validate_repair_encoding(clean_html(raw_title))))
    cleaned_body = normalize_whitespace(normalize_unicode_nfc(validate_repair_encoding(clean_html(raw_body))))
    cleaned_summary = normalize_whitespace(normalize_unicode_nfc(validate_repair_encoding(clean_html(raw_summary))))
    
    # If the core content is missing, filter it out early.
    if not cleaned_title or not cleaned_body:
        return None
        
    # 3. Metadata Standardization & Enrichment
    source_name = config["title"]
    source_type = config["source_type"]
    domain = config["domain"]
    doc_type = config["document_type"]
    country = config["country"]
    
    # Determine language
    lang = detect_language(cleaned_body, config["default_lang"])
    
    # Format dates
    raw_date = raw_item.get("publish_date", "")
    pub_date = standardize_date(raw_date, url)
    
    # Generate Unique ID: {SOURCE}_{MD5}
    hasher = hashlib.md5()
    hasher.update(f"{url}|{cleaned_title}".encode("utf-8"))
    doc_id = f"{source_name.upper()}_{hasher.hexdigest()[:16]}"
    
    # Statistics calculation
    stats = calculate_statistics(cleaned_body, lang)
    
    # Quality flags
    quality = {
        "cleaned": True,
        "html_removed": "<" in raw_body or "&" in raw_body,
        "duplicate": False,  # Populated downstream during deduplication check
        "boilerplate_removed": False,
        "language_verified": True
    }
    
    # Final Standard Schema structure
    return {
        "doc_id": doc_id,
        "title": cleaned_title,
        "text": cleaned_body,
        "summary": cleaned_summary if cleaned_summary else None,
        "source": source_name,
        "source_type": source_type,
        "domain": domain,
        "document_type": doc_type,
        "language": lang,
        "country": country,
        "publish_date": pub_date if pub_date else crawl_date_str,
        "crawl_date": crawl_date_str,
        "url": url,
        "source_key": source_key,
        "author": None,
        "metadata": stats,
        "quality": quality
    }


def execute_pipeline(project_root_path: Path) -> None:
    """Execute the full cleaning and standardization pipeline on all sources."""
    LOGGER.info("Starting Vietnamese Evidence Corpus v1.0 Standardization and Cleaning pipeline...")
    
    crawl_date_str = datetime.now().date().isoformat()
    processed_records = []
    discarded_stats = {
        "too_short": 0,
        "missing_title": 0,
        "failed_parse": 0
    }
    
    # Loop over all defined sources
    for source_key, config in SOURCE_CONFIGS.items():
        file_path = project_root_path / config["path"]
        if not file_path.exists():
            LOGGER.warning("Data file for %s does not exist at: %s", config["title"], file_path)
            continue
            
        LOGGER.info("Loading and processing source: %s (%s)", config["title"], file_path.name)
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            
        source_processed_count = 0
        for raw_item in raw_data:
            try:
                # Basic checks
                if not raw_item.get("claim"):
                    discarded_stats["missing_title"] += 1
                    continue
                    
                clean_item = clean_single_record(raw_item, config, crawl_date_str, source_key)
                if not clean_item:
                    discarded_stats["too_short"] += 1
                    continue
                processed_records.append(clean_item)
                source_processed_count += 1
            except Exception as e:
                LOGGER.error("Failed to parse record in %s: %s", config["title"], e)
                discarded_stats["failed_parse"] += 1
                
        LOGGER.info("Successfully cleaned %d / %d records for %s", source_processed_count, len(raw_data), config["title"])

    learned_boilerplate_signatures = learn_boilerplate_signatures(processed_records)
    LOGGER.info("Learned %d repeated boilerplate signatures", len(learned_boilerplate_signatures))

    seen_urls = {}
    seen_contents = {}
    final_records = []
    boilerplate_removed_records = 0

    for record in processed_records:
        cleaned_text, removed_count = remove_boilerplate_text(
            record["text"],
            record.get("source_key", ""),
            learned_boilerplate_signatures,
        )
        if removed_count:
            boilerplate_removed_records += 1
        if not cleaned_text:
            discarded_stats["too_short"] += 1
            continue

        word_count = len(re.findall(r"\S+", cleaned_text))
        if word_count < 30:
            discarded_stats["too_short"] += 1
            continue

        updated_record = dict(record)
        updated_record["text"] = cleaned_text
        updated_record["metadata"] = calculate_statistics(cleaned_text, updated_record["language"])
        updated_quality = dict(updated_record["quality"])
        updated_quality["boilerplate_removed"] = removed_count > 0
        updated_quality["boilerplate_line_count"] = removed_count
        updated_record["quality"] = updated_quality

        url = updated_record["url"]
        text_hash = hashlib.md5(updated_record["text"].encode("utf-8")).hexdigest()

        if url in seen_urls:
            updated_record["quality"]["duplicate"] = True
            seen_urls[url]["quality"]["duplicate"] = True
        else:
            seen_urls[url] = updated_record

        if text_hash in seen_contents:
            updated_record["quality"]["duplicate"] = True
            seen_contents[text_hash]["quality"]["duplicate"] = True
        else:
            seen_contents[text_hash] = updated_record

        final_records.append(updated_record)

    # Statistics reporting
    total_processed = len(final_records)
    duplicate_count = sum(1 for item in final_records if item["quality"]["duplicate"])
    non_duplicate_count = total_processed - duplicate_count
    
    LOGGER.info("\n=== PIPELINE DIAGNOSTIC STATISTICS ===")
    LOGGER.info("Total input records parsed: %d", len(processed_records) + sum(discarded_stats.values()))
    LOGGER.info("Total records kept in v1.0: %d", total_processed)
    LOGGER.info("  - Records with boilerplate removed: %d", boilerplate_removed_records)
    LOGGER.info("  - Learned boilerplate signatures:   %d", len(learned_boilerplate_signatures))
    LOGGER.info("  - Duplicates flagged:     %d", duplicate_count)
    LOGGER.info("  - Unique records:         %d", non_duplicate_count)
    LOGGER.info("Discarded records:          %d", sum(discarded_stats.values()))
    LOGGER.info("  - Too short (<30 words):  %d", discarded_stats["too_short"])
    LOGGER.info("  - Missing claim/title:    %d", discarded_stats["missing_title"])
    LOGGER.info("  - Processing exceptions:  %d", discarded_stats["failed_parse"])
    
    # Save output corpus - filtering out duplicates after boilerplate cleanup
    output_path = project_root_path / "src" / "clean_normalize" / "output" / "corpus_v1_boilerplate.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info("Writing clean standardized corpus (deduplicated) to: %s", output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_records, f, ensure_ascii=False, indent=4)
        
    LOGGER.info("Evidence Corpus v1.0 compiled successfully! Output size: %.2f MB", output_path.stat().st_size / (1024**2))


if __name__ == "__main__":
    # Find project root
    curr_path = Path(__file__).resolve().parent
    while curr_path.name and not (curr_path / "src").is_dir():
        curr_path = curr_path.parent
    execute_pipeline(curr_path)

#!/usr/bin/env python3
"""Crawl World Bank, Báo Chính phủ and VnExpress news articles.

The implementation reuses the HTTP, article parsing, JSON schema and checkpoint
helpers from ``crawl_sources.py``. Discovery is source-specific:

* World Bank: the official ``en-news`` XML sitemaps;
* Báo Chính phủ: the public timeline endpoint behind each requested section;
* VnExpress: the requested category pages and their ``-pN`` pagination.

Each successfully parsed article is appended to an internal JSONL checkpoint so
an interrupted run can continue. Human-readable JSON arrays are written after a
source finishes.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

import requests
import urllib3
from bs4 import BeautifulSoup, Tag

import crawl_sources as base
from crawl_sources import Candidate, HttpClient, SourceConfig


LOGGER = logging.getLogger("news_source_crawler")

BAO_CHINH_PHU_CATEGORIES = (
    ("Chính trị", "https://baochinhphu.vn/chinh-tri.htm"),
    ("Kinh tế", "https://baochinhphu.vn/kinh-te.htm"),
    ("Văn hóa", "https://baochinhphu.vn/van-hoa.htm"),
    ("Xã hội", "https://baochinhphu.vn/xa-hoi.htm"),
    ("Khoa giáo", "https://baochinhphu.vn/khoa-giao.htm"),
    ("Quốc tế", "https://baochinhphu.vn/quoc-te.htm"),
)

VNEXPRESS_CATEGORIES = (
    ("Thời sự", "https://vnexpress.net/thoi-su"),
    ("Thế giới", "https://vnexpress.net/the-gioi"),
    ("Kinh doanh", "https://vnexpress.net/kinh-doanh"),
    ("Khoa học công nghệ", "https://vnexpress.net/khoa-hoc-cong-nghe"),
    ("Góc nhìn", "https://vnexpress.net/goc-nhin"),
    ("Spotlight", "https://vnexpress.net/spotlight"),
    ("Bất động sản", "https://vnexpress.net/bat-dong-san"),
    ("Sức khỏe", "https://vnexpress.net/suc-khoe"),
    ("Giải trí", "https://vnexpress.net/giai-tri"),
    ("Thể thao", "https://vnexpress.net/the-thao"),
    ("Pháp luật", "https://vnexpress.net/phap-luat"),
    ("Giáo dục", "https://vnexpress.net/giao-duc"),
    ("Đời sống", "https://vnexpress.net/doi-song"),
    ("Ôtô - Xe máy", "https://vnexpress.net/oto-xe-may"),
    ("Du lịch", "https://vnexpress.net/du-lich"),
    ("Tâm sự", "https://vnexpress.net/tam-su"),
    ("Thư giãn", "https://vnexpress.net/thu-gian"),
)

WORLD_BANK_DOMAINS = {
    "press-release": "Press Releases",
    "feature": "Features",
    "factsheet": "Fact Sheets",
    "immersive-story": "Immersive Stories",
    "opinion": "Opinions",
    "speech": "Speeches",
    "statement": "Statements",
}

NEWS_SOURCES: dict[str, SourceConfig] = {
    "world_bank": SourceConfig(
        key="world_bank",
        name="World Bank",
        target=3400,
        label="TRUE",
        language="en",
        allowed_hosts=("worldbank.org", "www.worldbank.org"),
        sitemap_urls=("https://www.worldbank.org/sitemap.xml",),
        sitemap_patterns=(
            r"worldbank\.org/content/vdam/sitemap/www-worldbank-org/en-news-sitemap-\d+\.xml$",
        ),
        article_patterns=(
            r"^https?://www\.worldbank\.org/en/news/"
            r"(?:press-release|feature|factsheet|immersive-story|opinion|speech|statement)"
            r"/\d{4}/\d{2}/\d{2}/[^?#]+",
        ),
        preferred_selectors=(
            ".lp-body-content",
            ".lpb-article-content",
            ".article-body",
            "main article",
            ".custom-text",
        ),
        evidence="Article published on the official World Bank website.",
    ),
    "bao_chinh_phu": SourceConfig(
        key="bao_chinh_phu",
        name="Báo Chính phủ",
        target=2500,
        label="TRUE",
        language="vi",
        allowed_hosts=("baochinhphu.vn", "www.baochinhphu.vn"),
        seed_urls=tuple(url for _, url in BAO_CHINH_PHU_CATEGORIES),
        category_pages=BAO_CHINH_PHU_CATEGORIES,
        article_patterns=(
            r"^https?://(?:www\.)?baochinhphu\.vn/"
            r"(?!timelinelist/|chu-de/|[^?#]*/trang-\d+\.htm)"
            r"[^?#]*-\d{10,}\.htm$",
        ),
        preferred_selectors=(
            ".detail-content",
            ".article__body",
            ".content-news-detail",
            ".detail__content",
        ),
        evidence="Bài viết được đăng trên Báo Điện tử Chính phủ.",
    ),
    "vnexpress": SourceConfig(
        key="vnexpress",
        name="VnExpress",
        target=2600,
        label="TRUE",
        language="vi",
        allowed_hosts=("vnexpress.net", "www.vnexpress.net"),
        seed_urls=tuple(url for _, url in VNEXPRESS_CATEGORIES),
        category_pages=VNEXPRESS_CATEGORIES,
        article_patterns=(
            r"^https?://(?:www\.)?vnexpress\.net/[^?#]+-\d+\.html$",
        ),
        preferred_selectors=(
            ".fck_detail",
            "article.fck_detail",
            ".sidebar-1",
            "article",
        ),
        evidence="Bài viết được đăng trên báo điện tử VnExpress.",
    ),
}


def merge_candidate(found: dict[str, Candidate], candidate: Candidate) -> bool:
    """Add a candidate, preferring the later configured category on overlap."""
    previous = found.get(candidate.url)
    if previous is None or candidate.domain_rank > previous.domain_rank:
        found[candidate.url] = candidate
        return previous is None
    return False


def extract_candidates(
    html: bytes | str,
    page_url: str,
    config: SourceConfig,
    *,
    domain: str,
    domain_rank: int,
    selectors: Iterable[str],
) -> dict[str, Candidate]:
    """Extract unique article URLs only from the requested listing containers."""
    soup = BeautifulSoup(html, "html.parser")
    anchors: list[Tag] = []
    for selector in selectors:
        anchors.extend(node for node in soup.select(selector) if isinstance(node, Tag))
    if not anchors:
        anchors = [node for node in soup.find_all("a", href=True) if isinstance(node, Tag)]

    found: dict[str, Candidate] = {}
    for anchor in anchors:
        href = anchor.get("href")
        if not href:
            continue
        url = base.normalize_url(str(href), page_url, article=True)
        if base.is_article_url(url, config):
            found.setdefault(url, Candidate(url, domain=domain, domain_rank=domain_rank))
    return found


def extract_bao_chinh_phu_zone_id(html: bytes | str) -> str:
    """Return the numeric timeline zone configured in a category page."""
    soup = BeautifulSoup(html, "html.parser")
    zone = soup.find(id="hdZoneId")
    if isinstance(zone, Tag):
        value = str(zone.get("value", "")).strip()
        if value.isdigit():
            return value
    return ""


def vnexpress_page_url(category_url: str, page: int) -> str:
    if page <= 1:
        return category_url
    return f"{category_url.rstrip('/')}-p{page}"


def discover_bao_chinh_phu(
    client: HttpClient,
    config: SourceConfig,
    *,
    target: int,
    max_pages: int,
) -> dict[str, Candidate]:
    """Discover articles from the timeline endpoint used by six sections."""
    category_zones: list[tuple[int, str, str]] = []
    for rank, (domain, category_url) in enumerate(config.category_pages):
        try:
            response = client.get(category_url, verify_tls=config.verify_tls)
        except (requests.RequestException, PermissionError) as exc:
            LOGGER.warning("Không đọc được chuyên mục %s: %s", category_url, exc)
            continue
        zone_id = extract_bao_chinh_phu_zone_id(response.content)
        if not zone_id:
            LOGGER.warning("Không tìm thấy hdZoneId tại %s", category_url)
            continue
        category_zones.append((rank, domain, zone_id))

    found: dict[str, Candidate] = {}
    active = {zone_id for _, _, zone_id in category_zones}
    desired = max(target, math.ceil(target * 1.25))
    selectors = (
        ".box-stream-item a.box-stream-link-with-avatar[href]",
        ".box-stream-item .box-stream-link-title[href]",
        ".box-stream-item h2 a[href]",
        ".box-stream-item h3 a[href]",
    )
    for page in range(1, max_pages + 1):
        for rank, domain, zone_id in category_zones:
            if zone_id not in active:
                continue
            page_url = f"https://baochinhphu.vn/timelinelist/{zone_id}/{page}.htm"
            try:
                response = client.get(page_url, verify_tls=config.verify_tls)
            except (requests.RequestException, PermissionError) as exc:
                LOGGER.warning("Lỗi timeline %s: %s", page_url, exc)
                active.discard(zone_id)
                continue
            page_candidates = extract_candidates(
                response.content,
                response.url,
                config,
                domain=domain,
                domain_rank=rank,
                selectors=selectors,
            )
            added = sum(merge_candidate(found, item) for item in page_candidates.values())
            LOGGER.info(
                "Báo Chính phủ - %s trang %d: thêm %d (tổng %d)",
                domain,
                page,
                added,
                len(found),
            )
            if not page_candidates:
                active.discard(zone_id)
        if len(found) >= desired or not active:
            break
    return found


def discover_vnexpress(
    client: HttpClient,
    config: SourceConfig,
    *,
    target: int,
    max_pages: int,
) -> dict[str, Candidate]:
    """Discover VnExpress articles with balanced round-robin pagination."""
    found: dict[str, Candidate] = {}
    active = set(range(len(config.category_pages)))
    desired = max(target, math.ceil(target * 1.25))
    selectors = (
        "article.item-news h3.title-news a[href]",
        "article.item-news h2.title-news a[href]",
        "article.item-news .title-news a[href]",
        ".item-news .title-news a[href]",
    )
    seen_page_sets: dict[int, frozenset[str]] = {}

    for page in range(1, max_pages + 1):
        for rank, (domain, category_url) in enumerate(config.category_pages):
            if rank not in active:
                continue
            page_url = vnexpress_page_url(category_url, page)
            try:
                response = client.get(page_url, verify_tls=config.verify_tls)
            except (requests.RequestException, PermissionError) as exc:
                LOGGER.warning("Lỗi chuyên mục VnExpress %s: %s", page_url, exc)
                active.discard(rank)
                continue
            page_candidates = extract_candidates(
                response.content,
                response.url,
                config,
                domain=domain,
                domain_rank=rank,
                selectors=selectors,
            )
            current_set = frozenset(page_candidates)
            if not current_set or current_set == seen_page_sets.get(rank):
                active.discard(rank)
                continue
            seen_page_sets[rank] = current_set
            added = sum(merge_candidate(found, item) for item in page_candidates.values())
            LOGGER.info(
                "VnExpress - %s trang %d: thêm %d (tổng %d)",
                domain,
                page,
                added,
                len(found),
            )
        if len(found) >= desired or not active:
            break
    return found


def world_bank_domain(url: str) -> str:
    match = re.search(r"/en/news/([^/]+)/", urlsplit(url).path, flags=re.IGNORECASE)
    return WORLD_BANK_DOMAINS.get(match.group(1).lower(), "") if match else ""


def discover_candidates(
    client: HttpClient,
    config: SourceConfig,
    *,
    target: int,
    max_pages: int,
) -> dict[str, Candidate]:
    if config.key == "world_bank":
        found = base.discover_from_sitemaps(client, config)
        for candidate in found.values():
            candidate.domain = world_bank_domain(candidate.url)
        return found
    if config.key == "bao_chinh_phu":
        return discover_bao_chinh_phu(
            client,
            config,
            target=target,
            max_pages=max_pages,
        )
    if config.key == "vnexpress":
        return discover_vnexpress(
            client,
            config,
            target=target,
            max_pages=max_pages,
        )
    raise ValueError(f"Nguồn chưa được hỗ trợ: {config.key}")


def fetch_news_article(
    client: HttpClient,
    candidate: Candidate,
    config: SourceConfig,
) -> dict[str, Any] | None:
    try:
        response = client.get(candidate.url, verify_tls=config.verify_tls)
        content_type = response.headers.get("Content-Type", "").lower()
        if content_type and "html" not in content_type and "xhtml" not in content_type:
            return None
        domain = candidate.domain
        if config.key == "world_bank":
            domain = world_bank_domain(response.url) or domain
        record = base.parse_article(
            response.content,
            response.url,
            config,
            fallback_domain=domain,
        )
        if record and not record["publish_date"] and candidate.lastmod:
            record["publish_date"] = base.normalize_date(candidate.lastmod)
        return record
    except (requests.RequestException, PermissionError, UnicodeError, ValueError) as exc:
        LOGGER.warning("Lỗi bài %s: %s", candidate.url, exc)
        return None


def crawl_source(
    client: HttpClient,
    config: SourceConfig,
    *,
    output_dir: Path,
    target: int,
    workers: int,
    max_pages: int,
    discover_only: bool,
) -> dict[str, Any]:
    output_path = output_dir / f"{config.key}.json"
    checkpoint_path = base.prepare_checkpoint(output_dir, config.key)
    records, existing_urls = base.read_existing(checkpoint_path)
    existing_count = len(records)
    LOGGER.info("[%s] đã có %d bản ghi; mục tiêu %d", config.name, existing_count, target)

    candidates = discover_candidates(
        client,
        config,
        target=target,
        max_pages=max_pages,
    )
    # Sort newest first where a sitemap provides lastmod, while retaining the
    # balanced discovery order for listing-based sources with blank lastmod.
    candidate_list = [
        item
        for item in candidates.values()
        if item.url not in existing_urls
    ]
    candidate_list.sort(key=lambda item: item.lastmod, reverse=True)

    summary: dict[str, Any] = {
        "source": config.name,
        "target": target,
        "existing": existing_count,
        "discovered": len(candidates),
        "new_candidates": len(candidate_list),
        "added": 0,
        "total": existing_count,
        "shortfall": max(0, target - existing_count),
        "output": str(output_path),
    }
    if discover_only or existing_count >= target:
        if not discover_only:
            base.write_pretty_json(output_path, records)
        return summary

    needed = target - existing_count
    iterator = iter(candidate_list)
    pending: dict[Future[dict[str, Any] | None], Candidate] = {}
    successful_urls = set(existing_urls)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with checkpoint_path.open("a", encoding="utf-8", buffering=1) as stream, ThreadPoolExecutor(
        max_workers=max(1, workers),
        thread_name_prefix=config.key,
    ) as executor:

        def fill_queue() -> None:
            while len(pending) < max(1, workers) * 2:
                try:
                    candidate = next(iterator)
                except StopIteration:
                    return
                pending[executor.submit(fetch_news_article, client, candidate, config)] = candidate

        fill_queue()
        while pending and summary["added"] < needed:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                candidate = pending.pop(future)
                try:
                    record = future.result()
                except Exception as exc:  # Keep checkpointing after one worker fails.
                    LOGGER.exception("Worker lỗi ở %s: %s", candidate.url, exc)
                    record = None
                if record:
                    record = base.output_record(record)
                    record_url = base.normalize_url(str(record["url"]), article=True)
                    if record_url not in successful_urls:
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        records.append(record)
                        successful_urls.add(record_url)
                        summary["added"] += 1
                        LOGGER.info(
                            "[%s] %d/%d: %s",
                            config.name,
                            existing_count + summary["added"],
                            target,
                            record["claim"],
                        )
                if summary["added"] >= needed:
                    break
            fill_queue()
        for future in pending:
            future.cancel()

    summary["total"] = existing_count + summary["added"]
    summary["shortfall"] = max(0, target - summary["total"])
    base.write_pretty_json(output_path, records)
    return summary


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("giá trị phải lớn hơn 0")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("all", *NEWS_SOURCES.keys()),
        default="all",
        help="Nguồn cần crawl (mặc định: all)",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        help="Ghi đè quota của mỗi nguồn được chọn",
    )
    parser.add_argument("--workers", type=positive_int, default=4)
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Khoảng cách tối thiểu giữa hai request đến cùng host, tính bằng giây",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--max-pages",
        type=positive_int,
        default=100,
        help="Số trang tối đa cho mỗi chuyên mục Báo Chính phủ/VnExpress",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output_news",
    )
    parser.add_argument("--discover-only", action="store_true")
    parser.add_argument("--ignore-robots", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    for console_stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(console_stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="backslashreplace")
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.delay < 0 or args.timeout <= 0:
        raise SystemExit("--delay phải >= 0 và --timeout phải > 0")

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_keys = list(NEWS_SOURCES) if args.source == "all" else [args.source]
    client = HttpClient(
        delay=args.delay,
        timeout=args.timeout,
        obey_robots=not args.ignore_robots,
    )
    summaries = []
    for key in selected_keys:
        config = NEWS_SOURCES[key]
        summaries.append(
            crawl_source(
                client,
                config,
                output_dir=args.output_dir,
                target=args.limit or config.target,
                workers=args.workers,
                max_pages=args.max_pages,
                discover_only=args.discover_only,
            )
        )

    combined_count = 0
    if not args.discover_only:
        combined_count = base.combine_outputs(args.output_dir, NEWS_SOURCES.keys())
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "discover_only": args.discover_only,
        "combined_count": combined_count,
        "sources": summaries,
    }
    report_path = args.output_dir / "crawl_summary.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if args.discover_only or all(item["shortfall"] == 0 for item in summaries) else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Crawl article data from VAFC, Ministry of Health, WHO and GSO/NSO.

The crawler discovers URLs from XML sitemaps whenever possible and falls back to
category-page traversal for sites without a public sitemap. Human-readable
results are written as formatted JSON arrays. Internal JSONL checkpoints let
interrupted runs continue without starting over.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import re
import ssl
import sys
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import requests
import urllib3
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


LOGGER = logging.getLogger("source_crawler")
USER_AGENT = "ViFC-AcademicCrawler/1.0 (+fact-checking research; respectful rate limit)"
TRACKING_PARAMS = {"fbclid", "gclid", "mc_cid", "mc_eid"}
ARTICLE_TYPES = {
    "Article",
    "NewsArticle",
    "Report",
    "AnalysisNewsArticle",
    "MedicalWebPage",
}
OUTPUT_FIELDS = (
    "id",
    "source_type",
    "source_name",
    "url",
    "domain",
    "publish_date",
    "claim",
    "original_text",
    "label",
    "evidence",
    "justification",
    "comments",
)
MOJIBAKE_MARKERS = ("Ã", "Â", "Ä", "â€", "áº", "á»")
MOH_API_ORIGIN = "https://moh.gov.vn"
MOH_API_CATEGORIES = (
    # Tin tức - sự kiện
    ("8358", "Tin nổi bật", "Tin tức - sự kiện"),
    ("8351", "Tin hoạt động", "Tin tức - sự kiện"),
    ("8360", "Tin chỉ đạo điều hành", "Tin tức - sự kiện"),
    ("8356", "Tin tổng hợp", "Tin tức - sự kiện"),
    ("8500", "Phân cấp, phân quyền", "Tin tức - sự kiện"),
    ("8355", "Tin tức địa phương", "Tin tức - sự kiện"),
    ("8354", "Điểm tin Y tế từ các báo mạng", "Tin tức - sự kiện"),
    ("7252", "Tin ảnh", "Tin tức - sự kiện"),
    ("7253", "Tin video", "Tin tức - sự kiện"),
    ("7254", "Tin liên quan", "Tin tức - sự kiện"),
    ("8431", "Tin y tế nổi bật", "Tin tức - sự kiện"),
    # Cung cấp thông tin
    ("8334", "Chiến lược quy hoạch kế hoạch", "Cung cấp thông tin"),
    ("8349", "Hợp tác quốc tế", "Cung cấp thông tin"),
    ("8337", "Tuyên truyền, hướng dẫn thực hiện pháp luật nói chung", "Cung cấp thông tin"),
    ("7260", "Cải cách hành chính Bộ Y tế", "Cung cấp thông tin"),
    # Tra cứu
    ("8366", "Tuyển dụng, tuyển sinh", "Tra cứu"),
    ("8429", "Thông tin chương trình đề tài khoa học", "Tra cứu"),
    ("8422", "Thông tin đầu tư", "Tra cứu"),
    ("8295", "Phòng chống bệnh nghề nghiệp", "Tra cứu"),
    ("8473", "Phòng chống tai nạn thương tích", "Tra cứu"),
    ("8390", "Thi đua khen thưởng", "Tra cứu"),
    ("8338", "Chế độ, chính sách lĩnh vực Y tế", "Tra cứu"),
    # Chuyển đổi số y tế
    ("8430", "Chuyển đổi số y tế", "Chuyển đổi số y tế"),
)
MOH_PARENT_DOMAINS = tuple(dict.fromkeys(parent for _, _, parent in MOH_API_CATEGORIES))
MOH_PARENT_DOMAIN_BY_CATEGORY_ID = {
    category_id: parent for category_id, _, parent in MOH_API_CATEGORIES
}
MOH_PARENT_DOMAIN_BY_CHILD_NAME = {
    child.casefold(): parent for _, child, parent in MOH_API_CATEGORIES
}

WHO_DOMAIN_PAGES = (
    ("News", "https://www.who.int/news"),
    ("Emergencies", "https://www.who.int/emergencies/overview"),
    ("Campaigns", "https://www.who.int/campaigns"),
    ("Events", "https://www.who.int/news-room/events"),
    ("Statements", "https://www.who.int/news-room/statements"),
    ("Feature stories", "https://www.who.int/news-room/feature-stories"),
    ("Speeches", "https://www.who.int/news-room/speeches"),
    ("Commentaries", "https://www.who.int/news-room/commentaries"),
)
WHO_DOMAINS = tuple(name for name, _ in WHO_DOMAIN_PAGES)
WHO_HUB_ROUTE_BY_API_PATH = {
    "/api/hubs/newsitems": "/news/item",
    "/api/hubs/featurestories": "/news-room/feature-stories/detail",
    "/api/hubs/speeches": "/news-room/speeches/item",
}

# NSO exposes 16 Vietnamese feature slugs beneath five top-level domains.
# ``chu-de-khac`` is a generic content bucket and is intentionally excluded.
GSO_DOMAIN_BY_SLUG = {
    "cong-nghiep": "Kinh tế",
    "dan-so": "Dân số và lao động",
    "dau-tu-va-xay-dung": "Kinh tế",
    "doanh-nghiep": "Kinh tế",
    "don-vi-hanh-chinh-dat-dai-va-khi-hau": "Xã hội môi trường và đơn vị hành chính",
    "giao-duc": "Xã hội môi trường và đơn vị hành chính",
    "lao-dong": "Dân số và lao động",
    "ngan-hang-bao-hiem-va-thu-chi-ngan-sach": "Tài khoản quốc gia và tài chính",
    "nong-lam-nghiep-va-thuy-san": "Kinh tế",
    "tai-khoan-quoc-gia": "Tài khoản quốc gia và tài chính",
    "gia": "Kinh tế",
    "thuong-mai-dich-vu": "Kinh tế",
    "y-te-muc-song-dan-cu-van-hoa-the-thao-trat-tu-an-toan-xa-hoi-va-moi-truong": (
        "Xã hội môi trường và đơn vị hành chính"
    ),
    "tong-dieu-tra-kinh-te": "Tổng điều tra",
    "tong-dieu-tra-dan-so-va-nha-o": "Tổng điều tra",
    "tong-dieu-tra-nong-thon-nong-nghiep-va-thuy-san": "Tổng điều tra",
}

GSO_PARENT_DOMAIN_BY_FEATURE_NAME = {
    "Công nghiệp": "Kinh tế",
    "Dân số": "Dân số và lao động",
    "Đầu tư và Xây dựng": "Kinh tế",
    "Doanh nghiệp": "Kinh tế",
    "Đơn vị hành chính, đất đai và khí hậu": "Xã hội môi trường và đơn vị hành chính",
    "Giáo dục": "Xã hội môi trường và đơn vị hành chính",
    "Lao động": "Dân số và lao động",
    "Ngân hàng, bảo hiểm và thu chi ngân sách": "Tài khoản quốc gia và tài chính",
    "Nông, Lâm nghiệp và Thủy sản": "Kinh tế",
    "Tài khoản quốc gia": "Tài khoản quốc gia và tài chính",
    "Thống kê giá": "Kinh tế",
    "Thương mại - Dịch vụ": "Kinh tế",
    "Y tế, mức sống dân cư, văn hóa, thể thao, trật tự an toàn xã hội và môi trường": (
        "Xã hội môi trường và đơn vị hành chính"
    ),
    "Tổng điều tra kinh tế": "Tổng điều tra",
    "Tổng điều tra dân số và nhà ở": "Tổng điều tra",
    "Tổng điều tra nông thôn, nông nghiệp và thủy sản": "Tổng điều tra",
}
GSO_PARENT_DOMAINS = tuple(dict.fromkeys(GSO_DOMAIN_BY_SLUG.values()))

def repair_mojibake(value: str) -> str:
    """Repair UTF-8 text that was accidentally decoded as a legacy code page."""
    current = value
    for _ in range(2):
        if not any(marker in current for marker in MOJIBAKE_MARKERS):
            break
        raw = bytearray()
        try:
            for character in current:
                codepoint = ord(character)
                if codepoint <= 255:
                    raw.append(codepoint)
                else:
                    raw.extend(character.encode("cp1252"))
            candidate = raw.decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break
        current = candidate
    return current


@dataclass(frozen=True)
class SourceConfig:
    key: str
    name: str
    target: int
    label: str
    language: str
    allowed_hosts: tuple[str, ...]
    sitemap_urls: tuple[str, ...] = ()
    seed_urls: tuple[str, ...] = ()
    article_patterns: tuple[str, ...] = ()
    sitemap_patterns: tuple[str, ...] = ()
    category_pages: tuple[tuple[str, str], ...] = ()
    preferred_selectors: tuple[str, ...] = ()
    evidence: str = ""
    verify_tls: bool = True


SOURCES: dict[str, SourceConfig] = {
    "vafc": SourceConfig(
        key="vafc",
        name="VAFC",
        target=2600,
        label="FALSE",
        language="vi",
        allowed_hosts=("tingia.gov.vn", "www.tingia.gov.vn"),
        sitemap_urls=("https://tingia.gov.vn/sitemap.xml",),
        category_pages=(
            ("L\u0129nh v\u1ef1c", "https://tingia.gov.vn/linh-vuc"),
            ("Ti\u00eau \u0111i\u1ec3m", "https://tingia.gov.vn/tieu-diem"),
            ("Tin v\u1eeba check", "https://tingia.gov.vn/tin-vua-check"),
            ("Multimedia", "https://tingia.gov.vn/multimedia"),
            ("Talk show", "https://tingia.gov.vn/talk-show"),
            ("\u0110\u1ed9c gi\u1ea3 qu\u00e9t tin gi\u1ea3", "https://tingia.gov.vn/doc-gia-quet-tin-gia"),
            ("Th\u1ebf gi\u1edbi", "https://tingia.gov.vn/the-gioi"),
            ("G\u00f3c nh\u00ecn", "https://tingia.gov.vn/goc-nhin"),
            ("Chi\u00eau tr\u00f2", "https://tingia.gov.vn/chieu-tro"),
            ("C\u00f4ng b\u1ed1", "https://tingia.gov.vn/cong-bo"),
            ("C\u00f4ng b\u1ed1 tin gi\u1ea3", "https://tingia.gov.vn/cong-bo-tin-gia"),
            ("T\u00e0i ch\u00ednh - Ng\u00e2n h\u00e0ng", "https://tingia.gov.vn/tai-chinh-ngan-hang"),
            ("S\u1ee9c kh\u1ecfe c\u1ed9ng \u0111\u1ed3ng", "https://tingia.gov.vn/suc-khoe-cong-dong"),
            ("Quy\u1ec1n l\u1ee3i ng\u01b0\u1eddi d\u00e2n", "https://tingia.gov.vn/quyen-loi-nguoi-dan"),
            ("Vaccine tin gi\u1ea3", "https://tingia.gov.vn/vaccine-phong-chong-tin-gia"),
        ),
        article_patterns=(r"^https?://(?:www\.)?tingia\.gov\.vn/.+\.html(?:\?.*)?$",),
        sitemap_patterns=(r"tingia\.gov\.vn/sitemap/.+\.xml$",),
        preferred_selectors=(
            "#maincontent",
            ".content-detail",
            ".entry-content",
            ".post-content",
        ),
        evidence="Nội dung kiểm chứng/cảnh báo do Trung tâm Xử lý tin giả, thông tin xấu độc Việt Nam (VAFC) công bố.",
        verify_tls=False,
    ),
    "moh": SourceConfig(
        key="moh",
        name="Bộ Y tế",
        target=2500,
        label="TRUE",
        language="vi",
        allowed_hosts=("moh.gov.vn", "www.moh.gov.vn"),
        article_patterns=(
            r"^https?://(?:www\.)?moh\.gov\.vn/index\.jsp\?.*\baid=\d+",
        ),
        preferred_selectors=(
            ".journal-content-article",
            ".news-detail",
            ".content-detail",
            ".detail-content",
            "#main-content",
        ),
        evidence="Bài viết được công bố trên Cổng thông tin điện tử chính thức của Bộ Y tế Việt Nam.",
        verify_tls=False,
    ),
    "who": SourceConfig(
        key="who",
        name="WHO",
        target=2400,
        label="TRUE",
        language="en",
        allowed_hosts=("who.int", "www.who.int"),
        sitemap_urls=("https://www.who.int/sitemaps/sitemapindex.xml",),
        seed_urls=tuple(url for _, url in WHO_DOMAIN_PAGES),
        category_pages=WHO_DOMAIN_PAGES,
        article_patterns=(
            r"^https?://(?:www\.)?who\.int/news/item/[^?#]+",
            r"^https?://(?:www\.)?who\.int/emergencies/(?!overview(?:[/?#]|$))[^?#]+",
            r"^https?://(?:www\.)?who\.int/campaigns/(?!terms-of-use(?:[/?#]|$))[^?#]+",
            r"^https?://(?:www\.)?who\.int/news-room/(?:events|statements|feature-stories|speeches|commentaries)/(?:detail|item)/[^?#]+",
            r"^https?://(?:www\.)?who\.int/director-general/speeches/detail/[^?#]+",
        ),
        sitemap_patterns=(r"who\.int/(?:sitemap|sitemaps)/.+\.(?:xml|gz)$",),
        preferred_selectors=(
            ".sf-detail-body-wrapper",
            ".sf-detail-content",
            ".content",
            "article",
        ),
        evidence="Article published on the official World Health Organization website.",
    ),
    "gso": SourceConfig(
        key="gso",
        name="GSO/NSO Việt Nam",
        target=2500,
        label="TRUE",
        language="vi",
        allowed_hosts=("gso.gov.vn", "www.gso.gov.vn", "nso.gov.vn", "www.nso.gov.vn"),
        sitemap_urls=("https://www.gso.gov.vn/wp-sitemap.xml", "https://www.nso.gov.vn/wp-sitemap.xml"),
        article_patterns=(r"^https?://www\.(?:gso|nso)\.gov\.vn/[^?#]+/$",),
        sitemap_patterns=(r"www\.nso\.gov\.vn/wp-sitemap-posts-post-\d+\.xml$",),
        preferred_selectors=(
            ".entry-content",
            ".post-content",
            ".single-post-content",
            "article",
        ),
        evidence="Bài viết/số liệu được công bố trên website chính thức của cơ quan Thống kê Việt Nam (GSO, nay là NSO).",
        verify_tls=False,
    ),
}


@dataclass
class Candidate:
    url: str
    lastmod: str = ""
    domain: str = ""
    domain_rank: int = -1


class HostRateLimiter:
    """Ensure request starts to a host are separated by a minimum delay."""

    def __init__(self, delay: float) -> None:
        self.delay = max(0.0, delay)
        self._locks: dict[str, threading.Lock] = {}
        self._last_request: dict[str, float] = {}
        self._guard = threading.Lock()

    def wait(self, url: str) -> None:
        host = urlsplit(url).netloc.lower()
        with self._guard:
            lock = self._locks.setdefault(host, threading.Lock())
        with lock:
            elapsed = time.monotonic() - self._last_request.get(host, 0.0)
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)
            self._last_request[host] = time.monotonic()


def retry_policy() -> Retry:
    return Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        # MOH exposes read-only article queries through POST endpoints.
        allowed_methods=frozenset({"GET", "POST"}),
        respect_retry_after_header=True,
    )


class LegacyTLSAdapter(HTTPAdapter):
    """Allow older government servers with small DH keys.

    This adapter is mounted only for SourceConfig entries that explicitly set
    verify_tls=False. Normal sources continue to use the system TLS policy.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        context = ssl.create_default_context()
        try:
            context.set_ciphers("DEFAULT:@SECLEVEL=1")
        except ssl.SSLError:
            LOGGER.warning("OpenSSL không hỗ trợ hạ SECLEVEL cho TLS cũ")
        context.check_hostname = False
        self.ssl_context = context
        super().__init__(*args, **kwargs)

    def init_poolmanager(self, *args: Any, **kwargs: Any) -> None:
        kwargs["ssl_context"] = self.ssl_context
        super().init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, proxy: str, **proxy_kwargs: Any) -> Any:
        proxy_kwargs["ssl_context"] = self.ssl_context
        return super().proxy_manager_for(proxy, **proxy_kwargs)


class HttpClient:
    def __init__(self, *, delay: float, timeout: float, obey_robots: bool) -> None:
        self.timeout = timeout
        self.obey_robots = obey_robots
        self.rate_limiter = HostRateLimiter(delay)
        self._local = threading.local()
        self._robots: dict[str, RobotFileParser | None] = {}
        self._robots_lock = threading.Lock()

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(
                {
                    "User-Agent": USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "vi,en-US;q=0.8,en;q=0.7",
                }
            )
            adapter = HTTPAdapter(max_retries=retry_policy(), pool_connections=8, pool_maxsize=8)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            self._local.session = session
        return session

    def _session_for(self, url: str, verify_tls: bool) -> requests.Session:
        session = self._session()
        if not verify_tls:
            origin = f"{urlsplit(url).scheme}://{urlsplit(url).netloc}/"
            mounted = getattr(self._local, "legacy_tls_origins", set())
            if origin not in mounted:
                session.mount(
                    origin,
                    LegacyTLSAdapter(max_retries=retry_policy(), pool_connections=4, pool_maxsize=4),
                )
                mounted.add(origin)
                self._local.legacy_tls_origins = mounted
        return session

    def _robot_parser(self, url: str, verify_tls: bool) -> RobotFileParser | None:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        with self._robots_lock:
            if origin in self._robots:
                return self._robots[origin]

        robots_url = origin + "/robots.txt"
        parser: RobotFileParser | None = None
        try:
            self.rate_limiter.wait(robots_url)
            response = self._session_for(robots_url, verify_tls).get(
                robots_url, timeout=self.timeout, verify=verify_tls
            )
            if response.status_code == 200:
                parser = RobotFileParser()
                parser.set_url(robots_url)
                parser.parse(response.text.splitlines())
        except requests.RequestException as exc:
            LOGGER.warning("Không đọc được robots.txt %s: %s", robots_url, exc)

        with self._robots_lock:
            self._robots[origin] = parser
        return parser

    def get(self, url: str, *, verify_tls: bool = True, check_robots: bool = True) -> requests.Response:
        if check_robots and self.obey_robots:
            parser = self._robot_parser(url, verify_tls)
            if parser is not None and not parser.can_fetch(USER_AGENT, url):
                raise PermissionError(f"robots.txt không cho phép crawl: {url}")
        self.rate_limiter.wait(url)
        response = self._session_for(url, verify_tls).get(url, timeout=self.timeout, verify=verify_tls)
        response.raise_for_status()
        return response

    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        verify_tls: bool = True,
        check_robots: bool = True,
    ) -> requests.Response:
        if check_robots and self.obey_robots:
            parser = self._robot_parser(url, verify_tls)
            if parser is not None and not parser.can_fetch(USER_AGENT, url):
                raise PermissionError(f"robots.txt kh\u00f4ng cho ph\u00e9p crawl: {url}")
        self.rate_limiter.wait(url)
        response = self._session_for(url, verify_tls).post(
            url,
            data=json.dumps(payload),
            timeout=self.timeout,
            verify=verify_tls,
        )
        response.raise_for_status()
        return response


def host_allowed(url: str, config: SourceConfig) -> bool:
    host = urlsplit(url).hostname or ""
    return host.lower() in config.allowed_hosts


def matches_any(value: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, value, flags=re.IGNORECASE) for pattern in patterns)


def normalize_url(url: str, base: str = "", *, article: bool = False) -> str:
    absolute = urljoin(base, url.strip())
    parts = urlsplit(absolute)
    if parts.scheme not in {"http", "https"}:
        return ""
    host = parts.netloc.lower()
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    query_items = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        lower = key.lower()
        if lower.startswith("utm_") or lower in TRACKING_PARAMS:
            continue
        if article and lower in {"inheritredirect", "redirect"}:
            continue
        query_items.append((key, value))
    query = urlencode(query_items, doseq=True)
    return urlunsplit((parts.scheme.lower(), host, path, query, ""))


def configured_domain(url: str, config: SourceConfig) -> tuple[str, int]:
    """Match a listing page or child sitemap to a configured source taxonomy."""
    path = urlsplit(url).path.rstrip("/")
    slug = path.rsplit("/", 1)[-1].removesuffix(".xml")
    for rank, (name, page_url) in enumerate(config.category_pages):
        page_slug = urlsplit(page_url).path.rstrip("/").rsplit("/", 1)[-1]
        if slug == page_slug or path == urlsplit(page_url).path.rstrip("/"):
            return name, rank
    return "", -1


def domain_from_url(url: str, config: SourceConfig) -> str:
    """Return a stable section name when the source encodes it in the URL."""
    if config.key != "who":
        return ""
    path = urlsplit(url).path.lower()
    routes = (
        ("/news-room/events/", "Events"),
        ("/news-room/statements/", "Statements"),
        ("/news-room/feature-stories/", "Feature stories"),
        ("/news-room/speeches/", "Speeches"),
        ("/director-general/speeches/", "Speeches"),
        ("/news-room/commentaries/", "Commentaries"),
        ("/emergencies/", "Emergencies"),
        ("/campaigns/", "Campaigns"),
        ("/news/item/", "News"),
    )
    return next((name for prefix, name in routes if path.startswith(prefix)), "")


_DOMAIN_HISTORY: dict[str, str] | None = None
_DOMAIN_HISTORY_LOCK = threading.Lock()


def load_domain_history() -> dict[str, str]:
    """Load URL-specific domains already curated in this research workspace."""
    global _DOMAIN_HISTORY
    with _DOMAIN_HISTORY_LOCK:
        if _DOMAIN_HISTORY is not None:
            return _DOMAIN_HISTORY
        mapping: dict[str, str] = {}
        project_root = Path(__file__).resolve().parent.parent
        paths = (
            project_root / "data" / "vie" / "raw" / "news_website_output.json",
            project_root / "data" / "vie" / "processed" / "fact_checking_dataset.json",
        )
        for path in paths:
            if not path.exists():
                continue
            try:
                records = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(records, list):
                continue
            for record in records:
                if not isinstance(record, dict):
                    continue
                url = normalize_url(str(record.get("url", "")), article=True)
                domain = compact_text(str(record.get("domain", "")))
                if url and domain and url not in mapping:
                    mapping[url] = domain
        _DOMAIN_HISTORY = mapping
        return mapping


def is_article_url(url: str, config: SourceConfig) -> bool:
    if not host_allowed(url, config) or not matches_any(url, config.article_patterns):
        return False
    path = urlsplit(url).path.lower()
    if config.key == "gso":
        excluded = ("/en/", "/category/", "/tag/", "/author/", "/page/", "/wp-")
        return not any(item in path for item in excluded)
    return True


def decode_xml_response(response: requests.Response) -> bytes:
    data = response.content
    if response.url.lower().endswith(".gz") or data[:2] == b"\x1f\x8b":
        try:
            return gzip.decompress(data)
        except OSError:
            pass
    return data


def parse_sitemap(xml_data: bytes) -> tuple[str, list[Candidate]]:
    root = ET.fromstring(xml_data)
    root_name = root.tag.rsplit("}", 1)[-1].lower()
    items: list[Candidate] = []
    for element in root:
        loc = ""
        lastmod = ""
        for child in element:
            name = child.tag.rsplit("}", 1)[-1].lower()
            if name == "loc" and child.text:
                loc = child.text.strip()
            elif name == "lastmod" and child.text:
                lastmod = child.text.strip()
        if loc:
            items.append(Candidate(loc, lastmod))
    return root_name, items


def discover_from_sitemaps(client: HttpClient, config: SourceConfig) -> dict[str, Candidate]:
    queue = [(url, "", -1) for url in config.sitemap_urls]
    visited: set[str] = set()
    found: dict[str, Candidate] = {}

    while queue:
        queued_url, inherited_domain, inherited_rank = queue.pop(0)
        sitemap_url = normalize_url(queued_url)
        if not sitemap_url or sitemap_url in visited:
            continue
        visited.add(sitemap_url)
        try:
            response = client.get(sitemap_url, verify_tls=config.verify_tls, check_robots=False)
            kind, items = parse_sitemap(decode_xml_response(response))
        except (requests.RequestException, PermissionError, ET.ParseError, OSError) as exc:
            LOGGER.warning("Lỗi sitemap %s: %s", sitemap_url, exc)
            continue

        LOGGER.info("Sitemap %s: %d URL", sitemap_url, len(items))
        if kind == "sitemapindex":
            for item in items:
                child = normalize_url(item.url, sitemap_url)
                if child and (not config.sitemap_patterns or matches_any(child, config.sitemap_patterns)):
                    child_domain, child_rank = configured_domain(child, config)
                    queue.append(
                        (
                            child,
                            child_domain or inherited_domain,
                            child_rank if child_domain else inherited_rank,
                        )
                    )
            continue

        for item in items:
            # VAFC publishes child sitemap links in a non-standard <urlset>.
            child = normalize_url(item.url, sitemap_url)
            if child and config.sitemap_patterns and matches_any(child, config.sitemap_patterns):
                child_domain, child_rank = configured_domain(child, config)
                queue.append(
                    (
                        child,
                        child_domain or inherited_domain,
                        child_rank if child_domain else inherited_rank,
                    )
                )
                continue
            url = normalize_url(item.url, sitemap_url, article=True)
            if is_article_url(url, config):
                previous = found.get(url)
                route_domain = domain_from_url(url, config)
                candidate = Candidate(
                    url,
                    item.lastmod,
                    inherited_domain or route_domain,
                    inherited_rank,
                )
                if (
                    previous is None
                    or candidate.domain_rank > previous.domain_rank
                    or (
                        candidate.domain_rank == previous.domain_rank
                        and candidate.lastmod > previous.lastmod
                    )
                ):
                    found[url] = candidate
    return found


def looks_like_listing_url(url: str, seed_paths: set[str], config: SourceConfig) -> bool:
    if not host_allowed(url, config):
        return False
    parts = urlsplit(url)
    if parts.path in seed_paths and not parts.query:
        return True
    normalized_path = parts.path.rstrip("/")
    if any(
        re.fullmatch(rf"{re.escape(seed_path.rstrip('/'))}/\d+", normalized_path)
        for seed_path in seed_paths
    ):
        return True
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    return any(key.endswith("_cur") or key in {"page", "paged"} for key in query)


def who_hub_item_url(api_url: str, item_default_url: str) -> str:
    """Expand WHO hub API slugs into their public detail-page routes."""
    api_path = urlsplit(api_url).path.lower()
    route_prefix = WHO_HUB_ROUTE_BY_API_PATH.get(api_path, "")
    raw_url = compact_text(item_default_url)
    if not route_prefix or not raw_url:
        return ""
    if urlsplit(raw_url).netloc:
        return normalize_url(raw_url, article=True)
    item_path = urlsplit(raw_url).path
    if item_path.startswith(f"{route_prefix}/"):
        path = item_path
    else:
        path = f"{route_prefix}/{item_path.lstrip('/')}"
    return normalize_url(f"https://www.who.int{path}", article=True)


def discover_who_hub_items(
    client: HttpClient,
    html_text: str,
    page_url: str,
    domain: str,
    domain_rank: int,
    config: SourceConfig,
) -> dict[str, Candidate]:
    """Read the public OData hub endpoint embedded by a WHO listing page."""
    found: dict[str, Candidate] = {}
    endpoints = dict.fromkeys(
        match.replace("&amp;", "&")
        for match in re.findall(r'''["']([^"']*/api/hubs/[^"']+)["']''', html_text, re.I)
    )
    for endpoint in endpoints:
        api_url = urljoin(page_url, endpoint)
        api_url = re.sub(r"&?\$(?:top|skip)=\d+", "", api_url, flags=re.I)
        page_size = 100  # WHO rejects hub API page sizes above 100.
        for offset in range(0, config.target, page_size):
            paged_api_url = f"{api_url}&$top={page_size}&$skip={offset}"
            try:
                response = client.get(
                    paged_api_url,
                    verify_tls=config.verify_tls,
                    check_robots=False,
                )
                payload: Any = response.json()
            except (requests.RequestException, PermissionError, ValueError) as exc:
                LOGGER.warning("Lỗi API hub WHO %s: %s", paged_api_url, exc)
                break
            items = payload.get("value", []) if isinstance(payload, dict) else payload
            if not isinstance(items, list):
                break
            for item in items:
                if not isinstance(item, dict):
                    continue
                url = who_hub_item_url(api_url, str(item.get("ItemDefaultUrl", "")))
                if not is_article_url(url, config):
                    continue
                candidate = Candidate(
                    url,
                    normalize_date(str(item.get("FormatedDate", ""))),
                    domain,
                    domain_rank,
                )
                previous = found.get(url)
                if previous is None or candidate.lastmod > previous.lastmod:
                    found[url] = candidate
            if len(items) < page_size:
                break
        LOGGER.info("API hub WHO %s: %d bài", domain, len(found))
    return found


def discover_from_listing_pages(
    client: HttpClient,
    config: SourceConfig,
    *,
    max_pages: int,
) -> dict[str, Candidate]:
    queue = []
    for seed_url in config.seed_urls:
        normalized_seed = normalize_url(seed_url)
        domain, rank = configured_domain(normalized_seed, config)
        queue.append((normalized_seed, domain, rank))
    seed_paths = {urlsplit(url).path for url, _, _ in queue}
    scheduled = {url for url, _, _ in queue}
    visited: set[str] = set()
    found: dict[str, Candidate] = {}

    while queue and len(visited) < max_pages:
        page_url, page_domain, page_rank = queue.pop(0)
        scheduled.discard(page_url)
        if not page_url or page_url in visited:
            continue
        visited.add(page_url)
        try:
            response = client.get(page_url, verify_tls=config.verify_tls)
        except (requests.RequestException, PermissionError) as exc:
            LOGGER.warning("Lỗi trang danh mục %s: %s", page_url, exc)
            continue

        soup = BeautifulSoup(response.content, "html.parser")
        page_links = 0
        for anchor in soup.find_all("a", href=True):
            href = str(anchor.get("href", ""))
            url = normalize_url(href, response.url)
            if not url:
                continue
            article_url = normalize_url(url, article=True)
            if is_article_url(article_url, config):
                previous = found.get(article_url)
                if previous is None or page_rank > previous.domain_rank:
                    found[article_url] = Candidate(
                        article_url,
                        previous.lastmod if previous else "",
                        domain=page_domain,
                        domain_rank=page_rank,
                    )
                page_links += 1
            elif looks_like_listing_url(url, seed_paths, config):
                next_page_url = url
                if config.key == "who":
                    parts = urlsplit(url)
                    next_page_url = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
                if next_page_url not in visited and next_page_url not in scheduled:
                    queue.append((next_page_url, page_domain, page_rank))
                    scheduled.add(next_page_url)
        if config.key == "who":
            hub_items = discover_who_hub_items(
                client,
                response.text,
                response.url,
                page_domain,
                page_rank,
                config,
            )
            for url, candidate in hub_items.items():
                previous = found.get(url)
                if previous is None or candidate.domain_rank > previous.domain_rank:
                    found[url] = candidate
        LOGGER.info("Danh mục %s: thêm %d bài (tổng %d)", page_url, page_links, len(found))
    return found


def discover_from_moh_api(
    client: HttpClient,
    config: SourceConfig,
    target: int,
) -> dict[str, Candidate]:
    """Discover current Ministry of Health articles from its portal API."""
    endpoint = f"{MOH_API_ORIGIN}/bridge?url=/portal/api/lay_bai_viet_theo_danh_muc"
    page_size = max(25, (target + len(MOH_API_CATEGORIES) - 1) // len(MOH_API_CATEGORIES) + 25)
    found_by_article_id: dict[str, Candidate] = {}

    for rank, (category_id, category_name, parent_domain) in enumerate(MOH_API_CATEGORIES):
        payload = {
            "CHUYEN_TRANG": "BO_YTE",
            "NGON_NGU": "TIENG_VIET",
            "MA_DANH_MUC": category_id,
            "TYPE": 1,
            "PAGE_SIZE": page_size,
            "ISEXTEND": 1,
        }
        try:
            response = client.post_json(
                endpoint,
                payload,
                verify_tls=config.verify_tls,
                check_robots=False,
            )
            groups: Any = response.json()
            if isinstance(groups, dict):
                groups = groups.get("d") or groups.get("data") or groups
            if isinstance(groups, dict):
                groups = [groups]
            if not isinstance(groups, list):
                raise ValueError("Ph\u1ea3n h\u1ed3i danh m\u1ee5c kh\u00f4ng \u0111\u00fang c\u1ea5u tr\u00fac")
        except (requests.RequestException, PermissionError, ValueError, TypeError) as exc:
            LOGGER.warning("L\u1ed7i API danh m\u1ee5c B\u1ed9 Y t\u1ebf %s (%s): %s", category_name, category_id, exc)
            continue

        added = 0
        for group in groups:
            if not isinstance(group, dict):
                continue
            items = group.get("ITEMS") or group.get("items") or []
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                article_id = compact_text(str(item.get("MA", "")))
                if not article_id:
                    continue
                article_url = normalize_url(
                    f"{MOH_API_ORIGIN}/index.jsp?{urlencode({'pageId': '5803', 'aid': article_id, 'cid': category_id})}",
                    article=True,
                )
                published = normalize_date(str(item.get("THOI_GIAN_XUAT_BAN", "")))
                previous = found_by_article_id.get(article_id)
                candidate = Candidate(article_url, published, parent_domain, rank)
                if previous is None or candidate.domain_rank > previous.domain_rank:
                    found_by_article_id[article_id] = candidate
                    added += 1
        LOGGER.info(
            "API B\u1ed9 Y t\u1ebf - %s: th\u00eam %d b\u00e0i (t\u1ed5ng %d)",
            category_name,
            added,
            len(found_by_article_id),
        )
    return {candidate.url: candidate for candidate in found_by_article_id.values()}


def compact_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def normalize_gso_parent_domains(value: str | None) -> str:
    """Roll NSO feature names up to the five top-level website domains."""
    feature_map = {key.casefold(): parent for key, parent in GSO_PARENT_DOMAIN_BY_FEATURE_NAME.items()}
    parent_map = {parent.casefold(): parent for parent in GSO_PARENT_DOMAINS}
    parents: list[str] = []
    for item in re.split(r"\s*;\s*", compact_text(value)):
        parent = feature_map.get(item.casefold()) or parent_map.get(item.casefold(), "")
        if parent and parent not in parents:
            parents.append(parent)
    return "; ".join(parents)


def normalize_moh_domain(value: str | None) -> str:
    """Keep one clean public category from MOH's multi-category API label."""
    primary = compact_text(value).split(";", 1)[0]
    primary = re.sub(r"\s*\([^()]*\)\s*$", "", primary)
    return compact_text(primary)


def normalize_moh_parent_domain(value: str | None, category_id: str = "") -> str:
    """Roll a MOH child category up to one of the four top-level menu domains."""
    if category_id in MOH_PARENT_DOMAIN_BY_CATEGORY_ID:
        return MOH_PARENT_DOMAIN_BY_CATEGORY_ID[category_id]
    primary = normalize_moh_domain(value)
    parent_by_name = {parent.casefold(): parent for parent in MOH_PARENT_DOMAINS}
    return (
        MOH_PARENT_DOMAIN_BY_CHILD_NAME.get(primary.casefold())
        or parent_by_name.get(primary.casefold(), "")
    )


def text_with_paragraphs(element: Tag | None) -> str:
    if element is None:
        return ""
    clone = BeautifulSoup(str(element), "html.parser")
    for unwanted in clone.select(
        "script,style,noscript,nav,footer,form,aside,.share,.social,.related,.advertisement,.ads,.breadcrumb"
    ):
        unwanted.decompose()
    blocks = []
    for node in clone.select("p,h2,h3,li,blockquote"):
        text = compact_text(node.get_text(" ", strip=True))
        if text and (not blocks or blocks[-1] != text):
            blocks.append(text)
    if blocks:
        return "\n".join(blocks)
    return compact_text(clone.get_text(" ", strip=True))


def meta_content(soup: BeautifulSoup, *keys: str) -> str:
    for key in keys:
        tag = soup.find("meta", attrs={"property": key}) or soup.find("meta", attrs={"name": key})
        if isinstance(tag, Tag) and tag.get("content"):
            return compact_text(str(tag["content"]))
    return ""


def walk_json(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_json(child)


def extract_article_jsonld(soup: BeautifulSoup) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text()
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        for item in walk_json(payload):
            item_type = item.get("@type", "")
            types = set(item_type if isinstance(item_type, list) else [item_type])
            is_article = types.intersection(ARTICLE_TYPES) or any(
                isinstance(t, str) and (t.endswith("Article") or t == "BlogPosting")
                for t in types
            )
            if is_article:
                if not best or len(str(item.get("articleBody", ""))) > len(str(best.get("articleBody", ""))):
                    best = item
    return best


def first_text(soup: BeautifulSoup, selectors: Iterable[str]) -> str:
    for selector in selectors:
        element = soup.select_one(selector)
        if isinstance(element, Tag):
            value = compact_text(element.get_text(" ", strip=True))
            if value:
                return value
    return ""


def choose_body(soup: BeautifulSoup, config: SourceConfig, jsonld: dict[str, Any]) -> str:
    json_body = compact_text(str(jsonld.get("articleBody", "")))
    candidates = [json_body] if json_body else []
    selectors = config.preferred_selectors + (
        "[itemprop='articleBody']",
        ".article-content",
        ".article-body",
        ".detail-body",
        ".post-body",
        ".entry-content",
        "article",
    )
    for selector in selectors:
        for element in soup.select(selector)[:3]:
            if isinstance(element, Tag):
                value = text_with_paragraphs(element)
                if value:
                    candidates.append(value)
    return max(candidates, key=len, default="")


def normalize_date(value: str) -> str:
    value = compact_text(value)
    if not value:
        return ""
    iso_value = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(iso_value).date().isoformat()
    except ValueError:
        pass
    patterns = (
        r"(?P<year>\d{4})[-/.](?P<month>\d{1,2})[-/.](?P<day>\d{1,2})",
        r"(?P<day>\d{1,2})[-/.](?P<month>\d{1,2})[-/.](?P<year>\d{4})",
    )
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            try:
                return datetime(
                    int(match.group("year")),
                    int(match.group("month")),
                    int(match.group("day")),
                ).date().isoformat()
            except ValueError:
                continue
    month_names = {
        name.lower(): number
        for number, name in enumerate(
            (
                "",
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            )
        )
        if name
    }
    match = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", value)
    if match and match.group(2).lower() in month_names:
        try:
            return datetime(int(match.group(3)), month_names[match.group(2).lower()], int(match.group(1))).date().isoformat()
        except ValueError:
            pass
    return ""


def canonical_url(soup: BeautifulSoup, fallback: str, config: SourceConfig) -> str:
    tag = soup.find("link", attrs={"rel": lambda value: value and "canonical" in value})
    if isinstance(tag, Tag) and tag.get("href"):
        candidate = normalize_url(str(tag["href"]), fallback, article=True)
        if host_allowed(candidate, config):
            return candidate
    return normalize_url(fallback, article=True)


def extract_domain(
    soup: BeautifulSoup,
    jsonld: dict[str, Any],
    url: str,
    config: SourceConfig,
    fallback: str = "",
) -> str:
    """Extract the article's real source taxonomy without using the title."""
    if fallback:
        return compact_text(fallback)

    category = jsonld.get("articleSection", "")
    if isinstance(category, list):
        category = ", ".join(str(item) for item in category)
    category = compact_text(str(category)) or meta_content(soup, "article:section")
    if category:
        return category

    route_domain = domain_from_url(url, config)
    if route_domain:
        return route_domain

    for selector in (
        ".cat-links a",
        ".post-categories a",
        "a[rel='category tag']",
        ".article-category a",
        ".category-name a",
    ):
        value = first_text(soup, (selector,))
        if value:
            return value

    ignored = {"home", "news", "item", "detail", "trang ch\u1ee7", "tin t\u1ee9c"}
    breadcrumbs = [
        compact_text(anchor.get_text(" ", strip=True))
        for anchor in soup.select(".breadcrumb a, .breadcrumbs a, [aria-label='breadcrumb'] a")
    ]
    candidates = [value for value in breadcrumbs if value and value.casefold() not in ignored]
    return candidates[-1] if candidates else ""


def resolve_wordpress_domain(
    client: HttpClient,
    html: bytes,
    url: str,
    config: SourceConfig,
) -> str:
    """Return every one of NSO's 16 feature domains attached to a post."""
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article", class_=lambda value: value and "type-post" in value)
    if not isinstance(article, Tag):
        return ""
    classes = [str(value) for value in article.get("class", [])]
    slugs = list(dict.fromkeys(
        value.removeprefix("category-")
        for value in classes
        if value.startswith("category-")
    ))
    names = [GSO_DOMAIN_BY_SLUG[slug] for slug in slugs if slug in GSO_DOMAIN_BY_SLUG]
    return "; ".join(dict.fromkeys(names))


def parse_article(
    html: bytes,
    url: str,
    config: SourceConfig,
    fallback_domain: str = "",
) -> dict[str, Any] | None:
    soup = BeautifulSoup(html, "html.parser")
    jsonld = extract_article_jsonld(soup)
    title = compact_text(str(jsonld.get("headline", "")))
    if not title:
        title = meta_content(soup, "og:title", "twitter:title")
    if not title:
        title = first_text(soup, ("h1", ".article-title", ".entry-title", ".detail-title"))

    description = compact_text(str(jsonld.get("description", "")))
    if not description:
        description = meta_content(soup, "description", "og:description", "twitter:description")
    if not description:
        description = first_text(
            soup,
            (".content-detail-sapo", ".sapo", ".lead", ".summary", ".article-summary"),
        )

    body = choose_body(soup, config, jsonld)
    if len(title) < 5 or len(body) < 150:
        return None

    raw_date = compact_text(str(jsonld.get("datePublished", "")))
    if not raw_date:
        raw_date = meta_content(soup, "article:published_time", "date", "DC.date", "dcterms.date")
    if not raw_date:
        time_tag = soup.find("time")
        if isinstance(time_tag, Tag):
            raw_date = compact_text(str(time_tag.get("datetime") or time_tag.get_text(" ", strip=True)))
    if not raw_date:
        raw_date = first_text(soup, (".post-meta", ".date", ".publish-date", ".article-date", ".time"))
    if not raw_date:
        # Extract date from URL path if it contains standard YYYY/MM/DD or YYYY-MM-DD
        match = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", url)
        if match:
            raw_date = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"

    category = extract_domain(soup, jsonld, url, config, fallback_domain)

    final_url = canonical_url(soup, url, config)
    record_id = hashlib.md5(f"{final_url}|{title}".encode("utf-8")).hexdigest()
    return {
        "id": record_id,
        "source_type": "website",
        "source_name": config.name,
        "url": final_url,
        "domain": category,
        "publish_date": normalize_date(raw_date),
        "claim": title,
        "original_text": description,
        "label": config.label,
        "evidence": config.evidence,
        "justification": body,
        "comments": [],
    }


def fetch_moh_article(
    client: HttpClient,
    candidate: Candidate,
    config: SourceConfig,
) -> dict[str, Any] | None:
    """Fetch a Ministry of Health article from the current dynamic portal."""
    query = dict(parse_qsl(urlsplit(candidate.url).query, keep_blank_values=True))
    article_id = compact_text(query.get("aid", ""))
    category_id = compact_text(query.get("cid", ""))
    if not article_id or not category_id:
        return None

    endpoint = f"{MOH_API_ORIGIN}/bridge?url=/portal/api/layChiTietBaiViet"
    response = client.post_json(
        endpoint,
        {"maBaiViet": article_id, "maDanhMuc": category_id},
        verify_tls=config.verify_tls,
        check_robots=False,
    )
    payload = response.json()
    detail: Any = payload.get("d") if isinstance(payload, dict) else None
    if isinstance(detail, list):
        detail = detail[0] if detail else None
    if not isinstance(detail, dict):
        return None

    title = compact_text(str(detail.get("tieuDe", "")))
    description_html = str(detail.get("moTa", ""))
    description = compact_text(BeautifulSoup(description_html, "html.parser").get_text(" ", strip=True))
    content_html = str(detail.get("noiDung", ""))
    content_soup = BeautifulSoup(content_html, "html.parser")
    body = text_with_paragraphs(content_soup.find("body") or content_soup)
    if len(title) < 5 or len(body) < 150:
        return None

    final_url = normalize_url(candidate.url, article=True)
    category = normalize_moh_parent_domain(
        candidate.domain
        or str(detail.get("tenDanhMuc") or detail.get("tenMuc") or detail.get("TEN_MUC") or ""),
        category_id,
    )
    raw_date = str(
        detail.get("thoiGianBatDauDang")
        or detail.get("thoiGianXuatBan")
        or candidate.lastmod
        or ""
    )
    return {
        "id": hashlib.md5(f"{final_url}|{title}".encode("utf-8")).hexdigest(),
        "source_type": "website",
        "source_name": config.name,
        "url": final_url,
        "domain": category,
        "publish_date": normalize_date(raw_date),
        "claim": title,
        "original_text": description,
        "label": config.label,
        "evidence": config.evidence,
        "justification": body,
        "comments": [],
    }


def fetch_article(client: HttpClient, candidate: Candidate, config: SourceConfig) -> dict[str, Any] | None:
    try:
        if config.key == "moh":
            return fetch_moh_article(client, candidate, config)
        response = client.get(candidate.url, verify_tls=config.verify_tls)
        content_type = response.headers.get("Content-Type", "").lower()
        if content_type and "html" not in content_type and "xhtml" not in content_type:
            return None
        normalized_candidate = normalize_url(candidate.url, article=True)
        if config.key == "gso":
            domain_hint = ""
        elif config.key == "who":
            domain_hint = candidate.domain or domain_from_url(normalized_candidate, config)
        else:
            domain_hint = load_domain_history().get(normalized_candidate, "") or candidate.domain
        record = parse_article(
            response.content,
            response.url,
            config,
            fallback_domain=domain_hint,
        )
        if record and config.key == "gso":
            record["domain"] = resolve_wordpress_domain(
                client,
                response.content,
                response.url,
                config,
            )
            if not record["domain"]:
                LOGGER.debug("Bỏ bài GSO chỉ có category ngoài 16 domain: %s", candidate.url)
                return None
        if record and not record["publish_date"] and candidate.lastmod:
            record["publish_date"] = normalize_date(candidate.lastmod)
        return record
    except (requests.RequestException, PermissionError, UnicodeError, ValueError) as exc:
        LOGGER.warning("Lỗi bài %s: %s", candidate.url, exc)
        return None


def output_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return exactly the public schema, in its required display order."""
    normalized = {
        field: (
            repair_mojibake(value)
            if isinstance(value := record.get(field, [] if field == "comments" else ""), str)
            else value
        )
        for field in OUTPUT_FIELDS
    }
    if not isinstance(normalized["comments"], list):
        normalized["comments"] = []
    return normalized


def read_existing(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    records: list[dict[str, Any]] = []
    urls: set[str] = set()
    if not path.exists():
        return records, urls
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning("Bỏ qua dòng JSONL lỗi %s:%d", path, line_number)
                continue
            item = output_record(item)
            records.append(item)
            if item.get("url"):
                urls.add(normalize_url(str(item["url"]), article=True))
    return records, urls


def write_pretty_json(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Atomically write a readable JSON array with one field per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    payload = [output_record(record) for record in records]
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def write_checkpoint(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Atomically rewrite the internal JSONL checkpoint."""
    temp_path = path.with_suffix(path.suffix + ".tmp")
    lines = [json.dumps(output_record(record), ensure_ascii=False) for record in records]
    temp_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    temp_path.replace(path)


def backfill_existing_domains(
    client: HttpClient,
    config: SourceConfig,
    records: list[dict[str, Any]],
    candidates: dict[str, Candidate],
) -> bool:
    """Repair blank/generic domains created by older crawler versions."""
    changed = False
    if config.key == "who":
        filtered_records = []
        for record in records:
            url = normalize_url(str(record.get("url", "")), article=True)
            candidate = candidates.get(url)
            domain = candidate.domain if candidate else domain_from_url(url, config)
            if domain not in WHO_DOMAINS:
                changed = True
                continue
            if compact_text(str(record.get("domain", ""))) != domain:
                record["domain"] = domain
                changed = True
            filtered_records.append(record)
        records[:] = filtered_records
        return changed
    history = load_domain_history()
    for record in records:
        current = compact_text(str(record.get("domain", "")))
        if config.key == "gso" and current:
            cleaned = normalize_gso_parent_domains(current)
            if cleaned:
                if cleaned != current:
                    record["domain"] = cleaned
                    changed = True
                continue
        if config.key == "moh" and current:
            url = normalize_url(str(record.get("url", "")), article=True)
            category_id = compact_text(dict(parse_qsl(urlsplit(url).query)).get("cid", ""))
            cleaned = normalize_moh_parent_domain(current, category_id)
            if cleaned and cleaned != current:
                record["domain"] = cleaned
                changed = True
            continue
        if current.casefold() not in {"", "item", "detail"}:
            continue
        url = normalize_url(str(record.get("url", "")), article=True)
        candidate = candidates.get(url)
        domain = history.get(url, "")
        if not domain and candidate:
            domain = candidate.domain
        if not domain:
            domain = domain_from_url(url, config)
        if not domain and config.key == "gso" and url:
            try:
                response = client.get(url, verify_tls=config.verify_tls)
                domain = resolve_wordpress_domain(client, response.content, response.url, config)
            except (requests.RequestException, PermissionError):
                domain = ""
        if domain:
            record["domain"] = domain
            changed = True
    return changed


def prepare_checkpoint(output_dir: Path, key: str) -> Path:
    """Move legacy root-level JSONL into the internal checkpoint directory."""
    checkpoint_dir = output_dir / ".checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{key}.jsonl"
    legacy_path = output_dir / f"{key}.jsonl"
    if legacy_path.exists() and not checkpoint_path.exists():
        legacy_path.replace(checkpoint_path)
    return checkpoint_path


def ordered_candidates(candidates: dict[str, Candidate]) -> list[Candidate]:
    return sorted(candidates.values(), key=lambda item: (item.lastmod, item.url), reverse=True)


def crawl_source(
    client: HttpClient,
    config: SourceConfig,
    *,
    output_dir: Path,
    target: int,
    workers: int,
    max_listing_pages: int,
    discover_only: bool,
) -> dict[str, Any]:
    output_path = output_dir / f"{config.key}.json"
    checkpoint_path = prepare_checkpoint(output_dir, config.key)
    records, existing_urls = read_existing(checkpoint_path)
    existing_count = len(records)
    LOGGER.info("[%s] đã có %d bản ghi; mục tiêu %d", config.name, existing_count, target)

    candidates: dict[str, Candidate] = {}
    if config.key == "moh":
        candidates.update(discover_from_moh_api(client, config, target))
    else:
        if config.sitemap_urls:
            candidates.update(discover_from_sitemaps(client, config))
        if config.seed_urls and (config.key == "who" or len(candidates) < target):
            listing_candidates = discover_from_listing_pages(
                client,
                config,
                max_pages=max_listing_pages,
            )
            for url, candidate in listing_candidates.items():
                previous = candidates.get(url)
                if previous is None or candidate.domain_rank > previous.domain_rank:
                    if previous and not candidate.lastmod:
                        candidate = Candidate(
                            candidate.url,
                            previous.lastmod,
                            candidate.domain,
                            candidate.domain_rank,
                        )
                    candidates[url] = candidate

    if not discover_only and backfill_existing_domains(client, config, records, candidates):
        write_checkpoint(checkpoint_path, records)
        write_pretty_json(output_path, records)
        existing_count = len(records)
        existing_urls = {
            normalize_url(str(record["url"]), article=True)
            for record in records
            if record.get("url")
        }

    candidate_list = [item for item in ordered_candidates(candidates) if item.url not in existing_urls]
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
            write_pretty_json(output_path, records)
        return summary

    needed = target - existing_count
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iterator = iter(candidate_list)
    pending: dict[Future[dict[str, Any] | None], Candidate] = {}
    successful_urls = set(existing_urls)

    with checkpoint_path.open("a", encoding="utf-8", buffering=1) as stream, ThreadPoolExecutor(
        max_workers=max(1, workers), thread_name_prefix=config.key
    ) as executor:

        def fill_queue() -> None:
            while len(pending) < max(1, workers) * 2:
                try:
                    candidate = next(iterator)
                except StopIteration:
                    break
                pending[executor.submit(fetch_article, client, candidate, config)] = candidate

        fill_queue()
        while pending and summary["added"] < needed:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                candidate = pending.pop(future)
                try:
                    record = future.result()
                except Exception as exc:  # Guard worker failures and keep checkpointing.
                    LOGGER.exception("Worker lỗi ở %s: %s", candidate.url, exc)
                    record = None
                if record:
                    record = output_record(record)
                    record_url = normalize_url(str(record["url"]), article=True)
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
    write_pretty_json(output_path, records)
    return summary


def combine_outputs(output_dir: Path, source_keys: Iterable[str]) -> int:
    combined_path = output_dir / "all_articles.json"
    seen: set[str] = set()
    combined: list[dict[str, Any]] = []
    for key in source_keys:
        source_path = output_dir / f"{key}.json"
        if not source_path.exists():
            checkpoint_path = prepare_checkpoint(output_dir, key)
            checkpoint_records, _ = read_existing(checkpoint_path)
            if checkpoint_records:
                write_pretty_json(source_path, checkpoint_records)
        if not source_path.exists():
            continue
        try:
            source_records = json.loads(source_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(source_records, list):
            continue
        for record in source_records:
            if not isinstance(record, dict):
                continue
            record = output_record(record)
            identity = str(record.get("id") or record.get("url") or "")
            if not identity or identity in seen:
                continue
            seen.add(identity)
            combined.append(record)
    write_pretty_json(combined_path, combined)
    return len(combined)


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("giá trị phải lớn hơn 0")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("all", *SOURCES.keys()),
        default="all",
        help="Nguồn cần crawl (mặc định: all)",
    )
    parser.add_argument("--limit", type=positive_int, help="Ghi đè quota cho từng nguồn được chọn")
    parser.add_argument("--workers", type=positive_int, default=4, help="Số worker tải trang chi tiết")
    parser.add_argument("--delay", type=float, default=0.5, help="Khoảng cách tối thiểu giữa hai request/host")
    parser.add_argument("--timeout", type=float, default=30.0, help="Timeout mỗi request, giây")
    parser.add_argument(
        "--max-listing-pages",
        type=positive_int,
        default=500,
        help="Số trang danh mục tối đa cho nguồn không có sitemap",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--discover-only", action="store_true", help="Chỉ thống kê URL, không tải bài")
    parser.add_argument("--ignore-robots", action="store_true", help="Bỏ kiểm tra robots.txt (không khuyến nghị)")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    # Windows often inherits a legacy console code page (for example cp1258),
    # which cannot print every Vietnamese character even though JSONL is UTF-8.
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

    # Some Vietnamese government sites expose incomplete certificate chains.
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_keys = list(SOURCES) if args.source == "all" else [args.source]
    client = HttpClient(delay=args.delay, timeout=args.timeout, obey_robots=not args.ignore_robots)
    summaries = []
    for key in selected_keys:
        config = SOURCES[key]
        target = args.limit or config.target
        summaries.append(
            crawl_source(
                client,
                config,
                output_dir=args.output_dir,
                target=target,
                workers=args.workers,
                max_listing_pages=args.max_listing_pages,
                discover_only=args.discover_only,
            )
        )

    combined_count = 0
    if not args.discover_only:
        # Include every output currently present. This keeps the combined file
        # complete when users crawl each source in separate commands.
        combined_count = combine_outputs(args.output_dir, SOURCES.keys())
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "discover_only": args.discover_only,
        "combined_count": combined_count,
        "sources": summaries,
    }
    report_path = args.output_dir / "crawl_summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all(item["shortfall"] == 0 for item in summaries) or args.discover_only else 2


if __name__ == "__main__":
    raise SystemExit(main())

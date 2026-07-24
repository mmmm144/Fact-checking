import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from crawl_sources import (
    Candidate,
    GSO_DOMAIN_BY_SLUG,
    MOH_API_CATEGORIES,
    MOH_PARENT_DOMAINS,
    OUTPUT_FIELDS,
    SOURCES,
    WHO_DOMAINS,
    WHO_DOMAIN_PAGES,
    backfill_existing_domains,
    configured_domain,
    discover_from_moh_api,
    discover_who_hub_items,
    domain_from_url,
    fetch_moh_article,
    load_domain_history,
    looks_like_listing_url,
    normalize_gso_parent_domains,
    normalize_moh_domain,
    normalize_moh_parent_domain,
    normalize_date,
    normalize_url,
    parse_article,
    parse_sitemap,
    repair_mojibake,
    resolve_wordpress_domain,
    who_hub_item_url,
    write_pretty_json,
)


class CrawlerTests(unittest.TestCase):
    def test_gso_taxonomy_has_exactly_16_feature_domains(self):
        self.assertEqual(len(GSO_DOMAIN_BY_SLUG), 16)
        self.assertEqual(
            set(GSO_DOMAIN_BY_SLUG.values()),
            {
                "Dân số và lao động",
                "Tài khoản quốc gia và tài chính",
                "Kinh tế",
                "Xã hội môi trường và đơn vị hành chính",
                "Tổng điều tra",
            },
        )
        self.assertNotIn("chu-de-khac", GSO_DOMAIN_BY_SLUG)
        self.assertIn("tong-dieu-tra-kinh-te", GSO_DOMAIN_BY_SLUG)
        self.assertIn("tong-dieu-tra-dan-so-va-nha-o", GSO_DOMAIN_BY_SLUG)
        self.assertIn("tong-dieu-tra-nong-thon-nong-nghiep-va-thuy-san", GSO_DOMAIN_BY_SLUG)

    def test_parse_sitemap_with_namespace(self):
        xml = b"""<?xml version='1.0'?>
        <urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>
          <url><loc>https://tingia.gov.vn/a.html</loc><lastmod>2026-01-02</lastmod></url>
        </urlset>"""
        kind, items = parse_sitemap(xml)
        self.assertEqual(kind, "urlset")
        self.assertEqual(items[0].url, "https://tingia.gov.vn/a.html")
        self.assertEqual(items[0].lastmod, "2026-01-02")

    def test_parse_sitemap_link_inside_urlset(self):
        xml = b"""<urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>
          <url><loc>https://tingia.gov.vn/sitemap/tin-vua-check.xml</loc></url>
        </urlset>"""
        kind, items = parse_sitemap(xml)
        self.assertEqual(kind, "urlset")
        self.assertTrue(items[0].url.endswith(".xml"))

    def test_normalize_url_removes_tracking(self):
        value = normalize_url(
            "/news/item/example?utm_source=x&id=2#part",
            "https://www.who.int/",
            article=True,
        )
        self.assertEqual(value, "https://www.who.int/news/item/example?id=2")

    def test_normalize_dates(self):
        self.assertEqual(normalize_date("2026-07-21T10:30:00Z"), "2026-07-21")
        self.assertEqual(normalize_date("21/07/2026 - 10:30"), "2026-07-21")
        self.assertEqual(normalize_date("21 July 2026"), "2026-07-21")

    def test_repair_mojibake(self):
        broken = "Lan truy\u00e1\u00bb\u0081n tin gi\u00e1\u00ba\u00a3"
        self.assertEqual(repair_mojibake(broken), "Lan truy\u1ec1n tin gi\u1ea3")
        self.assertEqual(repair_mojibake("Official update"), "Official update")

    def test_parse_article_from_jsonld(self):
        html = b"""<html><head>
        <script type='application/ld+json'>
        {"@type":"NewsArticle","headline":"Official health update",
         "description":"A short summary","datePublished":"2026-07-21",
         "articleSection":"Health","articleBody":"This is a sufficiently long official article body. This sentence is repeated so that the parser treats it as a real article. This is a sufficiently long official article body. This sentence is repeated so that the parser treats it as a real article."}
        </script></head><body><h1>Fallback title</h1></body></html>"""
        item = parse_article(html, "https://www.who.int/news/item/example", SOURCES["who"])
        self.assertIsNotNone(item)
        assert item is not None
        self.assertEqual(item["claim"], "Official health update")
        self.assertEqual(item["publish_date"], "2026-07-21")
        self.assertEqual(item["domain"], "Health")
        self.assertEqual(tuple(item), OUTPUT_FIELDS)
        self.assertEqual(item["comments"], [])

    def test_vafc_taxonomy_matches_every_configured_category(self):
        expected = [name for name, _ in SOURCES["vafc"].category_pages]
        actual = []
        for _, page_url in SOURCES["vafc"].category_pages:
            slug = page_url.rstrip("/").rsplit("/", 1)[-1]
            name, _ = configured_domain(
                f"https://tingia.gov.vn/sitemap/{slug}.xml",
                SOURCES["vafc"],
            )
            actual.append(name)
        self.assertEqual(actual, expected)
        self.assertGreater(len(set(actual)), 3)

    def test_curated_domains_preserve_project_taxonomy(self):
        history = load_domain_history()
        expected = {
            "canh-bao-ve-cac-hinh-thuc-lua-dao-va-gian-lan-thuong-mai-tai-uae.html": "Quy\u1ec1n l\u1ee3i ng\u01b0\u1eddi d\u00e2n",
            "canh-bao-mao-danh-ngan-hang-nha-nuoc-gui-duong-link-cap-nhat-thong-tin-sinh-trac-hoc-de-lua-dao.html": "Vaccine tin gi\u1ea3",
            "chieu-danh-cap-ma-otp-bang-cuoc-goi-tu-dong.html": "T\u00e0i ch\u00ednh - Ng\u00e2n h\u00e0ng",
        }
        for slug, domain in expected.items():
            self.assertEqual(history[f"https://tingia.gov.vn/{slug}"], domain)

    def test_who_has_exactly_eight_requested_parent_domains(self):
        self.assertEqual(tuple(name for name, _ in WHO_DOMAIN_PAGES), WHO_DOMAINS)
        self.assertEqual(
            WHO_DOMAINS,
            (
                "News",
                "Emergencies",
                "Campaigns",
                "Events",
                "Statements",
                "Feature stories",
                "Speeches",
                "Commentaries",
            ),
        )
        self.assertEqual(SOURCES["who"].category_pages, WHO_DOMAIN_PAGES)

    def test_who_domain_comes_from_requested_routes(self):
        cases = {
            "https://www.who.int/news/item/example": "News",
            "https://www.who.int/emergencies/situations/example": "Emergencies",
            "https://www.who.int/campaigns/world-health-day": "Campaigns",
            "https://www.who.int/news-room/events/detail/example": "Events",
            "https://www.who.int/news-room/statements/detail/example": "Statements",
            "https://www.who.int/news-room/feature-stories/detail/example": "Feature stories",
            "https://www.who.int/news-room/commentaries/detail/example": "Commentaries",
            "https://www.who.int/director-general/speeches/detail/example": "Speeches",
        }
        for url, expected in cases.items():
            self.assertEqual(domain_from_url(url, SOURCES["who"]), expected)
        self.assertEqual(
            domain_from_url(
                "https://www.who.int/news-room/fact-sheets/detail/example",
                SOURCES["who"],
            ),
            "",
        )

    def test_who_hub_slugs_expand_to_public_detail_routes(self):
        cases = {
            "https://www.who.int/api/hubs/newsitems?x=1": "https://www.who.int/news/item/example",
            "https://www.who.int/api/hubs/featurestories?x=1": (
                "https://www.who.int/news-room/feature-stories/detail/example"
            ),
            "https://www.who.int/api/hubs/speeches?x=1": (
                "https://www.who.int/news-room/speeches/item/example"
            ),
        }
        for api_url, expected in cases.items():
            self.assertEqual(who_hub_item_url(api_url, "/example"), expected)

    def test_who_numeric_listing_pages_keep_parent_domain(self):
        self.assertTrue(
            looks_like_listing_url(
                "https://www.who.int/news-room/statements/4",
                {"/news-room/statements"},
                SOURCES["who"],
            )
        )
        self.assertFalse(
            looks_like_listing_url(
                "https://www.who.int/news?unrelated=1",
                {"/news"},
                SOURCES["who"],
            )
        )

    def test_who_hub_api_uses_supported_page_size(self):
        class FakeResponse:
            def json(self):
                return {
                    "value": [
                        {
                            "ItemDefaultUrl": "/example",
                            "FormatedDate": "21 July 2026",
                        }
                    ]
                }

        class FakeClient:
            def __init__(self):
                self.urls = []

            def get(self, url, **kwargs):
                self.urls.append(url)
                return FakeResponse()

        client = FakeClient()
        html = '''<script>load("/api/hubs/newsitems?sf_culture=en")</script>'''
        found = discover_who_hub_items(
            client,
            html,
            "https://www.who.int/news",
            "News",
            0,
            SOURCES["who"],
        )
        self.assertIn("https://www.who.int/news/item/example", found)
        self.assertIn("$top=100&$skip=0", client.urls[0])

    def test_who_backfill_prefers_discovery_domain_and_removes_old_sections(self):
        statement_url = "https://www.who.int/news/item/example"
        records = [
            {"url": statement_url, "domain": "Health"},
            {
                "url": "https://www.who.int/news-room/fact-sheets/detail/old",
                "domain": "Fact sheets",
            },
        ]
        candidates = {
            statement_url: Candidate(statement_url, domain="Statements", domain_rank=4)
        }
        self.assertTrue(
            backfill_existing_domains(object(), SOURCES["who"], records, candidates)
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["domain"], "Statements")

    def test_wordpress_domain_keeps_all_matching_feature_domains(self):
        html = (
            b"<article class='post type-post category-chu-de-khac "
            b"category-tai-khoan-quoc-gia category-tong-dieu-tra-kinh-te'></article>"
        )
        domain = resolve_wordpress_domain(
            object(),
            html,
            "https://www.nso.gov.vn/tin-tuc-khac/example/",
            SOURCES["gso"],
        )
        self.assertEqual(
            domain,
            "T\u00e0i kho\u1ea3n qu\u1ed1c gia v\u00e0 t\u00e0i ch\u00ednh; T\u1ed5ng \u0111i\u1ec1u tra",
        )

    def test_gso_feature_names_roll_up_and_deduplicate_parent_domains(self):
        self.assertEqual(
            normalize_gso_parent_domains(
                "C\u00f4ng nghi\u1ec7p; Doanh nghi\u1ec7p; Th\u1ed1ng k\u00ea gi\u00e1; D\u00e2n s\u1ed1; Lao \u0111\u1ed9ng"
            ),
            "Kinh t\u1ebf; D\u00e2n s\u1ed1 v\u00e0 lao \u0111\u1ed9ng",
        )

    def test_wordpress_domain_rejects_chu_de_khac(self):
        html = b"<article class='post type-post category-chu-de-khac'></article>"
        self.assertEqual(
            resolve_wordpress_domain(
                object(),
                html,
                "https://www.nso.gov.vn/tin-tuc-khac/example/",
                SOURCES["gso"],
            ),
            "",
        )

    def test_moh_api_discovers_articles_with_real_categories(self):
        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload

            def json(self):
                return self.payload

        class FakeClient:
            @staticmethod
            def post_json(url, payload, **kwargs):
                category_id = payload["MA_DANH_MUC"]
                return FakeResponse(
                    [{
                        "ITEMS": [{
                            "MA": f"article-{category_id}",
                            "TEN_MUC": "Chuy\u1ec3n \u0111\u1ed5i s\u1ed1 y t\u1ebf",
                            "THOI_GIAN_XUAT_BAN": "21/07/2026 03:00 PM",
                        }]
                    }]
                )

        candidates = discover_from_moh_api(FakeClient(), SOURCES["moh"], 5)
        self.assertEqual(len(candidates), len(MOH_API_CATEGORIES))
        candidate = next(item for item in candidates.values() if "cid=8356" in item.url)
        self.assertEqual(candidate.domain, "Tin t\u1ee9c - s\u1ef1 ki\u1ec7n")
        self.assertEqual(candidate.lastmod, "2026-07-21")

    def test_moh_taxonomy_has_four_parent_domains(self):
        self.assertEqual(
            set(MOH_PARENT_DOMAINS),
            {
                "Tin t\u1ee9c - s\u1ef1 ki\u1ec7n",
                "Cung c\u1ea5p th\u00f4ng tin",
                "Tra c\u1ee9u",
                "Chuy\u1ec3n \u0111\u1ed5i s\u1ed1 y t\u1ebf",
            },
        )
        self.assertEqual(len(MOH_API_CATEGORIES), 23)

    def test_moh_duplicate_article_prefers_more_specific_parent_domain(self):
        class FakeResponse:
            @staticmethod
            def json():
                return [{"ITEMS": [{"MA": "same-article"}]}]

        class FakeClient:
            @staticmethod
            def post_json(*args, **kwargs):
                return FakeResponse()

        candidates = discover_from_moh_api(FakeClient(), SOURCES["moh"], 5)
        self.assertEqual(len(candidates), 1)
        candidate = next(iter(candidates.values()))
        self.assertEqual(candidate.domain, "Chuy\u1ec3n \u0111\u1ed5i s\u1ed1 y t\u1ebf")
        self.assertIn("cid=8430", candidate.url)

    def test_moh_api_fetches_exact_output_schema(self):
        class FakeResponse:
            @staticmethod
            def json():
                paragraph = "N\u1ed9i dung ch\u00ednh th\u1ee9c c\u1ee7a B\u1ed9 Y t\u1ebf \u0111\u01b0\u1ee3c c\u00f4ng b\u1ed1 \u0111\u1ea7y \u0111\u1ee7. " * 4
                return {
                    "d": {
                        "tieuDe": "B\u1ed9 Y t\u1ebf c\u00f4ng b\u1ed1 th\u00f4ng tin m\u1edbi",
                        "moTa": "<strong>T\u00f3m t\u1eaft b\u00e0i vi\u1ebft</strong>",
                        "noiDung": f"<p>{paragraph}</p>",
                        "thoiGianBatDauDang": "21/07/2026 03:00 PM",
                    }
                }

        class FakeClient:
            @staticmethod
            def post_json(*args, **kwargs):
                return FakeResponse()

        candidate = Candidate(
            "https://moh.gov.vn/index.jsp?pageId=5803&aid=168692&cid=8356",
            domain="Tin t\u1ed5ng h\u1ee3p",
        )
        item = fetch_moh_article(FakeClient(), candidate, SOURCES["moh"])
        self.assertIsNotNone(item)
        assert item is not None
        self.assertEqual(tuple(item), OUTPUT_FIELDS)
        self.assertEqual(item["domain"], "Tin t\u1ee9c - s\u1ef1 ki\u1ec7n")
        self.assertEqual(item["publish_date"], "2026-07-21")
        self.assertEqual(item["original_text"], "T\u00f3m t\u1eaft b\u00e0i vi\u1ebft")
        self.assertEqual(item["comments"], [])

    def test_moh_domain_keeps_one_clean_primary_category(self):
        self.assertEqual(
            normalize_moh_domain(
                "Tin ho\u1ea1t \u0111\u1ed9ng (Tin l\u00e3nh \u0111\u1ea1o B\u1ed9);Tin n\u1ed5i b\u1eadt (Tin t\u1ee9c - s\u1ef1 ki\u1ec7n)"
            ),
            "Tin ho\u1ea1t \u0111\u1ed9ng",
        )
        self.assertEqual(
            normalize_moh_parent_domain("Tin ho\u1ea1t \u0111\u1ed9ng", "8351"),
            "Tin t\u1ee9c - s\u1ef1 ki\u1ec7n",
        )
        self.assertEqual(
            normalize_moh_parent_domain("Th\u00f4ng tin \u0111\u1ea7u t\u01b0", "8422"),
            "Tra c\u1ee9u",
        )

    def test_pretty_json_has_exact_schema_and_one_field_per_line(self):
        record = {field: [] if field == "comments" else field for field in OUTPUT_FIELDS}
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "output.json"
            write_pretty_json(path, [record])
            lines = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(lines[0], "[")
        self.assertEqual(lines[1], "    {")
        self.assertEqual(lines[-1], "]")
        field_lines = [line for line in lines if line.strip().startswith(chr(34))]
        self.assertEqual(len(field_lines), len(OUTPUT_FIELDS))


if __name__ == "__main__":
    unittest.main()

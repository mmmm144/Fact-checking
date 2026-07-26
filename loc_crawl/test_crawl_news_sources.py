import unittest

from crawl_sources import Candidate, OUTPUT_FIELDS
from crawl_news_sources import (
    BAO_CHINH_PHU_CATEGORIES,
    NEWS_SOURCES,
    VNEXPRESS_CATEGORIES,
    WORLD_BANK_DOMAINS,
    discover_bao_chinh_phu,
    discover_vnexpress,
    extract_bao_chinh_phu_zone_id,
    extract_candidates,
    fetch_news_article,
    merge_candidate,
    vnexpress_page_url,
    world_bank_domain,
)


class FakeResponse:
    def __init__(self, content, url, content_type="text/html; charset=utf-8"):
        self.content = content.encode("utf-8") if isinstance(content, str) else content
        self.url = url
        self.headers = {"Content-Type": content_type}


class NewsCrawlerTests(unittest.TestCase):
    def test_default_targets(self):
        self.assertEqual(NEWS_SOURCES["world_bank"].target, 3400)
        self.assertEqual(NEWS_SOURCES["bao_chinh_phu"].target, 2500)
        self.assertEqual(NEWS_SOURCES["vnexpress"].target, 2600)

    def test_requested_category_lists_are_complete(self):
        self.assertEqual(
            tuple(url for _, url in BAO_CHINH_PHU_CATEGORIES),
            (
                "https://baochinhphu.vn/chinh-tri.htm",
                "https://baochinhphu.vn/kinh-te.htm",
                "https://baochinhphu.vn/van-hoa.htm",
                "https://baochinhphu.vn/xa-hoi.htm",
                "https://baochinhphu.vn/khoa-giao.htm",
                "https://baochinhphu.vn/quoc-te.htm",
            ),
        )
        self.assertEqual(len(VNEXPRESS_CATEGORIES), 17)
        self.assertIn(("Spotlight", "https://vnexpress.net/spotlight"), VNEXPRESS_CATEGORIES)
        self.assertIn(("Thư giãn", "https://vnexpress.net/thu-gian"), VNEXPRESS_CATEGORIES)

    def test_world_bank_uses_only_english_news_sitemaps(self):
        config = NEWS_SOURCES["world_bank"]
        self.assertEqual(config.sitemap_urls, ("https://www.worldbank.org/sitemap.xml",))
        self.assertTrue(config.sitemap_patterns[0].endswith(r"\.xml$"))

    def test_world_bank_domain_from_route(self):
        for slug, expected in WORLD_BANK_DOMAINS.items():
            url = f"https://www.worldbank.org/en/news/{slug}/2026/07/01/example"
            self.assertEqual(world_bank_domain(url), expected)
        self.assertEqual(
            world_bank_domain("https://www.worldbank.org/en/news/video/2026/07/01/example"),
            "",
        )

    def test_world_bank_opinion_article_date_parsing(self):
        from crawl_sources import parse_article
        config = NEWS_SOURCES["world_bank"]
        html = b"""<html><head>
        <script type='application/ld+json'>
        {"@context": "https://schema.org",
         "@type": "OpinionNewsArticle",
         "headline": "Panama's challenge: growing with the best human talent",
         "datePublished": "2026-03-04T14:35:14.396"}
        </script></head><body>
        <main><article class="lp-body-content">
        This is a sufficiently long body content to satisfy the length check of 150 characters.
        This is a sufficiently long body content to satisfy the length check of 150 characters.
        </article></main>
        </body></html>"""
        item = parse_article(html, "https://www.worldbank.org/en/news/opinion/2026/03/04/el-desafio-panama-crecer-con-mejor-talento-humano", config)
        self.assertIsNotNone(item)
        self.assertEqual(item["publish_date"], "2026-03-04")

    def test_world_bank_date_fallback_from_url(self):
        from crawl_sources import parse_article
        config = NEWS_SOURCES["world_bank"]
        # HTML without any JSON-LD or meta tags or time tags
        html = b"""<html><body>
        <h1>A sufficiently long article title</h1>
        <main><article class="lp-body-content">
        This is a sufficiently long body content to satisfy the length check of 150 characters.
        This is a sufficiently long body content to satisfy the length check of 150 characters.
        </article></main>
        </body></html>"""
        item = parse_article(html, "https://www.worldbank.org/en/news/opinion/2026/03/04/el-desafio-panama-crecer-con-mejor-talento-humano", config)
        self.assertIsNotNone(item)
        self.assertEqual(item["publish_date"], "2026-03-04")

    def test_zone_id_parser(self):
        html = '<input name="hdZoneId" id="hdZoneId" value="102442">'
        self.assertEqual(extract_bao_chinh_phu_zone_id(html), "102442")
        self.assertEqual(extract_bao_chinh_phu_zone_id("<html></html>"), "")
        self.assertEqual(
            extract_bao_chinh_phu_zone_id('<input id="hdZoneId" value="not-a-number">'),
            "",
        )

    def test_vnexpress_page_urls(self):
        seed = "https://vnexpress.net/thoi-su"
        self.assertEqual(vnexpress_page_url(seed, 1), seed)
        self.assertEqual(vnexpress_page_url(seed, 2), f"{seed}-p2")

    def test_extract_candidates_scopes_to_listing_selector(self):
        config = NEWS_SOURCES["vnexpress"]
        html = """
        <article class="item-news">
          <h3 class="title-news">
            <a href="/bai-hop-le-1234567.html?utm_source=test">Bài hợp lệ</a>
          </h3>
        </article>
        <aside><a href="/bai-ngoai-khoi-7654321.html">Không lấy</a></aside>
        """
        found = extract_candidates(
            html,
            "https://vnexpress.net/thoi-su",
            config,
            domain="Thời sự",
            domain_rank=0,
            selectors=("article.item-news h3.title-news a[href]",),
        )
        self.assertEqual(list(found), ["https://vnexpress.net/bai-hop-le-1234567.html"])
        self.assertEqual(next(iter(found.values())).domain, "Thời sự")

    def test_merge_candidate_prefers_later_category_rank(self):
        url = "https://vnexpress.net/example-1234567.html"
        found = {url: Candidate(url, domain="Thời sự", domain_rank=0)}
        self.assertFalse(
            merge_candidate(found, Candidate(url, domain="Kinh doanh", domain_rank=-1))
        )
        self.assertFalse(
            merge_candidate(found, Candidate(url, domain="Khoa học", domain_rank=3))
        )
        self.assertEqual(found[url].domain, "Khoa học")

    def test_bao_chinh_phu_discovery_uses_timeline_endpoint(self):
        config = NEWS_SOURCES["bao_chinh_phu"]

        class FakeClient:
            def __init__(self):
                self.urls = []

            def get(self, url, **kwargs):
                self.urls.append(url)
                if url == config.category_pages[0][1]:
                    return FakeResponse(
                        '<input id="hdZoneId" value="42">',
                        url,
                    )
                if url in {item[1] for item in config.category_pages[1:]}:
                    return FakeResponse("<html></html>", url)
                if url.endswith("/42/1.htm"):
                    return FakeResponse(
                        """
                        <div class="box-stream-item">
                          <a class="box-stream-link-with-avatar"
                             href="/tin-thu-nghiem-102260701123456789.htm">Tin</a>
                        </div>
                        """,
                        url,
                    )
                return FakeResponse("<html></html>", url)

        client = FakeClient()
        found = discover_bao_chinh_phu(client, config, target=1, max_pages=2)
        self.assertIn(
            "https://baochinhphu.vn/tin-thu-nghiem-102260701123456789.htm",
            found,
        )
        self.assertTrue(any("/timelinelist/42/1.htm" in url for url in client.urls))

    def test_vnexpress_discovery_stops_on_repeated_page(self):
        config = NEWS_SOURCES["vnexpress"]

        class FakeClient:
            def __init__(self):
                self.urls = []

            def get(self, url, **kwargs):
                self.urls.append(url)
                article_id = 1000000 + len(self.urls)
                if url.startswith(config.category_pages[0][1]):
                    html = f"""
                    <article class="item-news"><h3 class="title-news">
                      <a href="/tin-mau-{article_id}.html">Tin</a>
                    </h3></article>
                    """
                else:
                    html = "<html></html>"
                return FakeResponse(html, url)

        client = FakeClient()
        found = discover_vnexpress(client, config, target=2, max_pages=2)
        self.assertGreaterEqual(len(found), 2)
        self.assertIn("https://vnexpress.net/thoi-su-p2", client.urls)

    def test_fetch_article_keeps_source_domain_and_public_schema(self):
        config = NEWS_SOURCES["vnexpress"]
        body = " ".join(["Nội dung bài viết chính thức đủ dài để kiểm thử parser."] * 8)
        html = f"""
        <html><head>
          <script type="application/ld+json">
          {{
            "@type": "NewsArticle",
            "headline": "Tiêu đề bài kiểm thử",
            "description": "Mô tả bài kiểm thử",
            "datePublished": "2026-07-24",
            "articleBody": "{body}"
          }}
          </script>
        </head></html>
        """

        class FakeClient:
            def get(self, url, **kwargs):
                return FakeResponse(html, url)

        url = "https://vnexpress.net/tieu-de-bai-1234567.html"
        record = fetch_news_article(
            FakeClient(),
            Candidate(url, domain="Giáo dục", domain_rank=11),
            config,
        )
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["domain"], "Giáo dục")
        self.assertEqual(record["publish_date"], "2026-07-24")
        self.assertEqual(tuple(record), OUTPUT_FIELDS)


if __name__ == "__main__":
    unittest.main()

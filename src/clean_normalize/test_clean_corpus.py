import unittest
from pathlib import Path
from clean_corpus import (
    clean_html,
    normalize_unicode_nfc,
    normalize_whitespace,
    validate_repair_encoding,
    detect_language,
    standardize_date,
    calculate_statistics,
    clean_single_record,
)

class CleanCorpusTests(unittest.TestCase):
    def test_clean_html_removes_tags_and_keeps_content(self):
        html = "<html><head><style>.ad{color:red}</style></head><body><h1>Main Title</h1><p>This is a paragraph. <script>alert(1)</script>And some text.</p><div class='footer'>Copyright 2026</div></body></html>"
        cleaned = clean_html(html)
        self.assertNotIn("color:red", cleaned)
        self.assertNotIn("alert(1)", cleaned)
        self.assertNotIn("Copyright 2026", cleaned)
        self.assertIn("Main Title", cleaned)
        self.assertIn("This is a paragraph", cleaned)

    def test_normalize_unicode_nfc(self):
        # Vietnamese combining accents normalization check
        # 'hòa' in decomposited (NFD) representation vs composited (NFC) representation
        nfd_str = "ho\u0300a" # h - o - ̀ - a
        nfc_str = "hòa"
        self.assertEqual(normalize_unicode_nfc(nfd_str), nfc_str)

    def test_normalize_whitespace(self):
        text = "  Too    many   spaces.\n\n\n\nNew paragraph.  "
        self.assertEqual(normalize_whitespace(text), "Too many spaces.\n\nNew paragraph.")

    def test_validate_repair_encoding(self):
        # Check stripping of system control characters (like \x01, \x12)
        text = "Text\x01 with\x12 control chars."
        self.assertEqual(validate_repair_encoding(text), "Text with control chars.")

    def test_detect_language(self):
        vi_text = "Dữ liệu thống kê về tình hình kinh tế xã hội của Việt Nam trong năm qua đã có nhiều chuyển biến tích cực."
        en_text = "The World Health Organization is monitoring the health inequality data across different countries and regions."
        self.assertEqual(detect_language(vi_text, "vi"), "vi")
        self.assertEqual(detect_language(en_text, "en"), "en")

    def test_standardize_date(self):
        # Valid date formats
        self.assertEqual(standardize_date("2026-07-27", "http://example.com"), "2026-07-27")
        self.assertEqual(standardize_date("27/07/2026", "http://example.com"), "2026-07-27")
        # Extract from URL path fallback
        self.assertEqual(standardize_date("", "http://example.com/en/news/opinion/2026/03/04/example-url"), "2026-03-04")
        self.assertEqual(standardize_date("invalid-date", "http://example.com/en/news/press-release/2025/12/08/example"), "2025-12-08")

    def test_calculate_statistics(self):
        text = "Một hai ba bốn năm. Sáu bảy tám chín mười!"
        stats = calculate_statistics(text, "vi")
        self.assertEqual(stats["word_count"], 10)
        self.assertEqual(stats["sentence_count"], 2)
        self.assertEqual(stats["reading_time"], 1) # ceil(10/150) = 1

    def test_clean_single_record_filters_short_articles(self):
        config = {
            "title": "TestPublisher",
            "source_type": "News",
            "domain": "General",
            "document_type": "News",
            "country": "Vietnam",
            "default_lang": "vi"
        }
        
        # Too short justification body (only 5 words)
        raw_short = {
            "claim": "Valid Title Here",
            "justification": "This body is too short.",
            "url": "http://example.com"
        }
        self.assertIsNone(clean_single_record(raw_short, config, "2026-07-27"))

        # Valid article
        raw_valid = {
            "claim": "Valid Title Here",
            "justification": "This is a sufficiently long body content to satisfy the length check of 150 characters. This is a sufficiently long body content to satisfy the length check of 150 characters.",
            "url": "http://example.com/2026/07/27/test-url"
        }
        cleaned = clean_single_record(raw_valid, config, "2026-07-27")
        self.assertIsNotNone(cleaned)
        self.assertEqual(cleaned["title"], "Valid Title Here")
        self.assertEqual(cleaned["source"], "TestPublisher")
        self.assertEqual(cleaned["publish_date"], "2026-07-27")
        self.assertTrue(cleaned["quality"]["cleaned"])

if __name__ == "__main__":
    unittest.main()

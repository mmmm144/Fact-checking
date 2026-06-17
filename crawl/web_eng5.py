import requests
from bs4 import BeautifulSoup
import json
import time
import re
import urllib3
import xml.etree.ElementTree as ET
from datetime import datetime

urllib3.disable_warnings()

OUTPUT_FILE = "news_website_output_en.json"

# Danh sách User-Agents để tránh bị chặn cơ bản
user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
]

import random

def get_random_user_agent():
    return random.choice(user_agents)

headers = {
    "User-Agent": get_random_user_agent(),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive"
}

# ==================== CẤU HÌNH SỐ LƯỢNG BÀI CRAWL MỖI TRANG ====================
# Chỉnh số ở đây để tăng/giảm số bài crawl cho từng nguồn.
# LƯU Ý: Guardian tính theo MỖI chuyên mục (có 3 mục) -> tổng = limit x 3.
# Các nguồn còn lại tính theo TỔNG số bài.
LIMITS = {
    "guardian": 20,          # mỗi chuyên mục
    "cnn_lite": 20,
    "apnews": 20,
    "npr": 20,
    "legiblenews": 20,
    "hackernews": 20,
    "factcheck_viral": 20,
    "factcheck_scicheck": 20,
    "snopes": 20,
    "bellingcat": 20,
    "poynter_ifcn": 20,
}

# ==================== 1. THE GUARDIAN SCRAPER (Giữ nguyên logic cũ) ====================
def crawl_guardian(limit=LIMITS["guardian"]):
    print("\n--- Starting The Guardian Scraper ---")
    
    categories = [
        {"name": "World", "url": "https://www.theguardian.com/international"},
        {"name": "Business", "url": "https://www.theguardian.com/business"},
        {"name": "Science", "url": "https://www.theguardian.com/science"},
    ]
    
    results = []
    
    for cat in categories:
        print(f"Fetching Guardian category: {cat['name']} ({cat['url']})")
        try:
            time.sleep(2)
            r = requests.get(cat['url'], headers=headers, timeout=15)
            soup = BeautifulSoup(r.content, "html.parser")
            
            links = soup.find_all("a", {"data-link-name": "article"})
            count = 0
            
            for link in links:
                if count >= limit:
                    break
                    
                href = link.get("href")
                if not href or not href.startswith("https://www.theguardian.com"):
                    continue
                
                print(f"  Crawling Guardian article: {href}")
                try:
                    time.sleep(1)
                    r_detail = requests.get(href, headers=headers, timeout=15)
                    soup_detail = BeautifulSoup(r_detail.content, "html.parser")
                    
                    title_elem = soup_detail.find("h1")
                    title = title_elem.text.strip() if title_elem else ""
                    
                    subtitle_elem = soup_detail.find("p", {"class": "dcr-standfirst"})
                    subtitle = subtitle_elem.text.strip() if subtitle_elem else ""
                    
                    paragraphs = []
                    for p in soup_detail.find_all("p"):
                        p_text = p.text.strip()
                        if p_text and len(p_text) > 20:
                            paragraphs.append(p_text)
                    
                    content = "\n".join(paragraphs[:15])
                    
                    date_elem = soup_detail.find("time")
                    date_str = date_elem.get("datetime") if date_elem else ""
                    
                    if title and content:
                        results.append({
                            "source_type": "website",
                            "source_name": "The Guardian",
                            "url": href,
                            "domain": cat['name'],
                            "label": "TRUE",
                            "claim": title,
                            "original_text": subtitle,
                            "evidence": "Published by The Guardian",
                            "justification": content,
                            "publish_date": date_str
                        })
                        count += 1
                        
                except Exception as e:
                    print(f"    [ERROR] Detail {href}: {e}")
                    
        except Exception as e:
            print(f"  [ERROR] Category {cat['url']}: {e}")
    
    print(f"Successfully crawled {len(results)} Guardian articles.")
    return results


# ==================== 2. CNN LITE SCRAPER ====================
def crawl_lite_cnn(limit=LIMITS["cnn_lite"]):
    print("\n--- Starting CNN Lite Scraper ---")
    url = "https://lite.cnn.com/"
    results = []
    
    try:
        print(f"Fetching CNN Lite: {url}")
        r = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(r.content, "html.parser")
        
        # Tìm các link bài viết (có dạng /2024/.../...)
        links = soup.find_all("a", href=re.compile(r"^/\d{4}/\d{2}/\d{2}/"))
        count = 0
        
        for link in links:
            if count >= limit:
                break
                
            href = "https://lite.cnn.com" + link.get("href")
            title = link.text.strip()
            
            if not title or len(title) < 10:
                continue
                
            print(f"  Crawling CNN Lite article: {href}")
            try:
                time.sleep(1)
                r_detail = requests.get(href, headers=headers, timeout=15)
                soup_detail = BeautifulSoup(r_detail.content, "html.parser")
                
                # CNN Lite thường có nội dung trong các thẻ <p> đơn giản
                paragraphs = []
                for p in soup_detail.find_all("p"):
                    p_text = p.text.strip()
                    if p_text and len(p_text) > 20:
                        paragraphs.append(p_text)
                
                content = "\n".join(paragraphs[:15])
                
                if title and content:
                    results.append({
                        "source_type": "website",
                        "source_name": "CNN Lite",
                        "url": href,
                        "domain": "Top Stories",
                        "label": "TRUE",
                        "claim": title,
                        "original_text": content[:200] + "...", # Lite ít khi có subtitle riêng
                        "evidence": "Published by CNN (Lite Version)",
                        "justification": content,
                        "publish_date": datetime.now().strftime("%Y-%m-%d")
                    })
                    count += 1
                    
            except Exception as e:
                print(f"    [ERROR] Detail {href}: {e}")
                
    except Exception as e:
        print(f"  [ERROR] CNN Lite main: {e}")
    
    print(f"Successfully crawled {len(results)} CNN Lite articles.")
    return results


# ==================== 3. AP NEWS SCRAPER ====================
def crawl_apnews(limit=LIMITS["apnews"]):
    print("\n--- Starting AP News Scraper ---")
    url = "https://apnews.com/"
    results = []
    
    try:
        print(f"Fetching AP News: {url}")
        r = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(r.content, "html.parser")
        
        # AP News thường có link bài viết chứa /article/
        links = soup.find_all("a", href=re.compile(r"https://apnews\.com/article/"))
        count = 0
        
        for link in links:
            if count >= limit:
                break
                
            href = link.get("href")
            if not href:
                continue
                
            print(f"  Crawling AP News article: {href}")
            try:
                time.sleep(1)
                r_detail = requests.get(href, headers=headers, timeout=15)
                soup_detail = BeautifulSoup(r_detail.content, "html.parser")
                
                title_elem = soup_detail.find("h1")
                title = title_elem.text.strip() if title_elem else ""
                
                paragraphs = []
                # AP News thường chứa nội dung trong div có class chứa "RichTextStoryBody" hoặc các thẻ p thông thường
                article_body = soup_detail.find("div", class_=re.compile(r"RichTextStoryBody|ArticleBody"))
                if article_body:
                    for p in article_body.find_all("p"):
                        p_text = p.text.strip()
                        if p_text and len(p_text) > 20:
                            paragraphs.append(p_text)
                else:
                    for p in soup_detail.find_all("p"):
                        p_text = p.text.strip()
                        if p_text and len(p_text) > 20:
                            paragraphs.append(p_text)
                
                content = "\n".join(paragraphs[:15])
                
                date_elem = soup_detail.find("time")
                date_str = date_elem.get("datetime") if date_elem else ""
                
                if title and content:
                    results.append({
                        "source_type": "website",
                        "source_name": "AP News",
                        "url": href,
                        "domain": "Top News",
                        "label": "TRUE",
                        "claim": title,
                        "original_text": content[:200] + "...",
                        "evidence": "Published by Associated Press",
                        "justification": content,
                        "publish_date": date_str
                    })
                    count += 1
                    
            except Exception as e:
                print(f"    [ERROR] Detail {href}: {e}")
                
    except Exception as e:
        print(f"  [ERROR] AP News main: {e}")
    
    print(f"Successfully crawled {len(results)} AP News articles.")
    return results


# ==================== 4. NPR SCRAPER (Dùng RSS để đảm bảo tính "tĩnh" tuyệt đối) ====================
def crawl_npr(limit=LIMITS["npr"]):
    print("\n--- Starting NPR Scraper (via RSS for maximum static reliability) ---")
    # text.npr.org đôi khi redirect hoặc trả về trang trống tùy region. 
    # RSS Feed của NPR là nguồn HTML/XML tĩnh, chuẩn và không bao giờ bị thay đổi class.
    url = "https://feeds.npr.org/1001/rss.xml" # World News
    results = []
    
    try:
        print(f"Fetching NPR RSS: {url}")
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        
        root = ET.fromstring(r.content)
        
        count = 0
        for item in root.findall(".//item"):
            if count >= limit:
                break
                
            title = item.find("title").text.strip() if item.find("title") is not None else ""
            link = item.find("link").text.strip() if item.find("link") is not None else ""
            description = item.find("description").text.strip() if item.find("description") is not None else ""
            pub_date = item.find("pubDate").text.strip() if item.find("pubDate") is not None else ""
            
            # Làm sạch description khỏi HTML tags nếu có
            clean_desc = BeautifulSoup(description, "html.parser").get_text().strip()
            
            if title and link:
                results.append({
                    "source_type": "website",
                    "source_name": "NPR",
                    "url": link,
                    "domain": "World News",
                    "label": "TRUE",
                    "claim": title,
                    "original_text": clean_desc[:200] + "...",
                    "evidence": "Published by NPR",
                    "justification": clean_desc, # RSS thường đã chứa tóm tắt đầy đủ
                    "publish_date": pub_date
                })
                count += 1
                
    except Exception as e:
        print(f"  [ERROR] NPR RSS: {e}")
    
    print(f"Successfully crawled {len(results)} NPR articles.")
    return results


# ==================== 5. LEGIBLE NEWS SCRAPER (TRUE) ====================
def crawl_legiblenews(limit=LIMITS["legiblenews"]):
    print("\n--- Starting Legible News Scraper ---")
    url = "https://legiblenews.com/"
    results = []

    try:
        print(f"Fetching Legible News: {url}")
        r = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(r.content, "html.parser")

        # Mỗi story là 1 <li> hoặc đoạn text có link nguồn
        # Tìm tất cả link bài viết nội bộ dạng /articles/...
        article_links = soup.find_all("a", href=re.compile(r"^/articles/"))
        seen = set()
        count = 0

        for link in article_links:
            if count >= limit:
                break
            href = "https://legiblenews.com" + link.get("href")
            title = link.text.strip()
            if not title or len(title) < 10 or href in seen:
                continue
            seen.add(href)

            print(f"  Crawling Legible News article: {href}")
            try:
                time.sleep(1)
                r_detail = requests.get(href, headers=headers, timeout=15)
                soup_detail = BeautifulSoup(r_detail.content, "html.parser")

                h1 = soup_detail.find("h1")
                claim = h1.text.strip() if h1 else title

                paragraphs = []
                for p in soup_detail.find_all("p"):
                    p_text = p.text.strip()
                    if p_text and len(p_text) > 20:
                        paragraphs.append(p_text)
                content = "\n".join(paragraphs[:15])

                date_elem = soup_detail.find("time")
                date_str = date_elem.get("datetime", "") if date_elem else ""

                if claim and content:
                    results.append({
                        "source_type": "website",
                        "source_name": "Legible News",
                        "url": href,
                        "domain": "World News",
                        "label": "TRUE",
                        "claim": claim,
                        "original_text": content[:200] + "...",
                        "evidence": "Published by Legible News",
                        "justification": content,
                        "publish_date": date_str
                    })
                    count += 1
            except Exception as e:
                print(f"    [ERROR] Detail {href}: {e}")

    except Exception as e:
        print(f"  [ERROR] Legible News main: {e}")

    print(f"Successfully crawled {len(results)} Legible News articles.")
    return results


# ==================== 6. HACKER NEWS SCRAPER (TRUE) ====================
def crawl_hackernews(limit=LIMITS["hackernews"]):
    print("\n--- Starting Hacker News Scraper ---")
    # Dùng HN Algolia API - JSON tĩnh, không cần parse HTML
    url = f"https://hn.algolia.com/api/v1/search?tags=front_page&hitsPerPage={limit}"
    results = []

    try:
        print(f"Fetching Hacker News API: {url}")
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()

        for item in data.get("hits", []):
            title = item.get("title", "").strip()
            story_url = item.get("url", "") or f"https://news.ycombinator.com/item?id={item.get('objectID', '')}"
            author = item.get("author", "")
            created_at = item.get("created_at", "")
            points = item.get("points", 0)

            if not title:
                continue

            justification = f"Submitted by {author} with {points} points on Hacker News (news.ycombinator.com). Original source: {story_url}"

            results.append({
                "source_type": "website",
                "source_name": "Hacker News",
                "url": story_url,
                "domain": "Technology / Science",
                "label": "TRUE",
                "claim": title,
                "original_text": title,
                "evidence": "Submitted to Hacker News (news.ycombinator.com) front page",
                "justification": justification,
                "publish_date": created_at
            })

    except Exception as e:
        print(f"  [ERROR] Hacker News API: {e}")

    print(f"Successfully crawled {len(results)} Hacker News items.")
    return results


# ==================== 7. FACTCHECK VIRAL SPIRAL SCRAPER (FALSE) ====================
def crawl_factcheck_viral(limit=LIMITS["factcheck_viral"]):
    print("\n--- Starting FactCheck Viral Spiral Scraper ---")
    url = "https://www.factcheck.org/viral-spiral/"
    results = []

    try:
        print(f"Fetching FactCheck Viral Spiral: {url}")
        r = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(r.content, "html.parser")

        # Bài viết trong các thẻ <article> hoặc <h3>/<h2> có link
        articles = soup.find_all("h3", class_=re.compile(r"entry-title|article-title"))
        if not articles:
            articles = soup.find_all("h3")

            count = 0
        for art in articles:
            if count >= limit:
                break
            a_tag = art.find("a")
            if not a_tag:
                continue
            href = a_tag.get("href", "")
            title = a_tag.text.strip()
            if not href or not title:
                continue

            print(f"  Crawling FactCheck article: {href}")
            try:
                time.sleep(1)
                r_detail = requests.get(href, headers=headers, timeout=15)
                soup_detail = BeautifulSoup(r_detail.content, "html.parser")

                paragraphs = []
                body = soup_detail.find("div", class_=re.compile(r"entry-content|article-content|post-content"))
                if body:
                    for p in body.find_all("p"):
                        p_text = p.text.strip()
                        if p_text and len(p_text) > 20:
                            paragraphs.append(p_text)
                content = "\n".join(paragraphs[:15])

                date_elem = soup_detail.find("time")
                date_str = date_elem.get("datetime", "") if date_elem else ""

                if title and content:
                    results.append({
                        "source_type": "website",
                        "source_name": "FactCheck.org (Viral Spiral)",
                        "url": href,
                        "domain": "Misinformation / Viral Claims",
                        "label": "FALSE",
                        "claim": title,
                        "original_text": content[:200] + "...",
                        "evidence": "Debunked by FactCheck.org Viral Spiral",
                        "justification": content,
                        "publish_date": date_str
                    })
                    count += 1
            except Exception as e:
                print(f"    [ERROR] Detail {href}: {e}")

    except Exception as e:
        print(f"  [ERROR] FactCheck Viral Spiral: {e}")

    print(f"Successfully crawled {len(results)} FactCheck Viral Spiral articles.")
    return results


# ==================== 8. FACTCHECK SCICHECK SCRAPER (FALSE) ====================
def crawl_factcheck_scicheck(limit=LIMITS["factcheck_scicheck"]):
    print("\n--- Starting FactCheck SciCheck Scraper ---")
    url = "https://www.factcheck.org/scicheck/"
    results = []

    try:
        print(f"Fetching FactCheck SciCheck: {url}")
        r = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(r.content, "html.parser")

        articles = soup.find_all("h3", class_=re.compile(r"entry-title|article-title"))
        if not articles:
            articles = soup.find_all("h3")

            count = 0
        for art in articles:
            if count >= limit:
                break
            a_tag = art.find("a")
            if not a_tag:
                continue
            href = a_tag.get("href", "")
            title = a_tag.text.strip()
            if not href or not title:
                continue

            print(f"  Crawling SciCheck article: {href}")
            try:
                time.sleep(1)
                r_detail = requests.get(href, headers=headers, timeout=15)
                soup_detail = BeautifulSoup(r_detail.content, "html.parser")

                paragraphs = []
                body = soup_detail.find("div", class_=re.compile(r"entry-content|article-content|post-content"))
                if body:
                    for p in body.find_all("p"):
                        p_text = p.text.strip()
                        if p_text and len(p_text) > 20:
                            paragraphs.append(p_text)
                content = "\n".join(paragraphs[:15])

                date_elem = soup_detail.find("time")
                date_str = date_elem.get("datetime", "") if date_elem else ""

                if title and content:
                    results.append({
                        "source_type": "website",
                        "source_name": "FactCheck.org (SciCheck)",
                        "url": href,
                        "domain": "Science Misinformation",
                        "label": "FALSE",
                        "claim": title,
                        "original_text": content[:200] + "...",
                        "evidence": "Debunked by FactCheck.org SciCheck",
                        "justification": content,
                        "publish_date": date_str
                    })
                    count += 1
            except Exception as e:
                print(f"    [ERROR] Detail {href}: {e}")

    except Exception as e:
        print(f"  [ERROR] FactCheck SciCheck: {e}")

    print(f"Successfully crawled {len(results)} FactCheck SciCheck articles.")
    return results


# ==================== 9. SNOPES SCRAPER (FALSE) ====================
def crawl_snopes(limit=LIMITS["snopes"]):
    print("\n--- Starting Snopes Scraper ---")
    url = "https://www.snopes.com/fact-check/"
    results = []

    try:
        print(f"Fetching Snopes: {url}")
        r = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(r.content, "html.parser")

        # Snopes dùng <article> với link bên trong
        articles = soup.find_all("article")
        count = 0

        for art in articles:
            if count >= limit:
                break
            a_tag = art.find("a", href=re.compile(r"https://www\.snopes\.com/fact-check/"))
            if not a_tag:
                continue
            href = a_tag.get("href", "")
            # Title thường trong <span class='title'> hoặc <h2>
            title_elem = art.find("span", class_=re.compile(r"title")) or art.find("h2") or art.find("h3")
            title = title_elem.text.strip() if title_elem else a_tag.text.strip()
            if not href or not title or len(title) < 10:
                continue

            print(f"  Crawling Snopes article: {href}")
            try:
                time.sleep(1)
                r_detail = requests.get(href, headers=headers, timeout=15)
                soup_detail = BeautifulSoup(r_detail.content, "html.parser")

                paragraphs = []
                body = soup_detail.find("div", class_=re.compile(r"single-body|article-text|fact-check-body"))
                if not body:
                    body = soup_detail.find("article")
                if body:
                    for p in body.find_all("p"):
                        p_text = p.text.strip()
                        if p_text and len(p_text) > 20:
                            paragraphs.append(p_text)
                content = "\n".join(paragraphs[:15])

                date_elem = soup_detail.find("time")
                date_str = date_elem.get("datetime", "") if date_elem else ""

                if title and content:
                    results.append({
                        "source_type": "website",
                        "source_name": "Snopes",
                        "url": href,
                        "domain": "Fact-Check / Misinformation",
                        "label": "FALSE",
                        "claim": title,
                        "original_text": content[:200] + "...",
                        "evidence": "Debunked by Snopes.com fact-checkers",
                        "justification": content,
                        "publish_date": date_str
                    })
                    count += 1
            except Exception as e:
                print(f"    [ERROR] Detail {href}: {e}")

    except Exception as e:
        print(f"  [ERROR] Snopes: {e}")

    print(f"Successfully crawled {len(results)} Snopes articles.")
    return results


# ==================== 10. BELLINGCAT SCRAPER (FALSE) ====================
def crawl_bellingcat(limit=LIMITS["bellingcat"]):
    print("\n--- Starting Bellingcat Scraper ---")
    url = "https://www.bellingcat.com/"
    results = []

    try:
        print(f"Fetching Bellingcat: {url}")
        r = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(r.content, "html.parser")

        # Link bài viết dạng /news/YYYY/MM/DD/...
        links = soup.find_all("a", href=re.compile(r"https://www\.bellingcat\.com/news/\d{4}/"))
        seen = set()
        count = 0

        for link in links:
            if count >= limit:
                break
            href = link.get("href", "")
            title = link.text.strip()
            if not href or not title or len(title) < 10 or href in seen:
                continue
            seen.add(href)

            print(f"  Crawling Bellingcat article: {href}")
            try:
                time.sleep(1)
                r_detail = requests.get(href, headers=headers, timeout=15)
                soup_detail = BeautifulSoup(r_detail.content, "html.parser")

                h1 = soup_detail.find("h1")
                claim = h1.text.strip() if h1 else title

                paragraphs = []
                body = soup_detail.find("div", class_=re.compile(r"entry-content|post-content|article-content"))
                if not body:
                    body = soup_detail.find("article")
                if body:
                    for p in body.find_all("p"):
                        p_text = p.text.strip()
                        if p_text and len(p_text) > 20:
                            paragraphs.append(p_text)
                content = "\n".join(paragraphs[:15])

                date_elem = soup_detail.find("time")
                date_str = date_elem.get("datetime", "") if date_elem else ""

                if claim and content:
                    results.append({
                        "source_type": "website",
                        "source_name": "Bellingcat",
                        "url": href,
                        "domain": "Misinformation / Investigative",
                        "label": "FALSE",
                        "claim": claim,
                        "original_text": content[:200] + "...",
                        "evidence": "Investigated and debunked by Bellingcat",
                        "justification": content,
                        "publish_date": date_str
                    })
                    count += 1
            except Exception as e:
                print(f"    [ERROR] Detail {href}: {e}")

    except Exception as e:
        print(f"  [ERROR] Bellingcat: {e}")

    print(f"Successfully crawled {len(results)} Bellingcat articles.")
    return results


# ==================== 11. POYNTER IFCN SCRAPER (FALSE) ====================
def crawl_poynter_ifcn(limit=LIMITS["poynter_ifcn"]):
    print("\n--- Starting Poynter IFCN Scraper ---")
    url = "https://www.poynter.org/ifcn/"
    results = []

    try:
        print(f"Fetching Poynter IFCN: {url}")
        r = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(r.content, "html.parser")

        # Link bài viết dạng /fact-checking/YYYY/... hoặc /news/ifcn/
        links = soup.find_all("a", href=re.compile(r"https://www\.poynter\.org/(fact-checking|news|ifcn)/\d{4}/"))
        seen = set()
        count = 0

        for link in links:
            if count >= limit:
                break
            href = link.get("href", "")
            title = link.text.strip()
            if not href or not title or len(title) < 10 or href in seen:
                continue
            seen.add(href)

            print(f"  Crawling Poynter article: {href}")
            try:
                time.sleep(1)
                r_detail = requests.get(href, headers=headers, timeout=15)
                soup_detail = BeautifulSoup(r_detail.content, "html.parser")

                h1 = soup_detail.find("h1")
                claim = h1.text.strip() if h1 else title

                paragraphs = []
                body = soup_detail.find("div", class_=re.compile(r"entry-content|article-content|post-content"))
                if not body:
                    body = soup_detail.find("article")
                if body:
                    for p in body.find_all("p"):
                        p_text = p.text.strip()
                        if p_text and len(p_text) > 20:
                            paragraphs.append(p_text)
                content = "\n".join(paragraphs[:15])

                date_elem = soup_detail.find("time")
                date_str = date_elem.get("datetime", "") if date_elem else ""

                if claim and content:
                    results.append({
                        "source_type": "website",
                        "source_name": "Poynter IFCN",
                        "url": href,
                        "domain": "Fact-Checking / Media Integrity",
                        "label": "FALSE",
                        "claim": claim,
                        "original_text": content[:200] + "...",
                        "evidence": "Reported by Poynter International Fact-Checking Network (IFCN)",
                        "justification": content,
                        "publish_date": date_str
                    })
                    count += 1
            except Exception as e:
                print(f"    [ERROR] Detail {href}: {e}")

    except Exception as e:
        print(f"  [ERROR] Poynter IFCN: {e}")

    print(f"Successfully crawled {len(results)} Poynter IFCN articles.")
    return results


# ==================== MAIN CRAWLER PROCESS ====================
def main():
    print("=================== ENGLISH NEWS WEB CRAWLER (STATIC SOURCES) ===================")
    
    # TRUE sources
    guardian_data = crawl_guardian()
    cnn_lite_data = crawl_lite_cnn()
    apnews_data = crawl_apnews()
    npr_data = crawl_npr()
    legible_data = crawl_legiblenews()
    hackernews_data = crawl_hackernews()

    # FALSE sources
    factcheck_viral_data = crawl_factcheck_viral()
    factcheck_sci_data = crawl_factcheck_scicheck()
    snopes_data = crawl_snopes()
    bellingcat_data = crawl_bellingcat()
    poynter_data = crawl_poynter_ifcn()

    # Combine all data
    all_data = (guardian_data + cnn_lite_data + apnews_data + npr_data +
                legible_data + hackernews_data +
                factcheck_viral_data + factcheck_sci_data +
                snopes_data + bellingcat_data + poynter_data)

    # Save to file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)

    print(f"\n=================== SCRAPING COMPLETE ===================")
    print(f"Total entries collected: {len(all_data)}")
    print(f"  [TRUE]  The Guardian:          {len(guardian_data)}")
    print(f"  [TRUE]  CNN Lite:              {len(cnn_lite_data)}")
    print(f"  [TRUE]  AP News:               {len(apnews_data)}")
    print(f"  [TRUE]  NPR:                   {len(npr_data)}")
    print(f"  [TRUE]  Legible News:          {len(legible_data)}")
    print(f"  [TRUE]  Hacker News:           {len(hackernews_data)}")
    print(f"  [FALSE] FactCheck Viral:       {len(factcheck_viral_data)}")
    print(f"  [FALSE] FactCheck SciCheck:    {len(factcheck_sci_data)}")
    print(f"  [FALSE] Snopes:                {len(snopes_data)}")
    print(f"  [FALSE] Bellingcat:            {len(bellingcat_data)}")
    print(f"  [FALSE] Poynter IFCN:          {len(poynter_data)}")
    print(f"Saved dataset to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
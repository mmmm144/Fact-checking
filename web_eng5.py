import requests
from bs4 import BeautifulSoup
import json
import time
import re
import urllib3
from datetime import datetime

urllib3.disable_warnings()

OUTPUT_FILE = "news_website_output_en.json"

# List User-Agents
user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
]

import random

def get_random_user_agent():
    return random.choice(user_agents)

# Dùng:
headers = {
    "User-Agent": get_random_user_agent()
}

# Helper: Parse date from various formats
def parse_date(date_str):
    try:
        # Try common date formats
        date_formats = [
            r'(\d{1,2})/(\d{1,2})/(\d{4})',  # dd/mm/yyyy
            r'(\w+)\s+(\d{1,2}),?\s+(\d{4})',  # Month dd, yyyy
            r'(\d{4})-(\d{1,2})-(\d{1,2})',  # yyyy-mm-dd
        ]
        
        for fmt in date_formats:
            match = re.search(fmt, date_str)
            if match:
                return datetime.now().year  # Just check if date can be parsed
    except:
        pass
    return None

# ==================== NYT SCRAPER ====================
def crawl_nytimes():
    print("\n--- Starting New York Times Scraper ---")
    
    categories = [
        {"name": "World", "url": "https://www.nytimes.com/section/world"},
        {"name": "Business", "url": "https://www.nytimes.com/section/business"},
        {"name": "Health", "url": "https://www.nytimes.com/section/health"},
    ]
    
    results = []
    
    for cat in categories:
        print(f"Fetching NYT category: {cat['name']} ({cat['url']})")
        try:
            time.sleep(2)
            r = requests.get(cat['url'], headers=headers, timeout=15)
            soup = BeautifulSoup(r.content, "html.parser")
            
            # Find article links - look for <a> tags inside <article> elements
            article_elements = soup.find_all("article")
            count = 0
            
            for article_elem in article_elements:
                if count >= 5:
                    break
                
                # Get the first link in this article that points to nytimes.com
                article_link = article_elem.find('a', href=lambda x: x and 'nytimes.com' in x)
                if not article_link:
                    continue
                    
                href = article_link.get("href")
                if not href or not href.startswith("https://www.nytimes.com"):
                    continue
                
                print(f"  Crawling NYT article: {href}")
                try:
                    time.sleep(1)
                    r_detail = requests.get(href, headers=headers, timeout=15)
                    soup_detail = BeautifulSoup(r_detail.content, "html.parser")
                    
                    # Extract title
                    title_elem = soup_detail.find("h1")
                    title = title_elem.text.strip() if title_elem else ""
                    
                    # Extract subtitle/description
                    subtitle_elem = soup_detail.find("p", {"data-testid": "subtitle"})
                    subtitle = subtitle_elem.text.strip() if subtitle_elem else ""
                    
                    # Extract content
                    content_elem = soup_detail.find("article")
                    paragraphs = []
                    if content_elem:
                        for p in content_elem.find_all("p", {"data-testid": "paragraph"}):
                            p_text = p.text.strip()
                            if p_text and len(p_text) > 20:
                                paragraphs.append(p_text)
                    
                    content = "\n".join(paragraphs)
                    
                    # Extract date
                    date_elem = soup_detail.find("time")
                    date_str = date_elem.get("datetime") if date_elem else ""
                    
                    if title and content:
                        results.append({
                            "source_type": "website",
                            "source_name": "New York Times",
                            "url": href,
                            "domain": cat['name'],
                            "label": "TRUE",
                            "claim": title,
                            "original_text": subtitle,
                            "evidence": "Published by New York Times",
                            "justification": content,
                            "publish_date": date_str
                        })
                        count += 1
                        
                except Exception as e:
                    print(f"    [ERROR] Detail {href}: {e}")
                    
        except Exception as e:
            print(f"  [ERROR] Category {cat['url']}: {e}")
    
    print(f"Successfully crawled {len(results)} NYT articles.")
    return results


# ==================== WASHINGTON POST SCRAPER ====================
def crawl_washington_post():
    print("\n--- Starting Washington Post Scraper ---")
    
    categories = [
        {"name": "World", "url": "https://www.washingtonpost.com/world/"},
        {"name": "Business", "url": "https://www.washingtonpost.com/business/"},
        {"name": "Health", "url": "https://www.washingtonpost.com/health/"},
    ]
    
    results = []
    
    for cat in categories:
        print(f"Fetching WP category: {cat['name']} ({cat['url']})")
        try:
            time.sleep(2)
            r = requests.get(cat['url'], headers=headers, timeout=15)
            soup = BeautifulSoup(r.content, "html.parser")
            
            # Find article links
            links = soup.find_all("a", {"class": "link"})
            count = 0
            
            for link in links:
                if count >= 5:
                    break
                    
                href = link.get("href")
                if not href or not href.startswith("https://www.washingtonpost.com"):
                    continue
                
                print(f"  Crawling WP article: {href}")
                try:
                    time.sleep(1)
                    r_detail = requests.get(href, headers=headers, timeout=15)
                    soup_detail = BeautifulSoup(r_detail.content, "html.parser")
                    
                    # Extract title
                    title_elem = soup_detail.find("h1", {"class": "headline"})
                    title = title_elem.text.strip() if title_elem else ""
                    
                    # Extract subtitle
                    subtitle_elem = soup_detail.find("p", {"class": "subheadline"})
                    subtitle = subtitle_elem.text.strip() if subtitle_elem else ""
                    
                    # Extract content
                    paragraphs = []
                    for p in soup_detail.find_all("p", {"class": "paragraph"}):
                        p_text = p.text.strip()
                        if p_text and len(p_text) > 20:
                            paragraphs.append(p_text)
                    
                    content = "\n".join(paragraphs)
                    
                    # Extract date
                    date_elem = soup_detail.find("span", {"data-testid": "publish-date"})
                    date_str = date_elem.text.strip() if date_elem else ""
                    
                    if title and content:
                        results.append({
                            "source_type": "website",
                            "source_name": "Washington Post",
                            "url": href,
                            "domain": cat['name'],
                            "label": "TRUE",
                            "claim": title,
                            "original_text": subtitle,
                            "evidence": "Published by Washington Post",
                            "justification": content,
                            "publish_date": date_str
                        })
                        count += 1
                        
                except Exception as e:
                    print(f"    [ERROR] Detail {href}: {e}")
                    
        except Exception as e:
            print(f"  [ERROR] Category {cat['url']}: {e}")
    
    print(f"Successfully crawled {len(results)} WP articles.")
    return results


# ==================== WALL STREET JOURNAL SCRAPER ====================
def crawl_wsj():
    print("\n--- Starting Wall Street Journal Scraper ---")
    
    categories = [
        {"name": "World", "url": "https://www.wsj.com/news/world"},
        {"name": "Business", "url": "https://www.wsj.com/news/business"},
        {"name": "Health", "url": "https://www.wsj.com/news/health"},
    ]
    
    results = []
    
    for cat in categories:
        print(f"Fetching WSJ category: {cat['name']} ({cat['url']})")
        try:
            time.sleep(2)
            r = requests.get(cat['url'], headers=headers, timeout=15)
            soup = BeautifulSoup(r.content, "html.parser")
            
            # Find article links
            links = soup.find_all("a", {"class": "WSJTheme--headlineLink"})
            count = 0
            
            for link in links:
                if count >= 5:
                    break
                    
                href = link.get("href")
                if not href:
                    continue
                
                if not href.startswith("http"):
                    href = "https://www.wsj.com" + href
                
                print(f"  Crawling WSJ article: {href}")
                try:
                    time.sleep(1)
                    r_detail = requests.get(href, headers=headers, timeout=15)
                    soup_detail = BeautifulSoup(r_detail.content, "html.parser")
                    
                    # Extract title
                    title_elem = soup_detail.find("h1", {"class": "WSJTheme--headline"})
                    title = title_elem.text.strip() if title_elem else ""
                    
                    # Extract subtitle
                    subtitle_elem = soup_detail.find("p", {"class": "WSJTheme--description"})
                    subtitle = subtitle_elem.text.strip() if subtitle_elem else ""
                    
                    # Extract content
                    paragraphs = []
                    for p in soup_detail.find_all("p"):
                        p_text = p.text.strip()
                        if p_text and len(p_text) > 30 and not p_text.startswith("From"):
                            paragraphs.append(p_text)
                    
                    content = "\n".join(paragraphs[:10])  # Limit to 10 paragraphs
                    
                    # Extract date
                    date_elem = soup_detail.find("time")
                    date_str = date_elem.get("datetime") if date_elem else ""
                    
                    if title and content:
                        results.append({
                            "source_type": "website",
                            "source_name": "Wall Street Journal",
                            "url": href,
                            "domain": cat['name'],
                            "label": "TRUE",
                            "claim": title,
                            "original_text": subtitle,
                            "evidence": "Published by Wall Street Journal",
                            "justification": content,
                            "publish_date": date_str
                        })
                        count += 1
                        
                except Exception as e:
                    print(f"    [ERROR] Detail {href}: {e}")
                    
        except Exception as e:
            print(f"  [ERROR] Category {cat['url']}: {e}")
    
    print(f"Successfully crawled {len(results)} WSJ articles.")
    return results


# ==================== AP NEWS SCRAPER ====================
def crawl_apnews():
    print("\n--- Starting AP News Scraper ---")
    
    categories = [
        {"name": "World", "url": "https://apnews.com/hub/world-news"},
        {"name": "Business", "url": "https://apnews.com/hub/business"},
        {"name": "Health", "url": "https://apnews.com/hub/health"},
    ]
    
    results = []
    
    for cat in categories:
        print(f"Fetching AP News category: {cat['name']} ({cat['url']})")
        try:
            time.sleep(2)
            r = requests.get(cat['url'], headers=headers, timeout=15)
            soup = BeautifulSoup(r.content, "html.parser")
            
            # Find article links
            links = soup.find_all("a", {"class": "component-headline-link"})
            count = 0
            
            for link in links:
                if count >= 5:
                    break
                    
                href = link.get("href")
                if not href or not href.startswith("https://apnews.com"):
                    continue
                
                print(f"  Crawling AP News article: {href}")
                try:
                    time.sleep(1)
                    r_detail = requests.get(href, headers=headers, timeout=15)
                    soup_detail = BeautifulSoup(r_detail.content, "html.parser")
                    
                    # Extract title
                    title_elem = soup_detail.find("h1")
                    title = title_elem.text.strip() if title_elem else ""
                    
                    # Extract subtitle/description
                    subtitle_elem = soup_detail.find("p", {"class": "Component-description"})
                    subtitle = subtitle_elem.text.strip() if subtitle_elem else ""
                    
                    # Extract content
                    paragraphs = []
                    article_elem = soup_detail.find("article")
                    if article_elem:
                        for p in article_elem.find_all("p"):
                            p_text = p.text.strip()
                            if p_text and len(p_text) > 20:
                                paragraphs.append(p_text)
                    
                    content = "\n".join(paragraphs[:15])
                    
                    # Extract date
                    date_elem = soup_detail.find("span", {"class": "Component-timestamp"})
                    date_str = date_elem.text.strip() if date_elem else ""
                    
                    if title and content:
                        results.append({
                            "source_type": "website",
                            "source_name": "AP News",
                            "url": href,
                            "domain": cat['name'],
                            "label": "TRUE",
                            "claim": title,
                            "original_text": subtitle,
                            "evidence": "Published by Associated Press",
                            "justification": content,
                            "publish_date": date_str
                        })
                        count += 1
                        
                except Exception as e:
                    print(f"    [ERROR] Detail {href}: {e}")
                    
        except Exception as e:
            print(f"  [ERROR] Category {cat['url']}: {e}")
    
    print(f"Successfully crawled {len(results)} AP News articles.")
    return results


# ==================== REUTERS SCRAPER ====================
def crawl_reuters():
    print("\n--- Starting Reuters Scraper ---")
    
    categories = [
        {"name": "World", "url": "https://www.reuters.com/world/"},
        {"name": "Business", "url": "https://www.reuters.com/business/"},
        {"name": "Health", "url": "https://www.reuters.com/health/"},
    ]
    
    results = []
    
    for cat in categories:
        print(f"Fetching Reuters category: {cat['name']} ({cat['url']})")
        try:
            time.sleep(2)
            r = requests.get(cat['url'], headers=headers, timeout=15)
            soup = BeautifulSoup(r.content, "html.parser")
            
            # Find article links
            links = soup.find_all("a", {"data-testid": "Link"})
            count = 0
            
            for link in links:
                if count >= 5:
                    break
                    
                href = link.get("href")
                if not href or not href.startswith("https://www.reuters.com"):
                    continue
                
                print(f"  Crawling Reuters article: {href}")
                try:
                    time.sleep(1)
                    r_detail = requests.get(href, headers=headers, timeout=15)
                    soup_detail = BeautifulSoup(r_detail.content, "html.parser")
                    
                    # Extract title
                    title_elem = soup_detail.find("h1")
                    title = title_elem.text.strip() if title_elem else ""
                    
                    # Extract subtitle
                    subtitle_elem = soup_detail.find("p", {"data-testid": "Heading"})
                    subtitle = subtitle_elem.text.strip() if subtitle_elem else ""
                    
                    # Extract content
                    paragraphs = []
                    article_elem = soup_detail.find("article")
                    if article_elem:
                        for p in article_elem.find_all("p"):
                            p_text = p.text.strip()
                            if p_text and len(p_text) > 20:
                                paragraphs.append(p_text)
                    
                    content = "\n".join(paragraphs[:15])
                    
                    # Extract date
                    date_elem = soup_detail.find("time")
                    date_str = date_elem.get("datetime") if date_elem else ""
                    
                    if title and content:
                        results.append({
                            "source_type": "website",
                            "source_name": "Reuters",
                            "url": href,
                            "domain": cat['name'],
                            "label": "TRUE",
                            "claim": title,
                            "original_text": subtitle,
                            "evidence": "Published by Reuters",
                            "justification": content,
                            "publish_date": date_str
                        })
                        count += 1
                        
                except Exception as e:
                    print(f"    [ERROR] Detail {href}: {e}")
                    
        except Exception as e:
            print(f"  [ERROR] Category {cat['url']}: {e}")
    
    print(f"Successfully crawled {len(results)} Reuters articles.")
    return results


# ==================== THE GUARDIAN SCRAPER ====================
def crawl_guardian():
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
            
            # Find article links
            links = soup.find_all("a", {"data-link-name": "article"})
            count = 0
            
            for link in links:
                if count >= 5:
                    break
                    
                href = link.get("href")
                if not href or not href.startswith("https://www.theguardian.com"):
                    continue
                
                print(f"  Crawling Guardian article: {href}")
                try:
                    time.sleep(1)
                    r_detail = requests.get(href, headers=headers, timeout=15)
                    soup_detail = BeautifulSoup(r_detail.content, "html.parser")
                    
                    # Extract title
                    title_elem = soup_detail.find("h1")
                    title = title_elem.text.strip() if title_elem else ""
                    
                    # Extract subtitle
                    subtitle_elem = soup_detail.find("p", {"class": "dcr-standfirst"})
                    subtitle = subtitle_elem.text.strip() if subtitle_elem else ""
                    
                    # Extract content
                    paragraphs = []
                    for p in soup_detail.find_all("p"):
                        p_text = p.text.strip()
                        if p_text and len(p_text) > 20:
                            paragraphs.append(p_text)
                    
                    content = "\n".join(paragraphs[:15])
                    
                    # Extract date
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


# ==================== FINANCIAL TIMES SCRAPER ====================
def crawl_ft():
    print("\n--- Starting Financial Times Scraper ---")
    
    categories = [
        {"name": "World", "url": "https://www.ft.com/world"},
        {"name": "Business", "url": "https://www.ft.com/business"},
        {"name": "Markets", "url": "https://www.ft.com/markets"},
    ]
    
    results = []
    
    for cat in categories:
        print(f"Fetching FT category: {cat['name']} ({cat['url']})")
        try:
            time.sleep(2)
            r = requests.get(cat['url'], headers=headers, timeout=15)
            soup = BeautifulSoup(r.content, "html.parser")
            
            # Find article links
            links = soup.find_all("a", {"class": "js-teaser-link"})
            count = 0
            
            for link in links:
                if count >= 5:
                    break
                    
                href = link.get("href")
                if not href:
                    continue
                
                if not href.startswith("http"):
                    href = "https://www.ft.com" + href
                
                print(f"  Crawling FT article: {href}")
                try:
                    time.sleep(1)
                    r_detail = requests.get(href, headers=headers, timeout=15)
                    soup_detail = BeautifulSoup(r_detail.content, "html.parser")
                    
                    # Extract title
                    title_elem = soup_detail.find("h1")
                    title = title_elem.text.strip() if title_elem else ""
                    
                    # Extract subtitle
                    subtitle_elem = soup_detail.find("p", {"class": "article__standfirst"})
                    subtitle = subtitle_elem.text.strip() if subtitle_elem else ""
                    
                    # Extract content
                    paragraphs = []
                    for p in soup_detail.find_all("p"):
                        p_text = p.text.strip()
                        if p_text and len(p_text) > 20 and not p_text.startswith("©"):
                            paragraphs.append(p_text)
                    
                    content = "\n".join(paragraphs[:15])
                    
                    # Extract date
                    date_elem = soup_detail.find("time")
                    date_str = date_elem.get("datetime") if date_elem else ""
                    
                    if title and content:
                        results.append({
                            "source_type": "website",
                            "source_name": "Financial Times",
                            "url": href,
                            "domain": cat['name'],
                            "label": "TRUE",
                            "claim": title,
                            "original_text": subtitle,
                            "evidence": "Published by Financial Times",
                            "justification": content,
                            "publish_date": date_str
                        })
                        count += 1
                        
                except Exception as e:
                    print(f"    [ERROR] Detail {href}: {e}")
                    
        except Exception as e:
            print(f"  [ERROR] Category {cat['url']}: {e}")
    
    print(f"Successfully crawled {len(results)} FT articles.")
    return results


# ==================== CNN SCRAPER ====================
def crawl_cnn():
    print("\n--- Starting CNN Scraper ---")
    
    categories = [
        {"name": "World", "url": "https://www.cnn.com/world"},
        {"name": "Business", "url": "https://www.cnn.com/business"},
        {"name": "Health", "url": "https://www.cnn.com/health"},
    ]
    
    results = []
    
    for cat in categories:
        print(f"Fetching CNN category: {cat['name']} ({cat['url']})")
        try:
            time.sleep(2)
            r = requests.get(cat['url'], headers=headers, timeout=15)
            soup = BeautifulSoup(r.content, "html.parser")
            
            # Find article links
            links = soup.find_all("span", {"class": "container__headline-text"})
            count = 0
            
            for link_span in links:
                if count >= 5:
                    break
                    
                link = link_span.find_parent("a")
                if not link:
                    continue
                    
                href = link.get("href")
                if not href or not href.startswith("https://www.cnn.com"):
                    continue
                
                print(f"  Crawling CNN article: {href}")
                try:
                    time.sleep(1)
                    r_detail = requests.get(href, headers=headers, timeout=15)
                    soup_detail = BeautifulSoup(r_detail.content, "html.parser")
                    
                    # Extract title
                    title_elem = soup_detail.find("span", {"class": "container__headline-text"})
                    title = title_elem.text.strip() if title_elem else ""
                    
                    # Extract subtitle
                    subtitle_elem = soup_detail.find("p", {"class": "container__headline-subtext"})
                    subtitle = subtitle_elem.text.strip() if subtitle_elem else ""
                    
                    # Extract content
                    paragraphs = []
                    for p in soup_detail.find_all("p"):
                        p_text = p.text.strip()
                        if p_text and len(p_text) > 20:
                            paragraphs.append(p_text)
                    
                    content = "\n".join(paragraphs[:15])
                    
                    # Extract date
                    date_elem = soup_detail.find("span", {"class": "container__publish-date"})
                    date_str = date_elem.text.strip() if date_elem else ""
                    
                    if title and content:
                        results.append({
                            "source_type": "website",
                            "source_name": "CNN",
                            "url": href,
                            "domain": cat['name'],
                            "label": "TRUE",
                            "claim": title,
                            "original_text": subtitle,
                            "evidence": "Published by CNN",
                            "justification": content,
                            "publish_date": date_str
                        })
                        count += 1
                        
                except Exception as e:
                    print(f"    [ERROR] Detail {href}: {e}")
                    
        except Exception as e:
            print(f"  [ERROR] Category {cat['url']}: {e}")
    
    print(f"Successfully crawled {len(results)} CNN articles.")
    return results


# ==================== BLOOMBERG SCRAPER ====================
def crawl_bloomberg():
    print("\n--- Starting Bloomberg Scraper ---")
    
    categories = [
        {"name": "Markets", "url": "https://www.bloomberg.com/asia"},
        {"name": "Technology", "url": "https://www.bloomberg.com/technology"},
        {"name": "Business", "url": "https://www.bloomberg.com/business"},
    ]
    
    results = []
    
    for cat in categories:
        print(f"Fetching Bloomberg category: {cat['name']} ({cat['url']})")
        try:
            time.sleep(2)
            r = requests.get(cat['url'], headers=headers, timeout=15)
            soup = BeautifulSoup(r.content, "html.parser")
            
            # Find article links
            links = soup.find_all("a", {"class": "storyLink"})
            count = 0
            
            for link in links:
                if count >= 5:
                    break
                    
                href = link.get("href")
                if not href:
                    continue
                
                if not href.startswith("http"):
                    href = "https://www.bloomberg.com" + href
                
                print(f"  Crawling Bloomberg article: {href}")
                try:
                    time.sleep(1)
                    r_detail = requests.get(href, headers=headers, timeout=15)
                    soup_detail = BeautifulSoup(r_detail.content, "html.parser")
                    
                    # Extract title
                    title_elem = soup_detail.find("h1")
                    title = title_elem.text.strip() if title_elem else ""
                    
                    # Extract subtitle
                    subtitle_elem = soup_detail.find("p", {"class": "paywall"})
                    if not subtitle_elem:
                        subtitle_elem = soup_detail.find("p")
                    subtitle = subtitle_elem.text.strip() if subtitle_elem else ""
                    
                    # Extract content
                    paragraphs = []
                    for p in soup_detail.find_all("p"):
                        p_text = p.text.strip()
                        if p_text and len(p_text) > 20 and not p_text.startswith("©"):
                            paragraphs.append(p_text)
                    
                    content = "\n".join(paragraphs[:15])
                    
                    # Extract date
                    date_elem = soup_detail.find("time")
                    date_str = date_elem.get("datetime") if date_elem else ""
                    
                    if title and content:
                        results.append({
                            "source_type": "website",
                            "source_name": "Bloomberg",
                            "url": href,
                            "domain": cat['name'],
                            "label": "TRUE",
                            "claim": title,
                            "original_text": subtitle,
                            "evidence": "Published by Bloomberg",
                            "justification": content,
                            "publish_date": date_str
                        })
                        count += 1
                        
                except Exception as e:
                    print(f"    [ERROR] Detail {href}: {e}")
                    
        except Exception as e:
            print(f"  [ERROR] Category {cat['url']}: {e}")
    
    print(f"Successfully crawled {len(results)} Bloomberg articles.")
    return results


# ==================== MAIN CRAWLER PROCESS ====================
def main():
    print("=================== ENGLISH NEWS WEB CRAWLER ===================")
    
    # Crawl all news sources
    nyt_data = crawl_nytimes()
    wp_data = crawl_washington_post()
    wsj_data = crawl_wsj()
    ap_data = crawl_apnews()
    reuters_data = crawl_reuters()
    guardian_data = crawl_guardian()
    ft_data = crawl_ft()
    cnn_data = crawl_cnn()
    bloomberg_data = crawl_bloomberg()
    
    # Combine all data
    all_data = (nyt_data + wp_data + wsj_data + ap_data + reuters_data + 
                guardian_data + ft_data + cnn_data + bloomberg_data)
    
    # Save to file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)
    
    print(f"\n=================== SCRAPING COMPLETE ===================")
    print(f"Total entries collected: {len(all_data)}")
    print(f"  - NYT: {len(nyt_data)}")
    print(f"  - Washington Post: {len(wp_data)}")
    print(f"  - WSJ: {len(wsj_data)}")
    print(f"  - AP News: {len(ap_data)}")
    print(f"  - Reuters: {len(reuters_data)}")
    print(f"  - The Guardian: {len(guardian_data)}")
    print(f"  - Financial Times: {len(ft_data)}")
    print(f"  - CNN: {len(cnn_data)}")
    print(f"  - Bloomberg: {len(bloomberg_data)}")
    print(f"Saved dataset to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

import requests
from bs4 import BeautifulSoup
import json
import time
import re
import os
import random
import urllib3

urllib3.disable_warnings()

SITEMAP_INDEX_URL = "https://tingia.gov.vn/sitemap.xml"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, '../data/vie/raw/vafc_sitemap_output.json')
TARGET_SITEMAPS = [
    "https://tingia.gov.vn/sitemap/tin-vua-check.xml",
    "https://tingia.gov.vn/sitemap/cong-bo-tin-gia.xml",
    "https://tingia.gov.vn/sitemap/vaccine-phong-chong-tin-gia.xml",
    "https://tingia.gov.vn/sitemap/linh-vuc.xml"
]

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def get_detail_urls():
    print("Scanning VAFC sub-sitemaps for detail URLs...")
    all_urls = []
    
    for sm_url in TARGET_SITEMAPS:
        print(f"Fetching sitemap: {sm_url}")
        try:
            r = requests.get(sm_url, headers=headers, timeout=15, verify=False)
            if r.status_code == 200:
                urls = re.findall(r'<loc>(https://tingia.gov.vn/[^<]+\.html)</loc>', r.text)
                print(f"  -> Found {len(urls)} HTML URLs")
                all_urls.extend(urls)
            else:
                print(f"  -> Failed to fetch. Status code: {r.status_code}")
        except Exception as e:
            print(f"  [ERROR] Fetching sitemap {sm_url}: {e}")
            
    # Deduplicate
    unique_urls = list(set(all_urls))
    print(f"Total unique detail URLs found: {len(unique_urls)}")
    return unique_urls

def crawl_details(urls, max_limit=180):
    # Shuffle URLs to get a balanced representation across different categories
    random.seed(42)
    random.shuffle(urls)
    
    selected_urls = urls[:max_limit]
    print(f"Selected {len(selected_urls)} URLs to crawl.")
    
    results = []
    success_count = 0
    
    for i, url in enumerate(selected_urls):
        print(f"[{i+1}/{len(selected_urls)}] Crawling: {url}")
        try:
            time.sleep(0.5) # Polite delay
            r = requests.get(url, headers=headers, timeout=15, verify=False)
            if r.status_code != 200:
                print(f"  -> Failed to load page. Status: {r.status_code}")
                continue
                
            soup = BeautifulSoup(r.content, "html.parser")
            
            title = soup.find("h1")
            title_text = title.text.strip() if title else ""
            
            sapo = soup.find("div", class_="content-detail-sapo")
            sapo_text = sapo.text.strip() if sapo else ""
            
            content_div = soup.find("div", id="maincontent") or soup.find("div", class_="entry-content")
            content_text = content_div.text.strip() if content_div else ""
            
            date_div = soup.find("div", class_="post-meta")
            date_str = date_div.text.strip() if date_div else ""
            
            if not title_text or not content_text:
                print("  -> Empty title or content, skipping.")
                continue
                
            results.append({
                "source_type": "website",
                "source_name": "VAFC (tingia.gov.vn)",
                "url": url,
                "claim": title_text,
                "original_text": sapo_text,
                "justification": content_text,
                "publish_date": date_str
            })
            success_count += 1
            
        except Exception as e:
            print(f"  [ERROR] Crawling {url}: {e}")
            
    print(f"Successfully crawled {success_count} VAFC detail pages.")
    return results

def main():
    urls = get_detail_urls()
    if not urls:
        print("No URLs found. Exiting.")
        return
        
    crawled_data = crawl_details(urls, max_limit=180)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(crawled_data, f, ensure_ascii=False, indent=4)
        
    print(f"Saved crawled data to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

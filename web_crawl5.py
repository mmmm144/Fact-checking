import requests
from bs4 import BeautifulSoup
import json
import time
import re
import os
from datetime import datetime
import urllib3

urllib3.disable_warnings()

COOKIE_FILE = "facebook_cookies.json"
OUTPUT_FILE = "news_website_output.json"

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Helper: Parse date from dd/mm/yyyy
def parse_date(date_str):
    try:
        # Extract dd/mm/yyyy
        match = re.search(r'(\d{1,2})/(\d{1,2})/(\d{4})', date_str)
        if match:
            day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
            return datetime(year, month, day)
    except:
        pass
    return None

# ==================== 1. VAFC (tingia.gov.vn) SCRAPER ====================
def crawl_vafc():
    print("\n--- Starting VAFC (tingia.gov.vn) Scraper ---")
    categories = {
        "Tài chính - Ngân hàng": "https://tingia.gov.vn/tai-chinh-ngan-hang",
        "Sức khỏe cộng đồng": "https://tingia.gov.vn/suc-khoe-cong-dong",
        "Quyền lợi người dân": "https://tingia.gov.vn/quyen-loi-nguoi-dan",
        "Vaccine tin giả": "https://tingia.gov.vn/vaccine-phong-chong-tin-gia"
    }
    
    article_links = {}
    for cat_name, cat_url in categories.items():
        print(f"Fetching VAFC category: {cat_name} ({cat_url})")
        try:
            r = requests.get(cat_url, headers=headers, timeout=15, verify=False)
            soup = BeautifulSoup(r.content, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a['href']
                # Clean and ensure absolute URL
                if href.startswith("/"):
                    href = "https://tingia.gov.vn" + href
                if href.startswith("https://tingia.gov.vn/") and href.endswith(".html") and "multimedia" not in href:
                    article_links[href] = cat_name
        except Exception as e:
            print(f"  [ERROR] {cat_url}: {e}")
            
    print(f"Found {len(article_links)} unique VAFC articles. Crawling detail pages...")
    results = []
    
    for url, cat_name in article_links.items():
        print(f"  Crawling VAFC detail: {url}")
        try:
            time.sleep(1) # politeness delay
            r = requests.get(url, headers=headers, timeout=15, verify=False)
            soup = BeautifulSoup(r.content, "html.parser")
            
            title = soup.find("h1")
            title_text = title.text.strip() if title else ""
            
            sapo = soup.find("div", class_="content-detail-sapo")
            sapo_text = sapo.text.strip() if sapo else ""
            
            content_div = soup.find("div", id="maincontent") or soup.find("div", class_="entry-content")
            justification = content_div.text.strip() if content_div else ""
            
            # Extract date
            date_div = soup.find("div", class_="post-meta")
            date_str = ""
            if date_div:
                date_str = date_div.text.strip()
                # e.g., "20/03/2024 - 18:13"
                dt = parse_date(date_str)
                if dt and dt.year < 2023:
                    print(f"    -> Skipped due to year {dt.year} < 2023")
                    continue
            
            # Extract evidence (commonly mentioned entities/agencies or the source)
            evidence = "Trung tâm xử lý tin giả Việt Nam (VAFC)"
            agencies = ["Bộ Y tế", "Cục An toàn thông tin", "Bộ Công an", "Công an", "Bộ Thông tin và Truyền thông", "NCSC"]
            mentioned = [agency for agency in agencies if agency in justification]
            if mentioned:
                evidence += f" dựa trên công bố từ " + ", ".join(mentioned)
                
            results.append({
                "source_type": "website",
                "source_name": "VAFC (tingia.gov.vn)",
                "url": url,
                "domain": cat_name,
                "label": "FALSE",
                "claim": title_text,
                "original_text": sapo_text,
                "evidence": evidence,
                "justification": justification,
                "publish_date": date_str
            })
            
        except Exception as e:
            print(f"    [ERROR] Detail {url}: {e}")
            
    print(f"Successfully crawled {len(results)} VAFC articles.")
    return results


# ==================== 2. NCSC (tinnhiemmang.vn) SCRAPER ====================
def crawl_ncsc_blacklist():
    print("\n--- Starting NCSC Blacklist Scraper ---")
    results = []
    
    # 2a. Website lừa đảo (website-lua-dao) - Crawl first 5 pages for historical entries
    for page in range(1, 6):
        url = f"https://tinnhiemmang.vn/website-lua-dao?page={page}"
        print(f"Fetching NCSC Blacklist page {page} ({url})")
        try:
            time.sleep(1)
            r = requests.get(url, headers=headers, timeout=15, verify=False)
            soup = BeautifulSoup(r.content, "html.parser")
            objs = soup.find_all("div", class_="obj")
            
            if not objs:
                break
                
            for o in objs:
                span_url = o.find("span", class_="webkit-box-2")
                fake_url = span_url.text.strip() if span_url else ""
                
                date_div = o.find("div", class_="date")
                date_str = date_div.text.strip() if date_div else ""
                
                # Check year
                # e.g., "Đã phát hiện ngày 18/02/2025"
                dt = parse_date(date_str)
                if dt and dt.year < 2023:
                    print(f"    -> Skipping old entries from year {dt.year}")
                    continue
                    
                org_link = o.find("div", class_="code").find("a") if o.find("div", class_="code") else None
                org_name = "Chưa xác định"
                if org_link:
                    # Remove "Mạo danh tổ chức:" label
                    org_name = org_link.text.replace("Mạo danh tổ chức:", "").strip()
                    
                claim = f"Website {fake_url} mạo danh tổ chức {org_name}"
                original_text = f"Phát hiện trang web giả mạo/lừa đảo trực tuyến: {fake_url} nhằm mạo danh thương hiệu, cơ quan {org_name} để đánh cắp thông tin hoặc chiếm đoạt tài sản người dùng."
                evidence = "Trung tâm Giám sát an toàn không gian mạng quốc gia (NCSC) dán nhãn cảnh báo lừa đảo."
                justification = f"Theo ghi nhận của NCSC, tên miền {fake_url} đang thực hiện hành vi giả mạo trang tin hoặc cổng dịch vụ của {org_name} vào ngày {date_str}. Trang web này có dấu hiệu lừa đảo trực tuyến, thu thập thông tin nhạy cảm của khách hàng hoặc phát tán mã độc. Người dân tuyệt đối không truy cập hoặc giao dịch trên trang này."
                
                results.append({
                    "source_type": "website",
                    "source_name": "NCSC (tinnhiemmang.vn)",
                    "url": "https://tinnhiemmang.vn/website-lua-dao",
                    "domain": "An toàn thông tin / Lừa đảo mạng",
                    "label": "FALSE",
                    "claim": claim,
                    "original_text": original_text,
                    "evidence": evidence,
                    "justification": justification,
                    "publish_date": date_str
                })
        except Exception as e:
            print(f"  [ERROR] NCSC Blacklist page {page}: {e}")
            
    # 2b. Chiến dịch cảnh báo (canh-bao-lua-dao) - Crawl first 3 pages
    warning_links = {}
    for page in range(1, 4):
        url = f"https://tinnhiemmang.vn/canh-bao-lua-dao?page={page}"
        print(f"Fetching NCSC Warnings page {page} ({url})")
        try:
            time.sleep(1)
            r = requests.get(url, headers=headers, timeout=15, verify=False)
            soup = BeautifulSoup(r.content, "html.parser")
            
            # Find posts
            posts = soup.find_all("div", class_=lambda x: x and ('post-vertical' in x or 'post-horizontal' in x))
            # Alternative: find h2 tags inside col-lg-*
            h2_tags = soup.find_all("h2", class_=lambda x: x and 'sf-bold' in x)
            
            for h2 in h2_tags:
                a_tag = h2.find_parent("a")
                if a_tag and a_tag.get("href"):
                    href = a_tag["href"]
                    if href.startswith("/"):
                        href = "https://tinnhiemmang.vn" + href
                    warning_links[href] = h2.text.strip()
        except Exception as e:
            print(f"  [ERROR] NCSC Warnings page {page}: {e}")
            
    print(f"Found {len(warning_links)} NCSC warnings. Crawling detail pages...")
    for url, title_text in warning_links.items():
        print(f"  Crawling NCSC warning: {url}")
        try:
            time.sleep(1)
            r = requests.get(url, headers=headers, timeout=15, verify=False)
            soup = BeautifulSoup(r.content, "html.parser")
            
            # Extract content
            post_content = soup.find("div", class_="post-content") or soup.find("div", class_="detail-content")
            if not post_content:
                post_content = soup.find("div", class_="col-md-8")
                
            justification = post_content.text.strip() if post_content else ""
            
            # Extract summary
            p_desc = soup.find("p", class_="webkit-box-2")
            original_text = p_desc.text.strip() if p_desc else title_text
            
            date_div = soup.find("div", class_="date-create") or soup.find("div", class_="date")
            date_str = ""
            if date_div:
                date_str = date_div.text.strip()
                dt = parse_date(date_str)
                if dt and dt.year < 2023:
                    print(f"    -> Skipped due to year {dt.year} < 2023")
                    continue
                    
            results.append({
                "source_type": "website",
                "source_name": "NCSC (tinnhiemmang.vn)",
                "url": url,
                "domain": "Cảnh báo bảo mật",
                "label": "FALSE",
                "claim": title_text,
                "original_text": original_text,
                "evidence": "Trung tâm Giám sát an toàn không gian mạng quốc gia (NCSC)",
                "justification": justification,
                "publish_date": date_str
            })
        except Exception as e:
            print(f"    [ERROR] Warning detail {url}: {e}")
            
    print(f"Successfully crawled {len(results)} NCSC blacklist/warning entries.")
    return results


# ==================== 3. TRUE NEWS SCRAPER (VnExpress, Tuoi Tre, Nhan Dan) ====================
def crawl_true_news():
    print("\n--- Starting Mainstream News Scraper (VnExpress, Tuoi Tre, Nhan Dan) ---")
    
    # Target urls mapped to Category
    targets = [
        # VnExpress
        {"source": "VnExpress", "url": "https://vnexpress.net/suc-khoe", "domain": "Y tế / Sức khỏe"},
        {"source": "VnExpress", "url": "https://vnexpress.net/kinh-doanh", "domain": "Kinh tế / Tài chính"},
        {"source": "VnExpress", "url": "https://vnexpress.net/giao-duc", "domain": "Giáo dục / Khoa học"},
        # Tuoi Tre
        {"source": "Tuổi Trẻ", "url": "https://tuoitre.vn/suc-khoe.htm", "domain": "Y tế / Sức khỏe"},
        {"source": "Tuổi Trẻ", "url": "https://tuoitre.vn/kinh-doanh.htm", "domain": "Kinh tế / Tài chính"},
        {"source": "Tuổi Trẻ", "url": "https://tuoitre.vn/giao-duc.htm", "domain": "Giáo dục / Khoa học"},
        # Nhan Dan
        {"source": "Nhân Dân", "url": "https://nhandan.vn/y-te", "domain": "Y tế / Sức khỏe"},
        {"source": "Nhân Dân", "url": "https://nhandan.vn/kinh-te", "domain": "Kinh tế / Tài chính"},
        {"source": "Nhân Dân", "url": "https://nhandan.vn/giao-duc", "domain": "Giáo dục / Khoa học"}
    ]
    
    article_links = []
    
    for t in targets:
        print(f"Scanning category: {t['source']} - {t['domain']} ({t['url']})")
        try:
            r = requests.get(t['url'], headers=headers, timeout=15, verify=False)
            soup = BeautifulSoup(r.content, "html.parser")
            
            # Find links
            links = soup.find_all("a", href=True)
            count = 0
            for a in links:
                href = a['href']
                
                # Normalize link
                if href.startswith("/"):
                    if t['source'] == "VnExpress":
                        href = "https://vnexpress.net" + href
                    elif t['source'] == "Tuổi Trẻ":
                        href = "https://tuoitre.vn" + href
                    elif t['source'] == "Nhân Dân":
                        href = "https://nhandan.vn" + href
                
                # Check validity for detail pages
                is_valid = False
                if t['source'] == "VnExpress" and "vnexpress.net/" in href and href.endswith(".html"):
                    is_valid = True
                elif t['source'] == "Tuổi Trẻ" and "tuoitre.vn/" in href and href.endswith(".htm"):
                    is_valid = True
                elif t['source'] == "Nhân Dân" and "nhandan.vn/" in href and not href.endswith(t['url'].split('/')[-1]) and len(href.split('/')) > 4:
                    is_valid = True
                    
                if is_valid and count < 8: # Limit to 8 links per category scan to manage crawl load
                    article_links.append({
                        "source_name": t['source'],
                        "url": href,
                        "domain": t['domain']
                    })
                    count += 1
        except Exception as e:
            print(f"  [ERROR] Scanning category {t['url']}: {e}")
            
    # Deduplicate detail links
    seen = set()
    unique_links = []
    for link in article_links:
        if link['url'] not in seen:
            seen.add(link['url'])
            unique_links.append(link)
            
    print(f"Found {len(unique_links)} unique news articles. Crawling content...")
    results = []
    
    for item in unique_links:
        url = item['url']
        src = item['source_name']
        domain = item['domain']
        print(f"  Crawling News: {src} -> {url}")
        
        try:
            time.sleep(1)
            r = requests.get(url, headers=headers, timeout=15, verify=False)
            soup = BeautifulSoup(r.content, "html.parser")
            
            claim = ""
            original_text = ""
            justification = ""
            publish_date = ""
            
            if src == "VnExpress":
                h1 = soup.find("h1", class_="title-detail")
                claim = h1.text.strip() if h1 else ""
                
                p_desc = soup.find("p", class_="description")
                original_text = p_desc.text.strip() if p_desc else ""
                
                # Content paragraphs
                paragraphs = soup.find_all("p", class_="normal")
                justification = "\n".join([p.text.strip() for p in paragraphs if p.text.strip()])
                
                # Date
                date_span = soup.find("span", class_="date")
                publish_date = date_span.text.strip() if date_span else ""
                
            elif src == "Tuổi Trẻ":
                h1 = soup.find("h1", class_="article-title") or soup.find("h1", class_="detail-title")
                claim = h1.text.strip() if h1 else ""
                
                sapo = soup.find("h2", class_="sapo") or soup.find("div", class_="detail-sapo")
                original_text = sapo.text.strip() if sapo else ""
                
                # Content
                body = soup.find("div", id="main-detail-body") or soup.find("div", class_="detail-content")
                if body:
                    justification = "\n".join([p.text.strip() for p in body.find_all("p") if p.text.strip()])
                    
                # Date
                date_div = soup.find("div", class_="date-time") or soup.find("div", class_="article-date")
                publish_date = date_div.text.strip() if date_div else ""
                
            elif src == "Nhân Dân":
                h1 = soup.find("h1", class_="title") or soup.find("h1", class_="detail-title")
                claim = h1.text.strip() if h1 else ""
                
                sapo = soup.find("div", class_="detail-sapo")
                original_text = sapo.text.strip() if sapo else ""
                
                # Content
                body = soup.find("div", class_="detail-content") or soup.find("div", class_="post-content")
                if body:
                    justification = "\n".join([p.text.strip() for p in body.find_all("p") if p.text.strip()])
                    
                # Date
                date_div = soup.find("div", class_="box-date-top") or soup.find("div", class_="date")
                publish_date = date_div.text.strip() if date_div else ""
                
            # Date checks: filter out old years
            dt = parse_date(publish_date)
            if dt and dt.year < 2023:
                print(f"    -> Skipped due to year {dt.year} < 2023")
                continue
                
            results.append({
                "source_type": "website",
                "source_name": src,
                "url": url,
                "domain": domain,
                "label": "TRUE",
                "claim": claim,
                "original_text": original_text,
                "evidence": f"Bài viết đăng tải trên báo chính thống {src}",
                "justification": justification,
                "publish_date": publish_date
            })
            
        except Exception as e:
            print(f"    [ERROR] News detail {url}: {e}")
            
    print(f"Successfully crawled {len(results)} TRUE news articles.")
    return results


# ==================== MAIN CRAWLER PROCESS ====================
def main():
    print("=================== FACT-CHECKING DATA WEB CRAWLER ===================")
    
    # 1. Crawl VAFC (FALSE claims & Debunks)
    vafc_data = crawl_vafc()
    
    # 2. Crawl NCSC Blacklist (FALSE campaigns & Scam websites)
    ncsc_data = crawl_ncsc_blacklist()
    
    # 3. Crawl True News (TRUE baseline)
    news_data = crawl_true_news()
    
    # Combine all crawled data
    all_data = vafc_data + ncsc_data + news_data
    
    # Save to file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)
        
    print(f"\n=================== SCRAPING COMPLETE ===================")
    print(f"Total entries collected: {len(all_data)}")
    print(f"  - FALSE from VAFC: {len(vafc_data)}")
    print(f"  - FALSE from NCSC: {len(ncsc_data)}")
    print(f"  - TRUE from Mainstream News: {len(news_data)}")
    print(f"Saved dataset to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

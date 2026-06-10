import json
import os
import re
import hashlib
import random
import sys

# Setup imports relative to script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from utils import clean_text, parse_vietnamese_date

FACEBOOK_OUTPUT = os.path.join(SCRIPT_DIR, "../data/vie/raw/output.json")
WEBSITE_OUTPUT = os.path.join(SCRIPT_DIR, "../data/vie/raw/news_website_output.json")
SITEMAP_OUTPUT = os.path.join(SCRIPT_DIR, "../data/vie/raw/vafc_sitemap_output.json")
FINAL_DATASET = os.path.join(SCRIPT_DIR, "../data/vie/processed/fact_checking_dataset.json")

def generate_hash_id(content_str):
    return hashlib.md5(content_str.encode('utf-8')).hexdigest()

def extract_facebook_claim(content):
    if not content:
        return ""
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    if not lines:
        return ""
    
    first_line = lines[0]
    
    if len(first_line) < 15 and len(lines) > 1:
        combined = f"{first_line} {lines[1]}"
        if len(combined) <= 150:
            return combined
        return combined[:147] + "..."
        
    if len(first_line) > 150:
        first_sentence = first_line.split('. ')[0]
        if len(first_sentence) <= 150 and len(first_sentence) > 15:
            return first_sentence + "."
        return first_line[:147] + "..."
        
    return first_line

def extract_vafc_rumor(title):
    """
    Rewrite VAFC warning titles into rumor claims (FALSE statements).
    E.g. "Thông tin sai sự thật về việc A" -> "A"
    """
    if not title:
        return ""
    
    orig = title
    
    # 1. Strip common warning prefixes
    title = re.sub(r'^[Cc]ảnh báo:?\s*', '', title)
    title = re.sub(r'^[Tt]hông tin (sai sự thật|xuyên tạc|đồn thổi) về (việc\s+)?', '', title, flags=re.IGNORECASE)
    title = re.sub(r'^[Tt]in đồn về (việc\s+)?', '', title, flags=re.IGNORECASE)
    title = re.sub(r'^[Tt]in giả về (việc\s+)?', '', title, flags=re.IGNORECASE)
    
    title = title.strip()
    if title:
        # Capitalize first letter
        title = title[0].upper() + title[1:]
        
    if len(title) < 10 or title == orig:
        # Fallback template if no prefix matched or it is too short
        return f"Tin đồn: {orig}"
        
    return title

def generate_ncsc_synthetic(ncsc_entries, count=75):
    """
    Generate synthetic FALSE claims from NCSC warnings.
    E.g. "Website tkmmi.com mạo danh tổ chức Tik tok shop"
    -> "Website tkmmi.com là trang web chính thức, an toàn của Tik tok shop."
    """
    synthetic_entries = []
    
    # Filter only NCSC entries
    ncsc_only = [e for e in ncsc_entries if "tinnhiemmang.vn" in e.get("url", "")]
    if not ncsc_only:
        return []
        
    random.seed(42)
    selected = random.sample(ncsc_only, min(count, len(ncsc_only)))
    
    templates = [
        "Website {fake_url} là trang web chính thức và an toàn của {org_name}.",
        "Bạn có thể truy cập {fake_url} để thực hiện các dịch vụ và thanh toán của {org_name}.",
        "Cổng thông tin giao dịch mới của {org_name} đã hoạt động tại địa chỉ {fake_url}.",
        "Đăng nhập tài khoản {org_name} của bạn tại link {fake_url} để nhận ưu đãi."
    ]
    
    for idx, entry in enumerate(selected):
        text = entry.get("original_text", "")
        fake_url = "unknown.com"
        org_name = "Chưa xác định"
        
        match_url = re.search(r'trực tuyến:\s*([a-zA-Z0-9.-]+\.[a-zA-Z]+)', text)
        if match_url:
            fake_url = match_url.group(1).strip()
            
        match_org = re.search(r'mạo danh thương hiệu, cơ quan\s+([a-zA-Z0-9.\s_]+)\s+để', text)
        if match_org:
            org_name = match_org.group(1).strip()
        else:
            title = entry.get("claim", "")
            match_title_org = re.search(r'mạo danh tổ chức\s+(.+)$', title)
            if match_title_org:
                org_name = match_title_org.group(1).strip()
                
        template = templates[idx % len(templates)]
        claim = template.format(fake_url=fake_url, org_name=org_name)
        
        unique_str = f"{entry.get('url')}_synthetic_{claim}"
        
        synthetic_entries.append({
            "id": generate_hash_id(unique_str),
            "source_type": "website",
            "source_name": "NCSC (tinnhiemmang.vn) - Synthetic",
            "url": entry.get("url", ""),
            "domain": "An toàn thông tin / Tin giả nhân tạo",
            "publish_date": parse_vietnamese_date(entry.get("publish_date")),
            "claim": claim,
            "original_text": entry.get("original_text", ""),
            "label": "FALSE",
            "evidence": f"Cảnh báo chính thức từ NCSC: Website {fake_url} là giả mạo {org_name}.",
            "justification": entry.get("justification", ""),
            "comments": []
        })
        
    return synthetic_entries

def main():
    print("=================== DATASET COMPILATION & GENERATION ===================")
    
    final_entries = []
    
    # ------------------ SOURCE 1: Web Crawl Output (TRUE Warnings & News) ------------------
    if os.path.exists(WEBSITE_OUTPUT):
        with open(WEBSITE_OUTPUT, "r", encoding="utf-8") as f:
            web_entries = json.load(f)
            
        web_true_count = 0
        for entry in web_entries:
            claim_val = clean_text(entry.get("claim", ""))
            original_text_val = clean_text(entry.get("original_text", ""))
            
            if not claim_val or not original_text_val:
                continue
                
            raw_date = entry.get("publish_date", "")
            std_date = parse_vietnamese_date(raw_date)
            if not std_date and raw_date:
                std_date = clean_text(raw_date).split('\n')[0]
                
            unique_str = f"{entry.get('url', '')}_{claim_val}"
            
            final_entries.append({
                "id": generate_hash_id(unique_str),
                "source_type": "website",
                "source_name": entry.get("source_name", ""),
                "url": entry.get("url", ""),
                "domain": entry.get("domain", ""),
                "publish_date": std_date,
                "claim": claim_val,
                "original_text": original_text_val,
                "label": "TRUE",
                "evidence": clean_text(entry.get("evidence", "")),
                "justification": clean_text(entry.get("justification", "")),
                "comments": []
            })
            web_true_count += 1
        print(f"Loaded {web_true_count} TRUE website warning entries.")
    else:
        print(f"Warning: {WEBSITE_OUTPUT} not found.")

    # ------------------ SOURCE 2: Facebook Output (TRUE News) ------------------
    if os.path.exists(FACEBOOK_OUTPUT):
        with open(FACEBOOK_OUTPUT, "r", encoding="utf-8") as f:
            fb_entries = json.load(f)
            
        fb_true_count = 0
        for entry in fb_entries:
            author = clean_text(entry.get("author", ""))
            content = clean_text(entry.get("content", ""))
            url = entry.get("url", "").strip()
            date = entry.get("date", "")
            media = entry.get("media", [])
            comments = [clean_text(c) for c in entry.get("comments", []) if clean_text(c)]
            
            if not content and not comments:
                continue
                
            claim = extract_facebook_claim(content)
            if not claim and comments:
                claim = extract_facebook_claim(comments[0])
                
            std_date = parse_vietnamese_date(date)
            if not std_date and date:
                std_date = clean_text(date)
                
            unique_str = f"{url}_{claim}"
            
            final_entries.append({
                "id": generate_hash_id(unique_str),
                "source_type": "facebook",
                "source_name": author,
                "url": url,
                "domain": "Mạng xã hội / Tin tức cộng đồng",
                "publish_date": std_date,
                "claim": claim,
                "original_text": content,
                "label": "TRUE",
                "evidence": f"Xác thực bởi trang chính thống: {author}",
                "justification": "\n".join(comments) if comments else "Đăng tải bởi cơ quan thông tấn chính thức.",
                "comments": comments,
                "media_links": media
            })
            fb_true_count += 1
        print(f"Loaded {fb_true_count} TRUE Facebook news entries.")
    else:
        print(f"Warning: {FACEBOOK_OUTPUT} not found.")

    # ------------------ SOURCE 3: Sitemap VAFC output (TRUE warnings & FALSE rumors) ------------------
    if os.path.exists(SITEMAP_OUTPUT):
        with open(SITEMAP_OUTPUT, "r", encoding="utf-8") as f:
            sitemap_entries = json.load(f)
            
        sitemap_true_count = 0
        sitemap_false_count = 0
        
        for entry in sitemap_entries:
            title = clean_text(entry.get("claim", ""))
            original_text_val = clean_text(entry.get("original_text", ""))
            justification_val = clean_text(entry.get("justification", ""))
            url = entry.get("url", "")
            raw_date = entry.get("publish_date", "")
            std_date = parse_vietnamese_date(raw_date)
            if not std_date and raw_date:
                std_date = clean_text(raw_date).split('\n')[0]
                
            if not title or not justification_val:
                continue
                
            unique_str_true = f"{url}_warning_{title}"
            final_entries.append({
                "id": generate_hash_id(unique_str_true),
                "source_type": "website",
                "source_name": "VAFC (tingia.gov.vn)",
                "url": url,
                "domain": "Cảnh báo tin giả sitemap",
                "publish_date": std_date,
                "claim": title,
                "original_text": original_text_val if original_text_val else title,
                "label": "TRUE",
                "evidence": "Trung tâm xử lý tin giả Việt Nam (VAFC)",
                "justification": justification_val,
                "comments": []
            })
            sitemap_true_count += 1
            
            rumor_claim = extract_vafc_rumor(title)
            unique_str_false = f"{url}_rumor_{rumor_claim}"
            final_entries.append({
                "id": generate_hash_id(unique_str_false),
                "source_type": "website",
                "source_name": "Tin đồn / Tin giả sitemap",
                "url": url,
                "domain": "Tin đồn mạng sitemap",
                "publish_date": std_date,
                "claim": rumor_claim,
                "original_text": original_text_val if original_text_val else rumor_claim,
                "label": "FALSE",
                "evidence": f"Xác minh bác bỏ từ VAFC: {title}",
                "justification": f"Bản tin xác thực của VAFC đã kết luận nội dung này là tin giả/tin sai sự thật. Chi tiết phản bác: {justification_val}",
                "comments": []
            })
            sitemap_false_count += 1
            
        print(f"Loaded {sitemap_true_count} TRUE sitemap warnings and {sitemap_false_count} FALSE sitemap rumors.")
    else:
        print(f"Warning: {SITEMAP_OUTPUT} not found. Crawl VAFC sitemap first.")

    # ------------------ SOURCE 4: NCSC Blacklist (Synthetic FALSE Claims) ------------------
    if os.path.exists(WEBSITE_OUTPUT):
        with open(WEBSITE_OUTPUT, "r", encoding="utf-8") as f:
            web_entries = json.load(f)
            
        synthetic_entries = generate_ncsc_synthetic(web_entries, count=75)
        final_entries.extend(synthetic_entries)
        print(f"Generated {len(synthetic_entries)} NCSC synthetic FALSE claims.")
    else:
        print(f"Warning: {WEBSITE_OUTPUT} not found, skipping NCSC synthetic generation.")

    # Save final dataset
    # Ensure processed directory exists
    os.makedirs(os.path.dirname(FINAL_DATASET), exist_ok=True)
    with open(FINAL_DATASET, "w", encoding="utf-8") as f:
        json.dump(final_entries, f, ensure_ascii=False, indent=4)
        
    print(f"\n=================== COMPILATION COMPLETE ===================")
    print(f"Total entries in final dataset: {len(final_entries)}")
    print(f"Saved merged dataset to: {FINAL_DATASET}")

if __name__ == "__main__":
    main()

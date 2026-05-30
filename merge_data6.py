import json
import os
import hashlib

FACEBOOK_OUTPUT = "output.json"
WEBSITE_OUTPUT = "news_website_output.json"
FINAL_DATASET = "fact_checking_dataset.json"

def generate_hash_id(content_str):
    return hashlib.md5(content_str.encode('utf-8')).hexdigest()

def merge_datasets():
    print("=================== DATA MERGE & STANDARDIZATION ===================")
    
    combined_data = []
    
    # 1. Process Website Output
    if os.path.exists(WEBSITE_OUTPUT):
        with open(WEBSITE_OUTPUT, "r", encoding="utf-8") as f:
            web_entries = json.load(f)
        print(f"Loaded {len(web_entries)} entries from Website Scraper ({WEBSITE_OUTPUT})")
        
        for entry in web_entries:
            # Generate a unique hash id based on URL and Claim
            claim_val = entry.get("claim", "")
            unique_str = f"{entry.get('url', '')}_{claim_val}"
            
            combined_data.append({
                "id": generate_hash_id(unique_str),
                "source_type": entry.get("source_type", "website"),
                "source_name": entry.get("source_name", ""),
                "url": entry.get("url", ""),
                "domain": entry.get("domain", ""),
                "original_text": entry.get("original_text", "").strip(),
                "publish_date": entry.get("publish_date", ""),
                "comments": []
            })
    else:
        print(f"Warning: {WEBSITE_OUTPUT} not found. Skipping website data.")
        
    # 2. Process Facebook Output
    if os.path.exists(FACEBOOK_OUTPUT):
        with open(FACEBOOK_OUTPUT, "r", encoding="utf-8") as f:
            fb_entries = json.load(f)
        print(f"Loaded {len(fb_entries)} entries from Facebook Crawler ({FACEBOOK_OUTPUT})")
        
        for entry in fb_entries:
            author = entry.get("author", "").strip()
            content = entry.get("content", "").strip()
            url = entry.get("url", "").strip()
            date = entry.get("date", "").strip()
            media = entry.get("media", [])
            comments = entry.get("comments", [])
            
            if not content and not comments:
                continue # skip empty posts
                
            # Formulate standard fields
            claim = content.split("\n")[0] if content else ""
            if len(claim) > 150 or not claim:
                claim = content[:120] + "..." if len(content) > 120 else content
                
            unique_str = f"{url}_{claim}"
            
            combined_data.append({
                "id": generate_hash_id(unique_str),
                "source_type": "facebook",
                "source_name": author,
                "url": url,
                "domain": "Mạng xã hội / Tin tức cộng đồng",
                "original_text": content.strip(),
                "publish_date": date,
                "media_links": media,
                "comments": comments
            })
    else:
        print(f"Warning: {FACEBOOK_OUTPUT} not found. Skipping facebook data.")
        
    # Save the combined dataset
    with open(FINAL_DATASET, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)
        
    print(f"\n=================== MERGE COMPLETE ===================")
    print(f"Total unified entries in final dataset: {len(combined_data)}")
    print(f"Saved merged dataset to: {FINAL_DATASET}")

if __name__ == "__main__":
    merge_datasets()

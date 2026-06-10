import json
import time
import re
import os
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC



import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COOKIE_FILE = os.path.join(SCRIPT_DIR, '../data/vie/raw/facebook_cookies.json')
INPUT_FILE = os.path.join(SCRIPT_DIR, '../data/vie/raw/output.json')
OUTPUT_FILE = os.path.join(SCRIPT_DIR, '../data/vie/raw/output.json')

def format_fb_date(ts):
    dt = datetime.fromtimestamp(ts)
    weekdays = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
    dow = weekdays[dt.weekday()]
    return f"{dow}, {dt.day} Tháng {dt.month}, {dt.year} lúc {dt.strftime('%H:%M')}"

def load_cookies(driver):
    if not os.path.exists(COOKIE_FILE):
        print(f"Warning: {COOKIE_FILE} not found. Skipping cookie loading.")
        return
    
    with open(COOKIE_FILE, "r") as f:
        cookies = json.load(f)
    for cookie in cookies:
        cookie.pop("sameSite", None)
        try:
            driver.add_cookie(cookie)
        except:
            pass

def init_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-notifications")
    options.add_argument("--start-maximized")
    return webdriver.Chrome(options=options)

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Find posts that need re-crawling
    posts_to_fix = []
    for i, post in enumerate(data):
        date_str = post.get("date", "").strip()
        # Find posts with empty date or invalid date format
        if not date_str or not re.match(r".+\d{4}", date_str):
            # Only process posts that have a URL (not empty)
            if post.get("url"):
                posts_to_fix.append((i, post))

    if not posts_to_fix:
        print("Không có bài viết nào cần cập nhật date.")
        return

    print(f"Tìm thấy {len(posts_to_fix)} bài viết cần cập nhật date.")
    
    driver = init_driver()
    driver.get("https://facebook.com")
    time.sleep(3)
    load_cookies(driver)
    driver.refresh()
    time.sleep(3)

    count_fixed = 0
    for idx, post in posts_to_fix:
        url = post["url"]
        print(f"\n[{count_fixed+1}/{len(posts_to_fix)}] Re-crawling: {url}")
        
        driver.get(url)
        time.sleep(5)
        
        src = driver.page_source
        
        # Method 1: Extract unix timestamp from page source
        # "publish_time":1718816820 or "creation_time":1718816820
        matches = re.findall(r'"(?:publish_time|creation_time)"\s*:\s*(\d{10})', src)
        if matches:
            # take the earliest timestamp (the creation time vs modification time)
            ts = int(min(matches))
            new_date = format_fb_date(ts)
            print(f"  -> Found timestamp: {ts} => {new_date}")
            data[idx]["date"] = new_date
            count_fixed += 1
            continue
            
        print("  -> Could not find timestamp via regex. Attempting tooltip fallback.")
        
        # Method 2: Tooltip fallback (hover logic)
        try:
            a_tags = driver.find_elements(By.XPATH, "//a | //span")
            date_found = False
            for a in a_tags:
                txt = a.text.strip()
                if "2023" in txt or "2024" in txt or "2025" in txt or "tháng" in txt.lower():
                    # Check if it has an aria-label that's a date
                    aria = a.get_attribute("aria-label") or ""
                    if aria and ("tháng" in aria.lower() or "lúc" in aria.lower() or "," in aria):
                        data[idx]["date"] = aria
                        print(f"  -> Found via aria-label: {aria}")
                        date_found = True
                        break
                        
                    # Trigger hover
                    driver.execute_script("arguments[0].dispatchEvent(new MouseEvent('mouseover', {'view': window, 'bubbles': true, 'cancelable': true}));", a)
                    time.sleep(1)
                    
                    tooltips = driver.find_elements(By.XPATH, "//div[@role='tooltip'] | //div[contains(@class,'tooltip')] | //span[@data-tooltip-content]")
                    for tip in tooltips:
                        tip_t = tip.text.strip()
                        if tip_t and len(tip_t) > 5 and ("tháng" in tip_t.lower() or "lúc" in tip_t.lower()):
                            data[idx]["date"] = tip_t
                            print(f"  -> Found via tooltip hover: {tip_t}")
                            date_found = True
                            break
                    if date_found:
                        break
            
            if date_found:
                count_fixed += 1
            else:
                print("  -> Could not update date.")
        except Exception as e:
            print(f"  -> Error: {e}")

    driver.quit()
    
    # Save back
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
    print(f"\nHoàn thành! Đã cập nhật {count_fixed}/{len(posts_to_fix)} posts vào {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

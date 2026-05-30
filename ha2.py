import time
import json
import os
import requests

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

COOKIE_FILE = "facebook_cookies.json"
LINK_FILE = "facebook_links.txt"
MEDIA_FOLDER = "media"

def init_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-notifications")
    options.add_argument("--start-maximized")
    return webdriver.Chrome(options=options)

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
        except Exception as e:
            pass

def load_links():
    with open(LINK_FILE, "r") as f:
        links = [line.strip() for line in f if line.strip()]
    return links

def download_file(url, filename):
    try:
        r = requests.get(url, stream=True, timeout=10)
        with open(filename, "wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)
    except Exception as e:
        print(f"  [ERROR] Downloading {url}: {e}")

def extract_post(driver, url, index):
    print(f"\n[{index}] Navigating to {url}")
    driver.get(url)
    time.sleep(5)
    
    data = {
        "id": index,
        "url": url,
        "author": "",
        "date": "",
        "content": "",
        "media": [],
        "comments": []
    }
    
    # Wait for post to load (could be a dialog popup or main div)
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//div[@role='dialog'] | //div[@role='main'] | //div[@data-pagelet='BeeperNewsFeed'] | //div[@role='article']"))
        )
    except:
        pass
    time.sleep(3)
    
    # Identify container
    try:
        dialogs = driver.find_elements(By.XPATH, "//div[@role='dialog']")
        if dialogs:
            # Find the top-most dialog that actually has text/headers (ignores invisible/empty overlay dialogs)
            container = dialogs[-1]
            for d in reversed(dialogs):
                if d.find_elements(By.XPATH, ".//strong | .//h2 | .//h3 | .//a[contains(@href, '/posts/')]"):
                    container = d
                    break
        else:
            articles = driver.find_elements(By.XPATH, "//div[@role='article']")
            container = articles[0] if articles else driver
    except:
        container = driver
        
    # 1. Author and Date
    try:
        header_els = container.find_elements(By.XPATH, ".//strong | .//h2 | .//h3")
        for el in header_els:
            txt = el.text.strip()
            if txt and len(txt) > 2 and '\n' not in txt:
                data["author"] = txt
                break
                
        # Date: Facebook hides the full date in a tooltip that appears on hover.
        # Strategy: find timestamp link (href contains the post path), hover over it, then read the tooltip.
        date_found = False
        a_tags = container.find_elements(By.XPATH, ".//a")
        
        for a in a_tags:
            href = a.get_attribute("href") or ""
            txt = a.text.strip()
            # Timestamp links have short text like "43 tuần", "1 ngày", "5 giờ"
            if not txt or len(txt) > 30 or '\n' in txt:
                continue
            if ("giờ" in txt or "phút" in txt or "năm" in txt or "tháng" in txt 
                    or "tuần" in txt or "ngày" in txt or "hrs" in txt or "mins" in txt
                    or "seconds" in txt or "giây" in txt):
                t_aria = a.get_attribute("aria-label") or ""
                # Check directly if it contains specific date formats to avoid modal close risk
                if t_aria and len(t_aria) > 8 and t_aria != txt and ("lúc" in t_aria or "tháng" in t_aria or "năm" in t_aria or "," in t_aria):
                    data["date"] = t_aria
                    date_found = True
                    break

                try:
                    # Alternative safer hover preventing page scroll layout jumps
                    driver.execute_script("arguments[0].dispatchEvent(new MouseEvent('mouseover', {'view': window, 'bubbles': true, 'cancelable': true}));", a)
                    try:
                        # Fallback for old ActionChains if JS dispatch is insufficient
                        ActionChains(driver).move_to_element(a).perform()
                    except:
                        pass
                    time.sleep(1)
                    # Try reading aria-label after hover (may be updated)
                    aria = a.get_attribute("aria-label") or ""
                    if aria and len(aria) > 5 and aria != txt:
                        data["date"] = aria
                    else:
                        # Try to find the tooltip element that appeared
                        tooltips = driver.find_elements(By.XPATH, 
                            "//div[@role='tooltip'] | //div[contains(@class,'tooltip')] | //span[@data-tooltip-content]")
                        tooltip_text = ""
                        for tip in tooltips:
                            t = tip.text.strip()
                            if t and len(t) > 5:
                                tooltip_text = t
                                break
                        if tooltip_text:
                            data["date"] = tooltip_text
                        else:
                            # Final fallback: use the relative text
                            data["date"] = txt
                    date_found = True
                    break
                except:
                    data["date"] = txt
                    date_found = True
                    break
        
        # Additional fallback: check aria-label of all timestamp-like links
        if not date_found:
            for a in a_tags:
                aria = a.get_attribute("aria-label") or ""
                if aria and ("tháng" in aria or "năm" in aria or "lúc" in aria) and len(aria) < 100:
                    data["date"] = aria
                    break

    except Exception as e:
        print(f"  [ERROR] Finding author/date: {e}")
        
    print(f"  -> Author: {data['author']}")
    print(f"  -> Date: {data['date']}")
    
    # Re-identify container in case hover Actions refreshed/closed the DOM
    try:
        dialogs = driver.find_elements(By.XPATH, "//div[@role='dialog']")
        if dialogs:
            container = dialogs[-1]
        else:
            articles = driver.find_elements(By.XPATH, "//div[@role='article']")
            container = articles[0] if articles else driver
    except:
        pass

    # 1.5 Content
    # First, try to click "Xem thêm" or "See more"
    try:
        see_more_btns = container.find_elements(By.XPATH, ".//div[@role='button' and (contains(text(), 'Xem thêm') or contains(text(), 'See more'))]")
        for btn in see_more_btns:
            try:
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", btn)
                time.sleep(1)
                btn.click()
                time.sleep(1)
            except:
                pass
    except Exception as e:
        print(f"  [ERROR] Clicking See more: {e}")

    try:
        # Strictly get the post message body to avoid comments
        content_els = container.find_elements(By.XPATH, ".//div[@data-ad-comet-preview='message']")
        
        # If the standard comet preview message isn't found, fall back to the main text area underneath the header
        # In photo/video viewers, the post text is often a simple div with dir='auto' before any comments
        if not content_els:
             content_els = container.find_elements(By.XPATH, ".//div[@dir='auto' and not(ancestor::ul) and not(ancestor::div[contains(@aria-label, 'Comment') or contains(@aria-label, 'Bình luận')])]")

        texts = []
        for el in content_els:
            txt = el.text.strip()
            # Ignore short strings, the author name, or date
            if txt and txt not in [data["author"], data["date"]] and len(txt) > 20: 
                texts.append(txt)
                
        if texts:
            # If we used the precise message selector, just join them (sometimes split into paragraphs)
            if container.find_elements(By.XPATH, ".//div[@data-ad-comet-preview='message']"):
                data["content"] = "\n".join(texts)
            else:
                # Fallback: We might capture header text like the author name again, just find the longest viable text.
                # Photo theater popups usually have the text in one of the first few divs.
                # Filter out obvious non-content
                valid_texts = [t for t in texts if len(t) > 50 and data["author"] not in t]
                if valid_texts:
                    data["content"] = valid_texts[0]
                    for t in valid_texts:
                         if len(t) > len(data["content"]):
                             data["content"] = t
                else: 
                     data["content"] = max(texts, key=len)
                
        print(f"  -> Content Length: {len(data['content'])} chars")
    except Exception as e:
        print(f"  [ERROR] Finding content: {e}")
    
    # 2. Media (Images and Videos)
    if not os.path.exists(MEDIA_FOLDER):
        os.makedirs(MEDIA_FOLDER)
        
    try:
        media_count = 0
        # Exclude images that are inside the comments area
        img_els = container.find_elements(By.XPATH, ".//img[(contains(@src, 'scontent') or contains(@src, 'fbcdn')) and not(ancestor::div[contains(@aria-label, 'Comment') or contains(@aria-label, 'Bình luận') or contains(@dir, 'auto') and descendant::a])]")
        
        for img in img_els:
            src = img.get_attribute("src")
            # Filter out small UI elements, stickers, emojis
            if not src or "p100x100" in src or "emoji" in src.lower() or "rsrc.php" in src:
                continue
                
            try:
                w_str = img.get_attribute("width") or img.get_attribute("naturalWidth")
                h_str = img.get_attribute("height") or img.get_attribute("naturalHeight")
                
                w = int(w_str) if w_str and w_str.isdigit() else 0
                h = int(h_str) if h_str and h_str.isdigit() else 0
                
                # Further strict checking: valid post images are usually quite large
                if w > 250 and h > 250: 
                    filename = f"post_{index}_img_{media_count}.jpg"
                    filepath = os.path.join(MEDIA_FOLDER, filename)
                    download_file(src, filepath)
                    data["media"].append(filepath)
                    media_count += 1
            except:
                pass
                
        vid_els = container.find_elements(By.XPATH, ".//video")
        for vid in vid_els:
            src = vid.get_attribute("src")
            if src:
                filename = f"post_{index}_vid_{media_count}.mp4"
                filepath = os.path.join(MEDIA_FOLDER, filename)
                download_file(src, filepath)
                data["media"].append(filepath)
                media_count += 1
                
        print(f"  -> Downloaded {media_count} media files.")
    except Exception as e:
        print(f"  [ERROR] Extracting media: {e}")

    # 3. Comments (Chính chủ)
    try:
        # Scroll logic
        driver.execute_script("window.scrollBy(0, 500);")
        time.sleep(2)
        driver.execute_script("window.scrollBy(0, 1000);")
        time.sleep(2)
            
        # Find comments
        comment_els = container.find_elements(By.XPATH, ".//div[contains(@aria-label, 'Comment') or contains(@aria-label, 'Bình luận') or contains(@dir, 'auto')]")
        
        # A common author format: "Bài viết của Thông tin Chính phủ" or just "Thông tin Chính phủ"
        # We need to strip standard prefixes if they exist
        clean_author = data["author"].replace("Bài viết của ", "").replace("Post by ", "").strip()

        for c in comment_els:
            aria = c.get_attribute("aria-label") or ""
            if "Comment" in aria or "Bình luận" in aria:
                try:
                    c_author_els = c.find_elements(By.XPATH, ".//a")
                    comment_author = ""
                    for a in c_author_els:
                        txt = a.text.strip()
                        if txt:
                            comment_author = txt
                            break # Assume first link is the commenter name
                    
                    if clean_author and clean_author in comment_author:
                        # This is a comment by the post author
                        # We must distinguish between the name link and the actual comment body
                        # Usually, the comment body is the immediate `div[dir='auto']` that does not match the name
                        text_divs = c.find_elements(By.XPATH, ".//div[@dir='auto']")
                        
                        texts = []
                        for div in text_divs:
                            t = div.text.strip()
                            # Filter out author name, "Tác giả" badges, generic replies/likes strings and short garbage
                            if t and t not in comment_author and t != "Tác giả" and t != "Author" and t not in ["Thích", "Phản hồi", "Chia sẻ", "Like", "Reply", "Share"] and t not in texts and len(t) > 2:
                                texts.append(t)
                        
                        full_text = "\n".join(texts)
                        if full_text and full_text not in data["comments"]:
                            data["comments"].append(full_text)
                except:
                    pass
        
        print(f"  -> Found {len(data['comments'])} comments from author.")
    except Exception as e:
        print(f"  [ERROR] Extracting comments: {e}")

    return data

def main():
    if not os.path.exists(LINK_FILE):
        print(f"Link file {LINK_FILE} not found!")
        return
        
    links = load_links()
    if not links:
        print("No links found!")
        return

    driver = init_driver()
    driver.get("https://facebook.com")
    time.sleep(5)
    load_cookies(driver)
    driver.refresh()
    time.sleep(5)

    results = []
    for i, link in enumerate(links):
        try:
            post_data = extract_post(driver, link, i+1)
            results.append(post_data)
        except Exception as e:
            print(f"  [ERROR] Processing {link}: {e}")

    driver.quit()

    output_file = "output.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"\nDone! Saved {len(results)} posts to {output_file}")

if __name__ == "__main__":
    main()

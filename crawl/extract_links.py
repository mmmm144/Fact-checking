import time
import json
import random
import csv
import os

from selenium import webdriver
from selenium.webdriver.common.by import By


import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COOKIE_FILE = os.path.join(SCRIPT_DIR, '../data/vie/raw/facebook_cookies.json')

def init_driver():

    options = webdriver.ChromeOptions()
    options.add_argument("--disable-notifications")
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")

    driver = webdriver.Chrome(options=options)

    return driver


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


def extract_post_links(driver):

    links = set()

    posts = driver.find_elements(By.CSS_SELECTOR, "div[role='article']")

    for post in posts:

        try:

            a_tags = post.find_elements(By.CSS_SELECTOR, "a[href]")

            for a in a_tags:

                href = a.get_attribute("href")

                if href and (
                    "/posts/" in href
                    or "story_fbid=" in href
                    or "/videos/" in href
                    or "/photos/" in href
                ):

                    links.add(href.split("?")[0])

        except:
            continue

    return links


def crawl_page(driver, url, scroll_times=30):

    driver.get(url)

    time.sleep(5)

    all_links = set()

    for i in range(scroll_times):

        print("Scrolling:", i + 1)

        links = extract_post_links(driver)

        all_links.update(links)

        driver.execute_script(
            "window.scrollTo(0, document.body.scrollHeight)"
        )

        time.sleep(random.uniform(3, 5))

    return all_links


# ---------- SAVE FUNCTIONS ----------

def save_to_txt(links, filename="facebook_links.txt"):

    with open(filename, "w", encoding="utf-8") as f:
        for link in links:
            f.write(link + "\n")


def save_to_csv(links, filename="facebook_links.csv"):

    with open(filename, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow(["post_link"])

        for link in links:
            writer.writerow([link])

# ---------- MAIN ----------

def main():

    pages = [
        "https://www.facebook.com/tintucvtv24",
        "https://www.facebook.com/ThongtanxaVietNam",
        "https://www.facebook.com/baotuoitre",
        "https://www.facebook.com/baothanhnien",
        "https://www.facebook.com/congdongvnexpress"
    ]

    driver = init_driver()

    driver.get("https://www.facebook.com")

    time.sleep(5)

    load_cookies(driver)

    driver.refresh()

    time.sleep(5)

    all_links = set()

    for page in pages:

        print("\nCrawling:", page)

        links = crawl_page(driver, page)

        all_links.update(links)

    driver.quit()

    print("\nTotal links:", len(all_links))

    # lưu file
    save_to_txt(all_links, os.path.join(SCRIPT_DIR, "../data/vie/raw/facebook_links.txt"))
    save_to_csv(all_links, os.path.join(SCRIPT_DIR, "../data/vie/raw/facebook_links.csv"))

    print("Saved to facebook_links.txt and facebook_links.csv")


if __name__ == "__main__":
    main()
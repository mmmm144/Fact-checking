# Vietnamese Fact-Checking Dataset (ViFC) Pipeline

Dự án này là hệ thống thu thập, tiền xử lý, trích xuất và gán nhãn tự động để xây dựng bộ dữ liệu kiểm chứng thông tin (Fact-Checking Dataset) tiếng Việt phục vụ nghiên cứu khoa học.

---

## 📁 Cấu trúc Thư mục Dự án

```
.
├── crawl/                      # Toàn bộ mã nguồn thu thập dữ liệu (Web, FB, Sitemap)
│   ├── web_crawl.py            # Cào cảnh báo VAFC/NCSC & tin tức VnExpress/Tuổi Trẻ/Nhân Dân
│   ├── crawl_sitemap.py        # Cào các sitemap chi tiết của VAFC để thu thập tin giả
│   ├── facebook_crawl.py       # Cào bài đăng Facebook chi tiết và dọn dẹp media
│   ├── extract_links.py        # Trích xuất liên kết bài viết từ các trang Facebook chính thống
│   └── recrawl_dates.py        # Cập nhật và sửa lỗi ngày tháng cho bài đăng Facebook
│
├── src/                        # Thư mục xử lý logic chính và đánh giá
│   ├── utils.py                # Hàm tiện ích làm sạch văn bản & chuẩn hóa ngày tháng tiếng Việt
│   ├── generate_dataset.py     # Tích hợp dữ liệu, trích xuất tin đồn (FALSE) và sinh tin giả nhân tạo
│   └── evaluate.py             # Đánh giá phân phối dữ liệu và kiểm định tính toàn vẹn
│
├── data/
│   └── vie/                    # Thư mục chứa dữ liệu tiếng Việt (Vietnamese Data)
│       ├── raw/                # Dữ liệu thô sau khi cào (JSON, CSV, TXT)
│       │   ├── output.json
│       │   ├── news_website_output.json
│       │   ├── vafc_sitemap_output.json
│       │   ├── facebook_links.txt
│       │   ├── facebook_links.csv
│       │   └── facebook_cookies.json
│       └── processed/          # Dữ liệu đích đã chuẩn hóa hoàn chỉnh
│           └── fact_checking_dataset.json
│
└── README.md                   # Hướng dẫn chi tiết sử dụng dự án
```

---

## 📊 Mô tả Schema Dữ liệu (`fact_checking_dataset.json`)

Mỗi mẫu tin tức trong tệp dữ liệu đích [fact_checking_dataset.json](file:///Users/ai/Documents/HCMUS/NCKH/crawl/data/vie/processed/fact_checking_dataset.json) tuân thủ định dạng cấu trúc chuẩn dưới đây:

| Trường dữ liệu | Kiểu dữ liệu | Mô tả |
| :--- | :--- | :--- |
| `id` | `String` | Mã định danh duy nhất (MD5 hash của URL + Claim) |
| `source_type` | `String` | Loại nguồn dữ liệu (`website` hoặc `facebook`) |
| `source_name` | `String` | Tên nguồn phát hành (Ví dụ: `VAFC`, `NCSC`, `VnExpress`, `VTV24`) |
| `url` | `String` | Liên kết gốc tới bài đăng hoặc cảnh báo |
| `domain` | `String` | Lĩnh vực/chuyên mục thông tin (Ví dụ: `Tài chính`, `Sức khỏe`, `Say nắng`) |
| `publish_date` | `String` | Ngày phát hành đã chuẩn hóa định dạng `YYYY-MM-DD` |
| `claim` | `String` | Phát ngôn/Tuyên bố cần kiểm chứng (Claim) |
| `original_text` | `String` | Đoạn văn bản gốc/sapo của bài viết |
| `label` | `String` | Nhãn độ xác thực: **`TRUE`** (Tin đúng) hoặc **`FALSE`** (Tin đồn/Lừa đảo) |
| `evidence` | `String` | Bằng chứng kiểm chứng trực tiếp (Đối sánh từ cơ quan chức năng) |
| `justification` | `String` | Nội dung lý giải, phân tích chi tiết vì sao đúng hoặc sai |
| `comments` | `List[String]` | Danh sách bình luận đính kèm (Chỉ áp dụng với nguồn Facebook) |
| `media_links` | `List[String]` | Danh sách liên kết ảnh/video đính kèm (Chỉ áp dụng với nguồn Facebook) |

---

## 🚀 Hướng dẫn Chạy Quy trình

Để cập nhật và tái tạo lại toàn bộ tập dữ liệu, hãy thực hiện theo thứ tự các bước sau:

### Bước 1: Trích xuất và Thu thập dữ liệu mạng xã hội
1. Chuẩn bị file cookie tài khoản Facebook tại `data/vie/raw/facebook_cookies.json` để tránh bị chặn.
2. Trích xuất danh sách link Facebook cần cào:
   ```bash
   python3 crawl/extract_links.py
   ```
3. Cào chi tiết các bài đăng từ danh sách link:
   ```bash
   python3 crawl/facebook_crawl.py
   ```
4. (Tùy chọn) Nếu phát hiện ngày tháng cào Facebook bị lỗi hoặc thiếu, chạy sửa lỗi:
   ```bash
   python3 crawl/recrawl_dates.py
   ```

### Bước 2: Thu thập dữ liệu từ Website Cảnh báo & Tin tức
1. Cào tin tức chính thống và blacklist cảnh báo thô:
   ```bash
   python3 crawl/web_crawl.py
   ```
2. Cào bổ sung dữ liệu sitemap VAFC để lấy tin đồn (phục vụ trích xuất tin giả thật):
   ```bash
   python3 crawl/crawl_sitemap.py
   ```

### Bước 3: Tổng hợp và gộp bộ dữ liệu Fact-Checking
Chạy script sinh dữ liệu chính. Script sẽ tự động chuẩn hóa ngày tháng, trích xuất claim tin đồn, sinh tin giả nhân tạo từ NCSC và gộp thành bộ dữ liệu đích:
```bash
python3 src/generate_dataset.py
```

### Bước 4: Kiểm tra và Đánh giá thống kê
Đo lường phân phối nhãn và kiểm tra tính toàn vẹn của dữ liệu:
```bash
python3 src/evaluate.py
```

---

## 📈 Thống kê Hiện tại của Tập Dữ liệu
* **Tổng số mẫu:** `618 mục`
* **TRUE:** `367 mục (59.39%)` (Cảnh báo chính thống + Tin chính thống)
* **FALSE:** `251 mục (40.61%)` (Tin đồn cào từ VAFC + Tin giả nhân tạo sinh từ NCSC)
* **Tỷ lệ khuyết thiếu dữ liệu:** `0%`



<!-- $env:GEMINI_API_KEY = " "; try {
      python .\generation_claim\fact_checking_Gemini_free.py
  } finally {
      Remove-Item Env:\GEMINI_API_KEY -ErrorAction SilentlyContinue
  } -->
# Crawler 4 nguồn dữ liệu

Pipeline thu thập tối đa **5.700 bài** theo quota mặc định:

| Source | Số bài mục tiêu | Nhãn mặc định |
|---|---:|---|
| VAFC (`tingia.gov.vn`) | 1.600 | `FALSE` |
| Bộ Y tế (`moh.gov.vn`) | 1.500 | `TRUE` |
| WHO (`who.int`) | 1.400 | `TRUE` |
| GSO/NSO (`gso.gov.vn` chuyển sang `nso.gov.vn`) | 1.200 | `TRUE` |

## Cài đặt và chạy

```powershell
python -m pip install -r loc_crawl/requirements.txt
python loc_crawl/crawl_sources.py
```

Chạy thử 5 bài mỗi nguồn:

```powershell
python loc_crawl/crawl_sources.py --limit 5 --delay 0.5
```

Chỉ dò và đếm URL, chưa tải nội dung:

```powershell
python loc_crawl/crawl_sources.py --discover-only
```

Chạy riêng một nguồn:

```powershell
python loc_crawl/crawl_sources.py --source vafc
python loc_crawl/crawl_sources.py --source moh
python loc_crawl/crawl_sources.py --source who
python loc_crawl/crawl_sources.py --source gso
```

## Đầu ra và resume

Kết quả nằm trong `loc_crawl/output/`:

- `vafc.json`, `moh.json`, `who.json`, `gso.json`: dữ liệu từng nguồn, được format dễ đọc;
- `all_articles.json`: dữ liệu đã gộp và loại trùng;
- `.checkpoints/*.jsonl`: checkpoint nội bộ dùng để resume;
- `crawl_summary.json`: số URL tìm được, số bài thành công và phần còn thiếu quota.

## Notebook EDA

Bốn notebook trong `loc_crawl/notebooks/` phân tích riêng từng nguồn, tập trung vào phân bố `domain`, độ phủ taxonomy, domain theo năm, bài đa-domain, chất lượng dữ liệu và độ dài nội dung:

- `eda_gso.ipynb`
- `eda_moh.ipynb`
- `eda_vafc.ipynb`
- `eda_who.ipynb`

Mở thư mục notebook bằng lệnh:

```powershell
jupyter notebook .\loc_crawl\notebooks
```

Notebook tự tìm JSON khi được mở từ thư mục gốc dự án, `loc_crawl/` hoặc `loc_crawl/notebooks/`. Mặc định notebook chỉ đọc và phân tích; đặt `EXPORT_TABLES = True` ở cell cuối nếu muốn xuất các bảng CSV vào `loc_crawl/output/eda/`.

Mỗi bài được ghi ngay sau khi parse thành công. Chạy lại cùng lệnh sẽ đọc các URL đã có và tiếp tục phần còn thiếu. Mã thoát `2` có nghĩa website không cung cấp đủ bài hợp lệ để đạt quota; crawler không tự tạo dữ liệu bù.

Mỗi record có đúng 12 trường theo thứ tự: `id`, `source_type`, `source_name`, `url`, `domain`, `publish_date`, `claim`, `original_text`, `label`, `evidence`, `justification`, `comments`. File JSON dùng indent 4 spaces để mỗi trường nằm trên một dòng riêng.

`domain` là taxonomy thật theo từng nguồn, không phải một danh sách ba nhãn cố định: VAFC dùng toàn bộ sitemap chuyên mục và giữ URL→domain đã được curate trong dữ liệu dự án. WHO crawl từ đúng 8 trang cha và gắn một trong 8 domain: `News`, `Emergencies`, `Campaigns`, `Events`, `Statements`, `Feature stories`, `Speeches`, `Commentaries`; domain của trang cha được ưu tiên vì nhiều mục WHO dùng chung route `/news/item/`. Bộ Y tế crawl 23 chuyên mục con có bài viết trong menu chính thức và quy về 4 domain cha: `Tin tức - sự kiện`, `Cung cấp thông tin`, `Tra cứu`, `Chuyển đổi số y tế`; bài trùng giữa nhiều mục được giữ một lần và ưu tiên domain chuyên biệt hơn. Với GSO/NSO, 16 feature tiếng Việt gắn trong WordPress category của bài được quy về 5 domain cha trên website: `Dân số và lao động`, `Tài khoản quốc gia và tài chính`, `Kinh tế`, `Xã hội môi trường và đơn vị hành chính`, `Tổng điều tra`. Category gom chung `Chủ đề khác` không được dùng làm domain; bài không có ít nhất một trong 16 feature sẽ bị bỏ qua. Nếu một bài thuộc feature của nhiều domain cha, các domain được giữ trong cùng chuỗi, phân cách bằng `;` và tự động loại trùng. Checkpoint cũ có tên feature con được tự động quy đổi khi chạy lại.

## Lưu ý về nhãn

Các nhãn trên mô phỏng quy ước của crawler cũ: VAFC là nguồn kiểm chứng/cảnh báo nên gán `FALSE`, còn nguồn cơ quan chính thức gán `TRUE`. Đây là **nhãn theo nguồn**, không phải kết luận tự động rằng mọi câu xuất hiện trong toàn văn đều đúng/sai. Nếu dùng cho nghiên cứu fact-checking ở mức claim, nên kiểm tra và gán nhãn thủ công sau bước crawl.

Crawler mặc định tuân thủ `robots.txt`, giới hạn tốc độ theo từng host và retry khi gặp lỗi tạm thời. Không nên đặt `--delay` quá thấp.

## Trạng thái nguồn đã kiểm tra (21/07/2026)

- Sitemap VAFC hiện chỉ cung cấp **502 URL bài duy nhất**. Vì vậy quota 1.600 sẽ được giữ làm mục tiêu nhưng báo cáo có `shortfall`; crawler không nhân bản hoặc bịa thêm bài.
- WHO kết hợp sitemap với API hub và phân trang của 8 trang cha để giữ đúng domain nguồn.
- Sitemap GSO, hiện chuyển hướng sang NSO, cung cấp 4.058 URL bài tiếng Việt.
- Bộ Y tế dùng cổng động mới: crawler đọc API danh mục và API chi tiết chính thức của `moh.gov.vn`, đồng thời lấy `domain` từ chuyên mục thật của từng bài. Các URL Liferay cũ và host `adminmoh.moh.gov.vn` không còn được gọi.

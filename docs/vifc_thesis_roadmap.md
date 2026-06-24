# Lộ trình Đồ án: Hệ thống RAG Fact-Checking Tiếng Việt (ViFC)

> Tài liệu này thay thế kế hoạch pilot trước đó. Mục tiêu: đưa ra plan chi tiết, đúng trọng tâm,
> bám sát hiện trạng dữ liệu/code thực tế và chạy xuyên suốt đến khi hoàn thành đồ án.

---

## 0. Hiện trạng (đã làm xong)

- Crawl: VAFC, NCSC, VnExpress/Tuổi Trẻ/Nhân Dân, Facebook fanpage (`crawl/`).
- Tiền xử lý + gán nhãn TRUE/FALSE, hợp nhất dataset (`src/generate_dataset.py`).
- Evidence corpus đã chunk: `data/vie/processed/fact_checking_dataset_chunked.json` (618 bản ghi, 1145 chunks).
- Semantic chunking bằng Underthesea + BGE-M3 (`chunking/chunking.py`).
- Claim evaluation set: `data/vie/processed/newdata.json` (70 bài, 420 claim, cân bằng 140/140/140).
- Pilot audit: `reports/pilot/` (audit_summary.json, pilot_report.md).

## 0.1 Vấn đề tồn đọng cần xử lý trước khi mở rộng

- Encoding mojibake: 82/14516 field nghi lỗi trong evidence corpus.
- Thiếu field: 47 bản ghi thiếu `justification`, 10 bản ghi không có `chunks`.
- Mất cân bằng nguồn: Facebook chỉ 15/618 (~2.4%).
- Quy mô nhỏ (618 evidence / 420 claim) chỉ đủ làm pilot.
- Nguy cơ data leakage: chưa tách train/dev/test theo `article_id`/`url`.
- API key bị hardcode trong `fact_checking_script.py` và `src/generate_claims.py` (rủi ro bảo mật).

---

## GIAI ĐOẠN 1 — Củng cố & làm sạch dữ liệu (Tuần 1-2)

Mục tiêu: dataset sạch, đủ field, không leakage, có thể tái lập.

1.1. Sửa encoding
- Viết `src/fix_encoding.py` dò và sửa mojibake (ftfy hoặc re-decode latin1→utf-8).
- Kiểm lại: `mojibake_like_fields = 0` trên cả 2 file.

1.2. Bổ sung field thiếu
- Sinh lại `chunks` cho 10 bản ghi rỗng (chạy lại `chunking/chunking.py` cho phần thiếu).
- Bù `justification` cho 47 bản ghi: lấy từ nguồn gốc hoặc đánh dấu rõ là rỗng có chủ đích.

1.3. Khử trùng lặp & chuẩn hóa
- Dedup theo URL + near-duplicate (hash nội dung / cosine BGE-M3 > 0.95).
- Chuẩn hóa `publish_date` về `YYYY-MM-DD`, thống nhất `source_name`.

1.4. Tách split chống leakage
- Tách train/dev/test theo `article_id`/`url` (không để chunk cùng bài lọt 2 tập).
- Tỉ lệ gợi ý 70/15/15, giữ cân bằng nhãn trong mỗi split.
- Lưu `data/vie/processed/splits/{train,dev,test}.json` + ghi seed cố định.

1.5. Bảo mật
- Gỡ API key hardcode, chuyển sang biến môi trường (`.env` + đọc qua `os.environ`).
- Thêm `.env` vào `.gitignore`; nếu key đã commit thì xoay (rotate) key.

Đầu ra: dataset v1 sạch + báo cáo audit cập nhật (`reports/data_v1_audit.json`).

---

## GIAI ĐOẠN 2 — Mở rộng dữ liệu (Tuần 2-4)

Mục tiêu: tăng quy mô và đa dạng, giảm mất cân bằng nguồn.

2.1. Mở rộng crawl VAFC/NCSC + báo chính thống (tăng evidence corpus mục tiêu ~2000+ bản ghi).
2.2. Tăng tỉ lệ Facebook (mục tiêu >= 10-15%) từ các fanpage chính thống.
2.3. Đa dạng chủ đề: tài chính, y tế, an ninh mạng, chính sách, đời sống.
2.4. Sinh claim evaluation set lớn hơn theo cùng schema 3 nhãn (mục tiêu >= 1000 claim).
2.5. Soát quy tắc sinh nhãn FALSE để giảm nhiễu (kiểm thủ công mẫu ngẫu nhiên).

Đầu ra: dataset v2 + thống kê phân phối nguồn/nhãn/chủ đề.

---

## GIAI ĐOẠN 3 — Kiểm duyệt chất lượng (Tuần 4-5)

Mục tiêu: nâng độ tin cậy nhãn cho tập đánh giá.

3.1. Kiểm duyệt thủ công tập test claim (đảm bảo evidence trích nguyên văn, nhãn đúng).
3.2. Đo độ đồng thuận (inter-annotator agreement, Cohen's kappa) trên mẫu.
3.3. Lập guideline gán nhãn ngắn để tái lập.
3.4. Loại/sửa claim mơ hồ giữa REFUTED và NOT_ENOUGH_INFO.

Đầu ra: gold test set đã kiểm duyệt + báo cáo chất lượng nhãn.

---

## GIAI ĐOẠN 4 — Module Retrieval (Tuần 5-7)

Mục tiêu: truy xuất evidence chunk chính xác cho claim.

4.1. Baseline lexical: BM25 / TF-IDF trên `chunks[].text` (`src/retrieval/bm25.py`).
4.2. Dense retrieval: BGE-M3 embedding + FAISS/Chroma index (`src/retrieval/dense.py`).
4.3. Hybrid: kết hợp BM25 + dense (reciprocal rank fusion).
4.4. (Tùy chọn) Re-ranker cross-encoder để tăng precision top-k.
4.5. Đánh giá: Recall@k (k=1,3,5,10), MRR, nDCG trên dev/test.

Đầu ra: bảng so sánh retrieval + chọn cấu hình tốt nhất.

---

## GIAI ĐOẠN 5 — Module Verification (Tuần 7-9)

Mục tiêu: phân loại claim thành SUPPORTED / REFUTED / NOT_ENOUGH_INFO.

5.1. Baseline zero/few-shot: LLM nhận claim + top-k evidence → verdict + giải thích.
5.2. Baseline học máy: fine-tune PhoBERT/XLM-R cho NLI 3 nhãn.
5.3. Xử lý NOT_ENOUGH_INFO (ngưỡng độ tin cậy retrieval / abstain).
5.4. Đánh giá: Accuracy, Macro-F1, confusion matrix theo từng nhãn.
5.5. Ablation: tác động của số evidence k và chất lượng retrieval lên verdict.

Đầu ra: bảng kết quả verification + phân tích lỗi.

---

## GIAI ĐOẠN 6 — Module Explanation (Tuần 9-10)

Mục tiêu: sinh giải thích minh bạch, có trích dẫn.

6.1. Sinh giải thích dựa trên evidence đã truy xuất (grounded, chống bịa).
6.2. Trích dẫn nguồn (URL + đoạn evidence).
6.3. Đánh giá thủ công theo rubric: đúng bằng chứng, hợp lý, ngắn gọn, không bịa.
6.4. (Tùy chọn) Đo faithfulness tự động (vd. tỉ lệ câu giải thích có evidence hỗ trợ).

Đầu ra: ví dụ giải thích + điểm rubric trung bình.

---

## GIAI ĐOẠN 7 — Tích hợp hệ thống end-to-end (Tuần 10-11)

Mục tiêu: ghép pipeline thành 1 hệ thống chạy được.

7.1. Pipeline: Claim → Preprocess → Retrieval → Verifier → Explanation + Citation.
7.2. API service (FastAPI) nhận claim, trả verdict + evidence + giải thích.
   - Lưu ý bảo mật: thêm auth (API key/token) trước khi expose ra mạng.
7.3. Demo UI tối giản (Streamlit/Gradio) cho người dùng nhập claim.
7.4. Đánh giá end-to-end trên test set (verdict + retrieval đồng thời).

Đầu ra: demo chạy được + script đánh giá toàn pipeline.

---

## GIAI ĐOẠN 8 — Thực nghiệm & phân tích (Tuần 11-13)

Mục tiêu: số liệu đầy đủ cho báo cáo.

8.1. Chạy đầy đủ các cấu hình (lexical vs dense vs hybrid; LLM vs fine-tune).
8.2. Bảng kết quả tổng hợp + biểu đồ.
8.3. Phân tích lỗi định tính (case study từng nhãn).
8.4. Kiểm tra lại không có leakage; báo cáo giới hạn (limitations).

Đầu ra: toàn bộ số liệu thực nghiệm + hình/bảng.

---

## GIAI ĐOẠN 9 — Viết báo cáo & bảo vệ (Tuần 13-15)

9.1. Viết báo cáo: giới thiệu, related work, dữ liệu, phương pháp, thực nghiệm, kết luận.
9.2. Hoàn thiện README + hướng dẫn tái lập (requirements, lệnh chạy từng bước).
9.3. Slide + demo cho buổi bảo vệ.
9.4. Đóng gói dataset + code, ghi rõ license/nguồn dữ liệu.

Đầu ra: báo cáo hoàn chỉnh + slide + repo tái lập được.

---

## Phụ lục A — Chỉ số đánh giá tổng hợp

| Module | Chỉ số chính |
| :--- | :--- |
| Retrieval | Recall@k, MRR, nDCG |
| Verification | Accuracy, Macro-F1, Confusion matrix |
| Explanation | Rubric thủ công, faithfulness |
| End-to-end | Verdict accuracy có điều kiện retrieval |

## Phụ lục B — Rủi ro & giảm thiểu

- Data leakage → tách split theo article_id, kiểm tra trùng giữa các tập.
- Nhãn nhiễu (FALSE sinh tự động) → kiểm duyệt thủ công + IAA.
- Dữ liệu nhỏ → mở rộng crawl + tăng đa dạng nguồn/chủ đề.
- Phụ thuộc API LLM → khóa version model, lưu cache output, có baseline offline.
- Lộ secret → biến môi trường, .gitignore, rotate key đã lộ.

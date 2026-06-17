# Kế Hoạch Pilot: RAG Fact-Checking Tiếng Việt Đa Bước

## 1. Tên hướng đề tài

Xây dựng hệ thống kiểm chứng thông tin tiếng Việt theo hướng RAG đa bước: truy xuất bằng chứng, phân loại mức độ đúng/sai và sinh giải thích.

## 2. Mục tiêu

Hệ thống nhận một claim tiếng Việt từ người dùng, tìm các đoạn bằng chứng liên quan trong kho dữ liệu, sau đó đưa ra kết luận:

- `SUPPORTED`: bằng chứng ủng hộ claim.
- `REFUTED`: bằng chứng bác bỏ claim.
- `NOT_ENOUGH_INFO`: chưa đủ bằng chứng để kết luận.

Ngoài nhãn dự đoán, hệ thống cần trả về evidence và giải thích ngắn để tăng tính minh bạch.

## 3. Dataset pilot hiện có

### Evidence corpus

- File: `data/vie/processed/fact_checking_dataset_chunked.json`
- Vai trò: kho tri thức đã chia chunk để phục vụ truy xuất evidence.
- Mỗi bản ghi gồm thông tin nguồn, claim/tựa bài, nhãn `TRUE/FALSE`, nội dung gốc, justification và danh sách `chunks`.

### Claim-level evaluation set

- File: `data/vie/processed/newdata.json`
- Vai trò: tập claim-level để đánh giá bước xác minh claim.
- Mỗi bài có các claim được gán nhãn `SUPPORTED`, `REFUTED`, `NOT_ENOUGH_INFO`, kèm evidence quote và reason.

Dataset hiện tại dùng cho pilot/prototype, chưa dùng để kết luận chất lượng mô hình ở quy mô lớn. Sau khi hướng được duyệt, dataset sẽ được mở rộng theo cùng schema.

## 4. Pipeline hệ thống

```text
User Claim
    ↓
Text Preprocessing
    ↓
Evidence Retrieval
    ↓
Top-k Evidence Chunks
    ↓
Claim Verifier
    ↓
Verdict: SUPPORTED / REFUTED / NOT_ENOUGH_INFO
    ↓
Explanation + Evidence Citation
```

## 5. Phương pháp baseline

### Retrieval baseline

- BM25 hoặc TF-IDF trên trường `chunks[].text`.
- Input: claim.
- Output: top-k evidence chunks liên quan nhất.

### Verification baseline

- Cách đơn giản ban đầu: dùng claim + top-k evidence làm input cho mô hình/ngữ cảnh suy luận.
- Cách nâng cao: fine-tune PhoBERT/XLM-R cho bài toán 3 nhãn.

### Explanation

- Trích các câu evidence gần nhất.
- Sinh lý do ngắn theo cấu trúc: claim được ủng hộ/bác bỏ/chưa đủ thông tin vì evidence cho thấy điều gì.

## 6. Đánh giá

### Retrieval

- `Recall@k`: evidence đúng có nằm trong top-k không.
- `MRR`: evidence đúng xuất hiện ở thứ hạng bao nhiêu.

### Verification

- `Accuracy`: tỷ lệ dự đoán đúng.
- `Macro-F1`: phù hợp khi các nhãn có thể mất cân bằng.

### Explanation

- Đánh giá thủ công theo rubric: đúng bằng chứng, hợp lý, ngắn gọn, không bịa thông tin.

## 7. Kế hoạch mở rộng sau khi được duyệt

1. Mở rộng crawl từ VAFC, NCSC và các nguồn báo chính thống.
2. Chuẩn hóa encoding, loại trùng và kiểm tra field thiếu.
3. Tạo evidence corpus lớn hơn và claim-level evaluation set cân bằng nhãn.
4. Tách train/dev/test theo `article_id` hoặc `url` để tránh data leakage.
5. Xây baseline BM25/TF-IDF trước, sau đó thử embedding retrieval và verifier model.
6. Làm demo app/API cho phép nhập claim và xem kết quả kiểm chứng.

## 8. Nội dung trình giảng viên

Hiện tại dataset còn nhỏ nên mục tiêu là chứng minh pipeline khả thi. Bộ dữ liệu pilot gồm một corpus đã chunk để truy xuất evidence và một tập claim-level có ba nhãn để đánh giá xác minh claim. Nếu hướng này được duyệt, em sẽ mở rộng dataset theo cùng cấu trúc, làm sạch dữ liệu và đánh giá bằng Recall@k, MRR, Accuracy và Macro-F1.

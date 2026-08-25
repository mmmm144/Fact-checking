# Quy trình từ Chunking đến Verifier

## Sơ đồ chi tiết

```text
┌───────────────────────────────────────────────────────────────┐
│              13.572 BÀI VIẾT ĐÃ LÀM SẠCH                    │
└──────────────────────────────┬────────────────────────────────┘
                               ▼
┌───────────────────────────────────────────────────────────────┐
│ 1. CHUNKING                                                  │
│ BGE-M3 tokenizer                                             │
│ • Tối đa 512 token/chunk • Overlap 64 token                 │
│ • Ưu tiên cắt cuối câu • Giữ doc_id, URL và metadata        │
└──────────────────────────────┬────────────────────────────────┘
                               ▼
┌───────────────────────────────────────────────────────────────┐
│                        52.605 CHUNK                           │
└───────────────┬───────────────────────────────┬───────────────┘
                │                               │
                ▼                               ▼
┌──────────────────────────────┐  ┌──────────────────────────────┐
│ 2A. EMBEDDING E5            │  │ 2B. BM25 INDEX              │
│ Chunk + "passage: "          │  │ Tiêu đề + nội dung chunk    │
│          ↓                   │  │          ↓                   │
│ multilingual-e5-large        │  │ Token hóa từ khóa           │
│          ↓                   │  │          ↓                   │
│ Vector 1.024 chiều, chuẩn L2 │  │ Lập chỉ mục BM25            │
└───────────────┬──────────────┘  └──────────────┬───────────────┘
                │                                │
                ▼                                │
┌──────────────────────────────┐                 │
│ PARQUET EMBEDDING            │                 │
│ chunk_id • doc_id • text     │                 │
│ embedding[1024]              │                 │
└───────────────┬──────────────┘                 │
                ▼                                │
┌──────────────────────────────┐                 │
│ 3. FAISS INDEX               │                 │
│ Lưu vector để tìm kiếm       │                 │
│ theo ngữ nghĩa               │                 │
└───────────────┬──────────────┘                 │
                └────────────────┬───────────────┘
                                 ▼
┌───────────────────────────────────────────────────────────────┐
│                  KHO RETRIEVAL SẴN SÀNG                      │
│           FAISS: ngữ nghĩa  +  BM25: từ khóa                 │
└───────────────────────────────────────────────────────────────┘

════════════════════ KHI CÓ CLAIM CẦN KIỂM CHỨNG ══════════════

┌───────────────────────────────────────────────────────────────┐
│ CLAIM NGƯỜI DÙNG                                             │
│ Ví dụ: “Chỉ số sản xuất công nghiệp tháng Ba tăng 18,8%.”    │
└──────────────────────────────┬────────────────────────────────┘
                               ▼
┌───────────────────────────────────────────────────────────────┐
│ 4. TIỀN XỬ LÝ CLAIM                                          │
│ Chuẩn hóa văn bản và loại khoảng trắng thừa                  │
└───────────────┬───────────────────────────────┬───────────────┘
                │                               │
                ▼                               ▼
┌──────────────────────────────┐  ┌──────────────────────────────┐
│ 5A. BM25 SEARCH              │  │ 5B. DENSE SEARCH            │
│ Tìm theo từ khóa             │  │ Claim + "query: "            │
│          ↓                   │  │          ↓                   │
│ BM25 top 100                 │  │ E5 tạo query vector         │
│                              │  │          ↓                   │
│                              │  │ FAISS dense top 100         │
└───────────────┬──────────────┘  └──────────────┬───────────────┘
                └────────────────┬───────────────┘
                                 ▼
┌───────────────────────────────────────────────────────────────┐
│ 6. HYBRID RETRIEVAL — RRF                                   │
│ Gộp thứ hạng BM25 và Dense bằng RRF, k = 60                 │
└──────────────────────────────┬────────────────────────────────┘
                               ▼
┌───────────────────────────────────────────────────────────────┐
│                  HYBRID CANDIDATES TOP 50                    │
└──────────────────────────────┬────────────────────────────────┘
                               ▼
┌───────────────────────────────────────────────────────────────┐
│ 7. BGE RERANKER                                              │
│ Model: BAAI/bge-reranker-v2-m3                              │
│ Đọc từng cặp claim + chunk, chấm điểm và xếp hạng lại       │
└──────────────────────────────┬────────────────────────────────┘
                               ▼
┌───────────────────────────────────────────────────────────────┐
│ TOP 5–10 EVIDENCE CHUNK                                      │
│ Tối đa 3 chunk từ cùng một tài liệu                          │
└──────────────────────────────┬────────────────────────────────┘
                               ▼
┌───────────────────────────────────────────────────────────────┐
│ 8. GHÉP EVIDENCE                                             │
│ [Bằng chứng 1] Tiêu đề + nội dung                            │
│ [Bằng chứng 2] Tiêu đề + nội dung                            │
│ [Bằng chứng 3] Tiêu đề + nội dung                            │
└──────────────────────────────┬────────────────────────────────┘
                               ▼
┌───────────────────────────────────────────────────────────────┐
│ 9. VERIFIER                                                  │
│ Model: FacebookAI/xlm-roberta-base                           │
│ Input: evidence đã ghép + claim                              │
│ Output: xác suất của ba nhãn                                 │
└──────────────────────────────┬────────────────────────────────┘
                               │
                  ┌────────────┼────────────┐
                  ▼            ▼            ▼
             SUPPORTED      REFUTED    NOT_ENOUGH_INFO
                Đúng           Sai      Chưa đủ thông tin
                  │            │            │
                  └────────────┴──────┬─────┘
                                     ▼
┌───────────────────────────────────────────────────────────────┐
│ KẾT QUẢ: Label • Confidence • Evidence • URL nguồn           │
└───────────────────────────────────────────────────────────────┘
```

## Luồng rút gọn

```text
Bài viết
   ↓
Chunking
   ↓
Các chunk
   ├──→ Embedding E5 → FAISS index ──┐
   └──→ BM25 index ──────────────────┤
                                      │
Claim → BM25 search ─────────────────┤
   └→ E5 query → FAISS search ───────┤
                                      ▼
                               Hybrid RRF
                                      ↓
                               BGE Reranker
                                      ↓
                               Top-k Evidence
                                      ↓
                               XLM-R Verifier
                                      ↓
                     Đúng / Sai / Chưa đủ thông tin
```

## Vai trò của từng thành phần

| Thành phần | Nhiệm vụ |
|---|---|
| **Chunking** | Chia bài viết dài thành các đoạn bằng chứng nhỏ. |
| **Embedding E5** | Biến chunk và claim thành vector biểu diễn ngữ nghĩa. |
| **FAISS** | Tìm các chunk gần claim nhất về ngữ nghĩa. |
| **BM25** | Tìm các chunk khớp với claim theo từ khóa. |
| **Hybrid RRF** | Gộp thứ hạng của BM25 và Dense/FAISS. |
| **BGE Reranker** | Chấm lại từng cặp claim–chunk và xếp hạng evidence. |
| **Verifier XLM-R** | Kết luận claim đúng, sai hoặc chưa đủ thông tin. |

## Đầu vào và đầu ra

```text
Tài liệu sạch
    ↓ Chunking
Danh sách chunk
    ↓ Embedding
Vector 1.024 chiều/chunk
    ↓ FAISS + BM25
Các chỉ mục retrieval
    ↓ Claim người dùng
Hybrid Retrieval + Reranker
    ↓ Top-k evidence
Verifier
    ↓
Label + confidence + evidence + URL nguồn
```

# Kế hoạch Triển khai Chi tiết: RAG Fact-Checking (Retrieval → Bảo vệ)

> Nối tiếp `docs/vifc_thesis_roadmap.md`. Tài liệu này đặc tả CHI TIẾT phần xây dựng hệ thống:
> Retrieval, Verification, Explanation, Tích hợp end-to-end, Thực nghiệm, Báo cáo, Bảo vệ.
> Bám đúng schema dữ liệu thực tế trong repo.

---

## Schema dữ liệu (đã xác minh trong repo)

Evidence corpus — `data/vie/processed/fact_checking_dataset_chunked.json` (618 bản ghi):
- Bản ghi: `id, source_type, source_name, url, domain, publish_date, claim, original_text, label, evidence, justification, comments, chunks`
- `chunks[]`: `chunk_id, sentence_count, word_count, text, sentences[]`

Claim eval set — `data/vie/processed/newdata.json` (70 bài, 420 claim):
- Bài: `id, date_iso, media, full_text, claims`
- `claims`: dict 3 nhãn `SUPPORTED / REFUTED / NOT_ENOUGH_INFO`
- Mỗi claim: `claim, label, evidence[], reason, image`
- `evidence[]`: `type, quote, article_id, url`  ← `article_id`/`url` là gold để chấm retrieval

Hệ quả thiết kế:
- Đơn vị index = chunk; gold của retrieval = chunk thuộc `article_id`/`url` trong evidence của claim.
- `quote` (trích nguyên văn) dùng để chấm retrieval mức câu/đoạn (chunk chứa quote = relevant).

---

## Cấu trúc thư mục code đề xuất

```
src/
  retrieval/
    base.py            # interface Retriever (index/search)
    bm25.py            # BM25 + tách từ underthesea
    dense.py           # BGE-M3 + FAISS
    hybrid.py          # RRF fusion
    rerank.py          # cross-encoder (tùy chọn)
    build_index.py     # build & lưu index
  verify/
    prompt_verifier.py # LLM few-shot 3 nhãn
    nli_verifier.py    # fine-tune PhoBERT/XLM-R
    threshold.py       # logic NOT_ENOUGH_INFO / abstain
  explain/
    generator.py       # sinh giải thích grounded + citation
  pipeline/
    factcheck.py       # ghép end-to-end
    api.py             # FastAPI service
    demo.py            # Streamlit/Gradio UI
  eval/
    eval_retrieval.py  # Recall@k, MRR, nDCG
    eval_verify.py     # Accuracy, Macro-F1, confusion
    eval_e2e.py        # đánh giá toàn pipeline
    build_qrels.py     # tạo gold relevance từ newdata.json
configs/
  default.yaml         # k, model name, ngưỡng, seed
```

---

## GIAI ĐOẠN 4 — Retrieval (chi tiết)

### 4.0 Chuẩn bị gold relevance (qrels)
- `src/eval/build_qrels.py`: với mỗi claim trong `newdata.json`, map sang chunk gold.
  - Relevant nếu chunk thuộc `article_id`/`url` của claim VÀ chứa (hoặc gần khớp) `quote`.
  - So khớp `quote` ↔ `chunk.text` bằng normalize (bỏ dấu thừa, lowercase) + substring/fuzzy ratio >= 0.9.
- Xuất `data/vie/processed/qrels.json`: `{claim_id: [relevant_chunk_id, ...]}`.

### 4.1 BM25 baseline (`src/retrieval/bm25.py`)
- Tokenize tiếng Việt bằng `underthesea.word_tokenize` (giữ thống nhất với chunking).
- Thư viện `rank_bm25` (BM25Okapi). Index trên `chunks[].text`.
- `search(query, k)` → list `(chunk_id, score)`.
- Tham số: b=0.75, k1=1.2 (tinh chỉnh trên dev).

### 4.2 Dense retrieval (`src/retrieval/dense.py`)
- Encoder `BAAI/bge-m3` (đã dùng ở chunking → nhất quán).
- Encode toàn bộ chunk → lưu `embeddings.npy` + FAISS `IndexFlatIP` (cosine sau normalize).
- Encode query (claim) tương tự, search top-k.
- Cache embedding để khỏi encode lại.

### 4.3 Hybrid (`src/retrieval/hybrid.py`)
- Reciprocal Rank Fusion: `score = Σ 1/(c + rank_i)`, c=60.
- Hợp nhất rank BM25 + dense, lấy top-k sau fusion.

### 4.4 Re-ranker (tùy chọn, `src/retrieval/rerank.py`)
- Cross-encoder `bge-reranker-v2-m3` chấm lại top-50 → top-k.
- Chỉ bật khi cần đẩy precision.

### 4.5 Đánh giá retrieval (`src/eval/eval_retrieval.py`)
- Chỉ số: `Recall@k` (k=1,3,5,10), `MRR@10`, `nDCG@10`.
- Chạy trên dev để chọn cấu hình, báo cáo trên test.
- Xuất bảng `reports/retrieval_results.md` so sánh BM25 / dense / hybrid / +rerank.

Tiêu chí xong GĐ4: có bảng so sánh + cấu hình retrieval tốt nhất được khóa lại trong `configs/default.yaml`.

---

## GIAI ĐOẠN 5 — Verification (chi tiết)

### 5.1 Prompt verifier (`src/verify/prompt_verifier.py`)
- Input: `claim` + top-k chunk text (kèm nguồn).
- Prompt yêu cầu trả JSON: `{label, confidence, cited_chunk_ids}`.
- Label ∈ {SUPPORTED, REFUTED, NOT_ENOUGH_INFO}.
- Few-shot 3-6 ví dụ (mỗi nhãn ≥1), lấy từ train.
- Khóa version model + lưu cache response (`reports/verify_cache/`) để tái lập.

### 5.2 NLI fine-tune (`src/verify/nli_verifier.py`)
- Backbone: `vinai/phobert-base` hoặc `xlm-roberta-base`.
- Cặp đầu vào: (claim, evidence_concat) → 3 lớp.
- Train trên split train, early-stopping theo Macro-F1 dev.
- Lưu checkpoint + log vào `reports/verify_train_log.json`.

### 5.3 Logic NOT_ENOUGH_INFO (`src/verify/threshold.py`)
- Nếu retrieval score top-1 < ngưỡng τ → ép NOT_ENOUGH_INFO (abstain).
- Tinh chỉnh τ trên dev để cân bằng 3 nhãn.

### 5.4 Đánh giá verification (`src/eval/eval_verify.py`)
- `Accuracy`, `Macro-F1`, `Precision/Recall` từng nhãn, confusion matrix.
- 2 chế độ: (a) gold evidence (đo riêng verifier), (b) retrieved evidence (đo end-to-end thực tế).
- Phân tích lỗi: bảng case sai theo từng nhãn.

### 5.5 Ablation
- Tác động của k (1/3/5/10) lên Macro-F1.
- So sánh prompt-verifier vs fine-tune.
- Tác động chất lượng retrieval (gold vs hybrid vs bm25) lên verdict.

Tiêu chí xong GĐ5: bảng kết quả verification (2 chế độ) + phân tích lỗi.

---

## GIAI ĐOẠN 6 — Explanation (chi tiết)

### 6.1 Sinh giải thích (`src/explain/generator.py`)
- Input: claim + verdict + evidence đã cite.
- Ràng buộc grounded: chỉ dùng nội dung trong evidence, cấm thêm dữ kiện ngoài.
- Output: 1-3 câu + danh sách citation (`url` + đoạn `quote`).

### 6.2 Đánh giá explanation
- Rubric thủ công (1-5) trên 4 tiêu chí: đúng bằng chứng, hợp lý, ngắn gọn, không bịa.
- Mẫu đánh giá ≥ 50 claim, báo cáo điểm trung bình + ví dụ tốt/xấu.
- (Tùy chọn) faithfulness tự động: tỉ lệ câu giải thích khớp evidence (NLI entailment).

Tiêu chí xong GĐ6: điểm rubric + ví dụ giải thích kèm citation.

---

## GIAI ĐOẠN 7 — Tích hợp end-to-end (chi tiết)

### 7.1 Pipeline (`src/pipeline/factcheck.py`)
- `fact_check(claim) -> {verdict, confidence, evidence[], explanation, citations[]}`.
- Thứ tự: preprocess → retrieve(top-k) → verify → explain.

### 7.2 API (`src/pipeline/api.py`, FastAPI)
- `POST /factcheck` body `{claim}` → JSON kết quả.
- BẢO MẬT: bắt buộc auth (API key/token) trước khi expose; rate limit; validate input.
- Không hardcode secret (đọc từ biến môi trường).

### 7.3 Demo UI (`src/pipeline/demo.py`, Streamlit/Gradio)
- Ô nhập claim → hiển thị verdict, evidence, giải thích, link nguồn.

### 7.4 Đánh giá end-to-end (`src/eval/eval_e2e.py`)
- Chạy full pipeline trên test: verdict accuracy + retrieval recall đồng thời.
- Đo latency trung bình mỗi claim.

Tiêu chí xong GĐ7: demo chạy được + số liệu end-to-end.

---

## GIAI ĐOẠN 8 — Thực nghiệm & phân tích (chi tiết)

- Chạy ma trận cấu hình: {BM25, dense, hybrid, +rerank} × {prompt, fine-tune}.
- Cố định seed, ghi rõ version model/lib (`requirements.txt` pinned).
- Tổng hợp `reports/final_results.md`: bảng + biểu đồ (Recall@k, Macro-F1).
- Phân tích lỗi định tính: 3-5 case study mỗi nhãn.
- Kiểm tra lại leakage (chunk train không lọt test), báo cáo limitations.

Tiêu chí xong GĐ8: bộ số liệu + hình/bảng hoàn chỉnh cho báo cáo.

---

## GIAI ĐOẠN 9 — Báo cáo & bảo vệ (chi tiết)

- Báo cáo: Giới thiệu → Related work → Dữ liệu (pipeline + thống kê) → Phương pháp (retrieval/verify/explain) → Thực nghiệm → Phân tích lỗi → Kết luận & hướng phát triển.
- README tái lập: cài đặt, build index, chạy eval từng module, chạy demo.
- Slide bảo vệ + kịch bản demo trực tiếp (chuẩn bị sẵn claim mẫu mỗi nhãn).
- Đóng gói: code + dataset + checkpoint + ghi rõ nguồn/license dữ liệu.

---

## Mốc thời gian gợi ý

| Tuần | Hạng mục |
| :--- | :--- |
| 1 | GĐ4.0-4.1 qrels + BM25 |
| 2 | GĐ4.2-4.5 dense/hybrid + eval retrieval |
| 3 | GĐ5.1-5.3 verifier (prompt + fine-tune) |
| 4 | GĐ5.4-5.5 eval verify + ablation |
| 5 | GĐ6 explanation + rubric |
| 6 | GĐ7 pipeline + API + demo |
| 7 | GĐ8 thực nghiệm đầy đủ + phân tích |
| 8 | GĐ9 viết báo cáo + slide + bảo vệ |

> Lưu ý: phải hoàn tất Giai đoạn 1 (làm sạch + tách split chống leakage) ở `docs/vifc_thesis_roadmap.md`
> TRƯỚC khi build index, nếu không mọi số liệu retrieval/verify đều có nguy cơ sai do leakage.

---

## Deliverables tổng

- `src/retrieval/*`, `src/verify/*`, `src/explain/*`, `src/pipeline/*`, `src/eval/*`
- `data/vie/processed/qrels.json`, `splits/{train,dev,test}.json`
- `reports/retrieval_results.md`, `reports/verify_results.md`, `reports/final_results.md`
- Demo app + API + README tái lập + báo cáo + slide

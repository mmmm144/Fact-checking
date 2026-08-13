# Hybrid evidence retrieval trên Kaggle

Pipeline này dùng embedding đã tạo tại
[`Loctran123/vietnamese-evidence-corpus-embeddings-e5-large`](https://huggingface.co/datasets/Loctran123/vietnamese-evidence-corpus-embeddings-e5-large)
để xây và đánh giá hệ thống retrieval:

```text
claim
  ├── BM25S top 100
  └── multilingual-e5-large + FAISS top 100
                ↓ RRF (k=60)
             hybrid top 50
                ↓ BAAI/bge-reranker-v2-m3
             evidence top 10
```

## File

- `build_retrieval_indexes.py`: tạo exact FAISS `IndexFlatIP`, BM25S và bảng
  ánh xạ metadata; upload lên Hugging Face.
- `hybrid_retriever.py`: cung cấp lớp `HybridRetriever` và hàm
  `retrieve(claim)`.
- `evaluate_retrieval.py`: so sánh BM25, dense, hybrid và hybrid+reranker.
- `kaggle_hybrid_retrieval.ipynb`: notebook Kaggle end-to-end.

Index mặc định được lưu lâu dài tại:
`Loctran123/vietnamese-evidence-retrieval-indexes`. Khi repo đã có
`manifest.json` hoàn chỉnh, notebook chỉ tải index về thay vì build lại.

## Cấu hình mặc định

- Dense index: `faiss.IndexFlatIP(1024)`; vector corpus/query đều chuẩn hóa L2.
- Dense query: tiền tố `query: `.
- BM25: Lucene BM25, `k1=1.5`, `b=0.75`; title lặp 2 lần rồi nối text.
- Vietnamese BM25 tokenization: Unicode word token, giữ token một ký tự/con số,
  không stemming, không bỏ stopword.
- Retrieval: 100 BM25 + 100 dense, RRF `k=60`.
- Reranker: `BAAI/bge-reranker-v2-m3`, top 50, FP16 trên GPU.
- Final retrieval: top 10, tối đa 3 chunks cho mỗi `doc_id`.

## Dùng trong Python

```python
from retrieval.hybrid_retriever import HybridRetriever

retriever = HybridRetriever("/path/to/retrieval_indexes", enable_reranker=True)
results = retriever.retrieve(
    "Chỉ số sản xuất công nghiệp tháng Ba thay đổi thế nào?",
    strategy="hybrid_rerank",
    top_k=5,
)
```

Mỗi kết quả có `row_id`, `chunk_id`, `doc_id`, metadata, nội dung và các điểm
BM25/dense/RRF/reranker liên quan.

## Đánh giá

Notebook mặc định lấy mẫu cố định 500 claim `SUPPORTED`/`REFUTED` có evidence
từ hai file `claim_01_part*.json`. Kết quả gồm:

- `summary.csv`: Recall/Hit@1,5,10,50, MRR@10, nDCG@10 và latency;
- `per_query.jsonl`: top 10 của mỗi chiến lược cho từng claim;
- `evaluation_config.json`: toàn bộ cấu hình thực nghiệm.

Đây là **silver in-corpus evaluation** trên claim sinh từ chính corpus. Không
nên dùng các con số này thay cho đánh giá test độc lập do con người gán nhãn.

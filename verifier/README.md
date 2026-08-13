# Vietnamese fact-checking verifier trên Kaggle

Thư mục này fine-tune `FacebookAI/xlm-roberta-base` để phân loại cặp
`(evidence, claim)` thành ba nhãn:

- `SUPPORTED`
- `REFUTED`
- `NOT_ENOUGH_INFO`

Notebook chạy end-to-end: `verifier/kaggle_train_verifier.ipynb`.

## Pipeline

1. Tải `data/claim_01.json` từ
   `Loctran123/vietnamese-fact-checking-claims`.
2. Chia train/validation/test theo `doc_id` với tỷ lệ 80/10/10. Một bài báo chỉ
   xuất hiện trong đúng một split để tránh leakage.
3. Fine-tune XLM-R base trên evidence chuẩn.
4. Đánh giá test theo hai chế độ:
   - `gold_evidence`: đo riêng năng lực verifier;
   - `retrieved_evidence`: dùng hybrid + BGE reranker top 5, đo end-to-end.
5. Upload split, model và báo cáo lên Hugging Face.

## Hugging Face output mặc định

- Dataset split: `Loctran123/vietnamese-fact-checking-verifier-data`
- Model: `Loctran123/vietnamese-fact-checking-verifier-xlm-roberta-base`
- Báo cáo: thư mục `evaluations/latest` trong model repo.

## File

- `prepare_verifier_data.py`: tạo split Parquet và manifest.
- `train_verifier.py`: fine-tune, chọn checkpoint có Macro-F1 validation cao nhất.
- `verifier_model.py`: inference trên claim + evidence.
- `evaluate_verifier.py`: Accuracy, Macro-F1, F1 từng nhãn và confusion matrix.

Đây là dữ liệu claim tổng hợp/silver. Kết quả không thay thế đánh giá trên một
tập test độc lập do con người gán nhãn.

# Embedding corpus bằng multilingual-e5-large trên Kaggle

Pipeline này đọc trực tiếp bộ chunk từ
[`Loctran123/vietnamese-evidence-corpus-chunked-e5-v2`](https://huggingface.co/datasets/Loctran123/vietnamese-evidence-corpus-chunked-e5-v2),
tạo passage embedding bằng
[`intfloat/multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large),
ghi Parquet theo shard và tải từng shard lên Hugging Face.

Mặc định, kết quả được ghi vào dataset riêng:
`Loctran123/vietnamese-evidence-corpus-embeddings-e5-large-v2`. Pipeline cố ý
không ghi vào dataset chunk gốc.

## Chạy trên Kaggle

1. Tạo notebook Kaggle từ file `kaggle_multilingual_e5_large.ipynb` trong thư
   mục này.
2. Trong **Notebook options**, bật **Internet** và chọn **GPU** (T4 hoặc P100).
3. Trong **Add-ons > Secrets**, thêm:
   - `HF_TOKEN`: Hugging Face write token có quyền tạo/ghi dataset đầu ra.
   - `GITHUB_TOKEN`: không bắt buộc với repo GitHub public; chỉ cần khi đổi sang
     repo private.
4. Chạy toàn bộ notebook. Mỗi shard hoàn tất được upload ngay. Nếu session Kaggle
   bị ngắt, chạy lại notebook với đúng cấu hình để tiếp tục từ shard kế tiếp.

Không đưa token vào cell, Git commit, output notebook hoặc URL remote.

## Chạy bằng CLI

```bash
python embedding/embed_e5_kaggle.py \
  --source-repo Loctran123/vietnamese-evidence-corpus-chunked-e5-v2 \
  --source-revision 3303a74a1ee6c23f231d9471d1eaa6d1a3009e12 \
  --output-repo Loctran123/vietnamese-evidence-corpus-embeddings-e5-large-v2 \
  --batch-size 24 \
  --rows-per-shard 5000
```

Smoke test 100 dòng, không upload:

```bash
python embedding/embed_e5_kaggle.py \
  --source-json data/vie/processed/corpus_v1_chunked.json \
  --output-dir embedding/smoke-output \
  --max-rows 100 \
  --no-upload
```

Nếu T4 báo thiếu VRAM, giảm `--batch-size` xuống `16` hoặc `8`.

## Định dạng đầu ra

Mỗi `data/train-xxxxx.parquet` giữ các trường định danh và metadata cần cho
retrieval, cộng thêm:

- `embedding`: vector `float32[1024]` đã chuẩn hóa L2;
- metadata Parquet: model, prefix và trạng thái chuẩn hóa;
- `embedding_config.json`: khóa cấu hình để ngăn resume nhầm model/shard size;
- `manifest.json`: chỉ xuất hiện khi toàn bộ pipeline hoàn tất.

Khi embedding claim truy vấn, phải dùng tiền tố `query: `. Corpus dùng tiền tố
`passage: ` theo đúng cách huấn luyện của E5.

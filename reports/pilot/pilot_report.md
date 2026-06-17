# Pilot Data Audit - RAG Fact-Checking Tiếng Việt

## Mục tiêu
Báo cáo này kiểm tra nhanh hai file dữ liệu pilot để chứng minh hướng RAG fact-checking đa bước khả thi trước khi mở rộng dataset.

## Evidence Corpus
- File: `data/vie/processed/fact_checking_dataset_chunked.json`
- Số bản ghi: `618`
- Phân bố nhãn: `{'TRUE': 367, 'FALSE': 251}`
- Tổng số chunks: `1145`
- Trung bình chunks/bản ghi: `1.85`
- Trung bình số từ/chunk: `204.6`
- Bản ghi chưa có chunks: `10`
- Field thiếu đáng chú ý: `{'justification': 47, 'chunks': 10}`
- Text fields nghi lỗi encoding: `82/14516`

## Claim Evaluation Set
- File: `data/vie/processed/newdata.json`
- Số bài gốc: `70`
- Số claim: `420`
- Phân bố nhãn: `{'SUPPORTED': 140, 'REFUTED': 140, 'NOT_ENOUGH_INFO': 140}`
- Field thiếu đáng chú ý: `{}`
- Text fields nghi lỗi encoding: `0/3570`

## Pipeline Pilot Đề Xuất
1. Nhập claim tiếng Việt cần kiểm chứng.
2. Truy xuất top-k evidence chunks từ corpus bằng BM25/TF-IDF hoặc embedding.
3. Verifier đọc claim + evidence và dự đoán SUPPORTED, REFUTED hoặc NOT_ENOUGH_INFO.
4. Sinh giải thích ngắn và trích dẫn evidence liên quan.
5. Đánh giá retrieval bằng Recall@k/MRR và đánh giá verdict bằng Accuracy/Macro-F1.

## Ví Dụ Demo
### Ví dụ 1
- Claim: Bài viết đề cập rằng đau ngực có thể do nhiều nguyên nhân khác nhau gây ra.
- Gold label: `SUPPORTED`
- Gold evidence: Đau ngực là triệu chứng có thể do nhiều nguyên nhân khác nhau
- Reason: Bài viết nêu rõ đau ngực là triệu chứng có thể do nhiều nguyên nhân khác nhau.

### Ví dụ 2
- Claim: Cơ quan quản lý đã chặn 500 website giả mạo các ngân hàng, tổ chức tài chính trong giai đoạn từ 4/2 đến 20/5/2021.
- Gold label: `REFUTED`
- Gold evidence: cơ quan quản lý đã chặn 100 website giả mạo các ngân hàng, tổ chức tài chính
- Reason: Bài viết nêu rõ con số là 100 website bị chặn, không phải 500 như claim đưa ra.

### Ví dụ 3
- Claim: Số vụ lừa đảo đặt vé máy bay trên mạng tăng 30% so với cùng kỳ năm trước.
- Gold label: `NOT_ENOUGH_INFO`
- Gold evidence: Công an thành phố Hà Nội khuyến cáo, vào kỳ nghỉ hè, việc đặt vé máy bay du lịch giá rẻ trên mạng đang là xu hướng lựa chọn của nhiều gia đình.
- Reason: Bài viết không cung cấp bất kỳ số liệu thống kê cụ thể nào về số vụ lừa đảo hay tỷ lệ tăng giảm so với năm trước.

### Ví dụ 4
- Claim: Tổng công ty Bưu điện Việt Nam đã cảnh báo về các fanpage mạo danh cuộc thi viết thư quốc tế UPU.
- Gold label: `SUPPORTED`
- Gold evidence: Tổng công ty Bưu điện Việt Nam vừa cảnh báo một số fanpage mạo danh cuộc thi viết thư quốc tế UPU
- Reason: Bài viết nêu rõ Tổng công ty Bưu điện Việt Nam đã đưa ra cảnh báo về các fanpage mạo danh cuộc thi viết thư quốc tế UPU.

### Ví dụ 5
- Claim: Việc sử dụng Deepfake trong bài viết được mô tả là không gây hại đến danh dự của bất kỳ ai.
- Gold label: `REFUTED`
- Gold evidence: nhằm chiếm đoạt tài sản, xúc phạm danh dự hoặc phá hoại uy tín của người khác
- Reason: Bài viết khẳng định rõ ràng rằng Deepfake được sử dụng để xúc phạm danh dự của người khác, trái ngược hoàn toàn với claim này.

### Ví dụ 6
- Claim: Tác giả đã tham gia mạng xã hội Facebook từ năm 2015.
- Gold label: `NOT_ENOUGH_INFO`
- Gold evidence: Là một trong nhiều người có tham gia vào mạng xã hội như Facebook
- Reason: Bài viết chỉ cho biết tác giả có tham gia Facebook nhưng không đề cập thời điểm bắt đầu sử dụng.

### Ví dụ 7
- Claim: Bài viết đề cập đến thông tin cho rằng chỉ cần một động tác đơn giản có thể kiểm tra được tình trạng sức khỏe tim mạch.
- Gold label: `SUPPORTED`
- Gold evidence: Có thông tin cho rằng chỉ với một động tác đơn giản, chúng ta có thể biết được tình trạng sức khỏe tim mạch của mình.
- Reason: Bài viết nêu rõ ràng thông tin về việc kiểm tra tim mạch chỉ bằng một động tác đơn giản.

### Ví dụ 8
- Claim: Trang web armoniaz.com.co là một trang web uy tín và đáng tin cậy.
- Gold label: `REFUTED`
- Gold evidence: Phát hiện trang web giả mạo/lừa đảo trực tuyến: armoniaz.com.co
- Reason: Bài viết khẳng định đây là trang web giả mạo/lừa đảo, hoàn toàn trái ngược với claim cho rằng trang web này uy tín và đáng tin cậy.

### Ví dụ 9
- Claim: Tỉnh Tây Ninh đã nhận khoản đầu tư 10 tỷ USD từ Trung Quốc.
- Gold label: `NOT_ENOUGH_INFO`
- Gold evidence: về việc đề nghị xử lý thông tin sai sự thật lan truyền trên mạng xã hội
- Reason: Bài viết chỉ đề cập việc có thông tin sai sự thật lan truyền nhưng không cung cấp chi tiết cụ thể về nội dung thông tin đó hay xác nhận/bác bỏ khoản đầu tư 10 tỷ USD.

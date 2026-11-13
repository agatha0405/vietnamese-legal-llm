# Báo cáo dataset hỏi đáp pháp luật giao thông đường bộ

Bộ dữ liệu gồm 500 cặp hỏi đáp tiếng Việt cho Qwen2.5-3B-Instruct. Mốc pháp lý và ngày thu thập là **13/11/2026** theo giờ Việt Nam. Mỗi đáp án dẫn đến một điều hoặc khoản và kèm đoạn văn bản làm căn cứ.

## Nguồn và rà soát hiệu lực

| Văn bản | Bản dùng để trích và đối chiếu | Số mẫu |
| --- | --- | ---: |
| Luật Trật tự, an toàn giao thông đường bộ 36/2024/QH15 | Văn bản hợp nhất 55/VBHN-VPQH. Áp dụng sửa đổi của Luật 118/2025/QH15 và ngày hiệu lực riêng trong chú thích. [Bản hợp nhất](https://vanban.chinhphu.vn/?docid=217295&pageid=27160) · [Luật sửa đổi](https://vbpl.moj.gov.vn/bocongan/Pages/vbpq-toanvan.aspx?ItemID=185923) | 245 |
| Luật Đường bộ 35/2024/QH15 | Văn bản hợp nhất 49/VBHN-VPQH và ngày hiệu lực theo điều khoản. Bản 144/2026/VBHN-LQ-VPQH có tích hợp sửa đổi của Luật 24/2026/QH16 từ 01/03/2027. Điều 29 được giữ ngoài tập và Điều 5 về quy hoạch được loại khỏi phạm vi trích. [Bản dùng để trích](https://vanban.chinhphu.vn/?docid=217234&pageid=27160) · [Bản hợp nhất tháng 9](https://congbao.chinhphu.vn/van-ban/van-ban-hop-nhat-so-144-2026-vbhn-lq-vpqh-470557.htm) · [Luật 24/2026/QH16](https://vanban.chinhphu.vn/?docid=219356&pageid=27160) · [Luật Quy hoạch](https://vanban.chinhphu.vn/?docid=216547&pageid=27160) | 145 |
| Nghị định 165/2024/NĐ-CP | Văn bản hợp nhất 65/2026/VBHN-NĐ-BXD, gồm sửa đổi theo Nghị định 243/2025/NĐ-CP và 241/2026/NĐ-CP. [Bản hợp nhất](https://vanban.chinhphu.vn/?docid=219312&pageid=27160) · [Nghị định 241/2026/NĐ-CP](https://vanban.chinhphu.vn/?docid=218631&pageid=27160) | 100 |
| Thông tư 38/2024/TT-BGTVT | Trích từ PDF gốc ở các khoản ngoài Điều 10 đã được sửa bởi Điều 26 Thông tư 09/2025/TT-BXD. Bản hợp nhất 74/2026/VBHN-TT-BXD xác nhận chuỗi sửa đổi. Dự thảo tháng 11 đang lấy ý kiến tại ngày chốt. [Thông tư gốc](https://vanban.chinhphu.vn/?docid=211873&pageid=27160) · [Thông tư sửa đổi](https://datafiles.chinhphu.vn/cpp/files/vbpq/2025/7/09-bxd.signed.pdf) · [Bản hợp nhất](https://congbao.chinhphu.vn/van-ban/van-ban-hop-nhat-so-74-2026-vbhn-tt-bxd-470578/68818.htm) · [Hồ sơ dự thảo](https://moc.gov.vn/vn/pages/chitietduthao.aspx?iDuThao=1155) | 10 |

Nghị định 168/2024/NĐ-CP đã được sửa bởi Nghị định 238/2026/NĐ-CP có hiệu lực từ 15/08/2026. Nghị định 151/2024/NĐ-CP đã được sửa bởi Nghị định 184/2025/NĐ-CP và 236/2026/NĐ-CP. Bản PDF ký số của 168, 238 và 236 tải được nhưng các trang nội dung là ảnh quét. Dataset hiện giới hạn ở bốn nguồn trong bảng. Để mở rộng sang mức phạt và các quy định hướng dẫn của Nghị định 151, cần bản chính thức có lớp văn bản hoặc bản chép đã đối chiếu từng trang, đồng thời cần hợp nhất đúng các điều khoản sửa đổi. [Nghị định 168](https://vanban.chinhphu.vn/?docid=212167&pageid=27160) · [Nghị định 238](https://vanban.chinhphu.vn/?docid=218613&pageid=27160) · [Nghị định 236](https://vanban.chinhphu.vn/?docid=218618&pageid=27160)

Một số trang lịch sử hiệu lực của VBPL và đường tải Công báo trả lỗi khi tra cứu. Quy trình dùng bốn PDF có thể tải từ Cổng Thông tin điện tử Chính phủ và cố định SHA-256. Điều 35 của Luật 36 được loại khỏi phạm vi để tránh trộn các mốc áp dụng thiết bị 2028–2029. Hai điều có bảng trong Thông tư 38 được loại khỏi bộ trích văn bản tuyến tính. [Cấu hình nguồn và phạm vi](configs/dataset.toml)

## Xây dựng và kiểm chứng

- Tách điều, khoản từ lớp văn bản của PDF. Giữ nội dung các điểm trong khoản, bỏ số trang và chỉ số chú thích khỏi đáp án, rồi chuẩn hóa NFC và khoảng trắng. Giữ dấu tiếng Việt, số liệu, ngoại lệ và dấu câu. Mỗi mẫu lưu SHA-256, phiên bản văn bản, trang PDF và chú thích đã dùng để xác định hiệu lực.
- Dùng 78 câu hỏi viết riêng và 422 câu hỏi tra cứu dựng từ tên điều và vị trí khoản. Đáp án gồm lời dẫn căn cứ và đoạn trích từ PDF. Mỗi đoạn được sử dụng một lần. Loại đoạn rỗng, lỗi trích xuất, hết hiệu lực, có ngày áp dụng sau mốc chốt hoặc vượt ngân sách 1.024 token.
- Gắn `verified` theo phép đối chiếu trích xuất với PDF chính thức và thông tin hiệu lực đã rà soát. `effective_date` là ngày hiệu lực của phiên bản đoạn trích, dựa trên hiệu lực chung, ngoại lệ hoặc chú thích sửa đổi. Mức kiểm chứng này xác nhận truy vết và đối chiếu văn bản. Dataset chưa trải qua một vòng thẩm định pháp lý độc lập.
- Chia theo nhóm văn bản và điều, gộp các điều liên quan đã chỉ định trong cấu hình và các đoạn có Jaccard ≥ 0,80 trên cụm ba đơn vị chữ hoặc số liên tiếp. Seed 42 và phân bổ nguyên nhóm cho tỷ lệ 80/10/10. Điểm chọn split xét phân phối nguồn, chủ đề và câu hỏi viết riêng, có ưu tiên chủ đề thiết yếu trong tập test.
- Áp dụng chat template chính thức của revision Qwen đã ghim. Mỗi split chứa `prompt`, `completion`, `input_ids` và `completion_mask`. Mask giữ nội dung assistant và token kết thúc lượt. Lưu thông tin pháp lý đầy đủ trong master và nối với các split qua `id`.

## Chạy và kiểm tra

| Tập | Số mẫu | Số nhóm | Câu hỏi viết riêng |
| --- | ---: | ---: | ---: |
| Train | 400 | 181 | 59 |
| Validation | 50 | 26 | 9 |
| Test | 50 | 23 | 10 |
| Tổng | 500 | 230 | 78 |

Có 500 mẫu `verified` và 0 mẫu `unverified` trong tập pháp luật. Phân bố có 13 chủ đề. Kết cấu hạ tầng chiếm 134 mẫu, quy tắc lưu thông chiếm 74 mẫu và phương tiện chiếm 55 mẫu. Trung vị câu hỏi và đáp án lần lượt là 28 và 76 đơn vị tách theo khoảng trắng. Chuỗi hội thoại có trung vị 237 token và dài nhất 807 token. Không phát hiện giao nhau giữa các nhóm hoặc đoạn gần trùng giữa các split theo phép kiểm đã mô tả. [Thống kê thực tế](data/processed/statistics.json)

Các đầu ra gồm [master có metadata](data/processed/legal_qa_master.jsonl), [train](data/processed/train.jsonl), [validation](data/processed/validation.jsonl), [test](data/processed/test.jsonl) và [12 câu ngoài phạm vi](data/processed/out_of_scope_test.jsonl).

Tại thư mục gốc dự án, dùng môi trường Python của dự án.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-analysis.txt
.\.venv\Scripts\python.exe -X utf8 scripts/prepare_legal_dataset.py --download --verify-only
```

Nếu cần dựng lại bộ dữ liệu, bỏ `--verify-only` và chỉ định thư mục đầu ra mới bằng `--output`. Nếu tokenizer đã nằm trong cache của dự án, đặt `$env:HF_HUB_CACHE = Join-Path $PWD 'artifacts/hf-cache'` và thêm `--offline` để dùng đúng revision trong cache. `--download` tải lại PDF còn thiếu sau khi đối chiếu SHA-256.

Notebook hiển thị số lượng, phân phối topic, histogram độ dài và ví dụ trong train. Script xác minh lại JSONL trên đĩa, tính duy nhất của ID và câu hỏi, nội dung bắt buộc, căn cứ, tính rời nhau của các nhóm, chuỗi token Qwen và nhãn thực tế do collator của TRL 0.22.2 tạo ra. Dùng `SFTConfig(max_length=1024, completion_only_loss=True, assistant_only_loss=False, packing=False)` để giữ mask đã kiểm tra. [Notebook](notebooks/dataset_preparation.ipynb)

Đã chạy pipeline dựng và xác minh lại qua notebook. Phép đối chiếu riêng với lớp văn bản PDF tìm thấy nội dung của cả 440 đoạn một trang và xác nhận thứ tự ký tự đã chuẩn hóa của 60 đoạn nhiều trang. Collator TRL kiểm tra mask trên 500 mẫu. Một batch ba mẫu đi qua DataLoader của `SFTTrainer` giữ đúng mask sau khi nạp từ train JSONL.

## Hạn chế

Phần lớn mẫu là câu hỏi tra cứu có gợi sẵn vị trí điều khoản. Điểm trên test phản ánh phạm vi tra cứu này. Phần câu hỏi viết riêng nên được báo riêng khi đánh giá. Tập ngoài phạm vi có 12 câu kiểm tra nhận diện chủ đề và dùng nhãn hành vi mong đợi, không chứa đáp án pháp lý.

Tập còn thiếu mức phạt cập nhật, các câu hỏi kết hợp nhiều nguồn và các tình huống chuyển tiếp. Cần bổ sung những phần đó cùng một tập test tình huống được rà soát độc lập trước khi dùng kết quả để kết luận về độ chính xác pháp lý trong sử dụng thực tế.

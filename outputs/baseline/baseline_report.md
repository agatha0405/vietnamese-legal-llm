# Baseline Qwen2.5-3B-Instruct

Đã sinh câu trả lời cho 50 câu hỏi trên NVIDIA GeForce RTX 3050 Ti Laptop GPU với NF4 FP16, greedy decoding và giới hạn 256 token mới.

| Chỉ số | Giá trị trung bình |
| --- | ---: |
| ROUGE-L | 0.228305 |
| Token F1 | 0.308649 |
| Exact Match | 0.000000 |
| Thời gian sinh (giây/câu) | 58.290 |

| Loại câu hỏi | Số mẫu | ROUGE-L | Token F1 | Exact Match | Giây/câu |
| --- | ---: | ---: | ---: | ---: | ---: |
| Viết riêng | 10 | 0.2118 | 0.3113 | 0.0000 | 57.66 |
| Tra cứu | 40 | 0.2324 | 0.3080 | 0.0000 | 58.45 |

Có 43 câu chạm giới hạn sinh. Các điểm tự động đo độ khớp với câu tham chiếu. Đánh giá pháp lý độc lập đang ở trạng thái `unverified`.

Xem [rubric và quy trình](../../docs/baseline_evaluation.md) và [phiếu chấm](legal_review_sheet.csv).

## Sáu ví dụ đối chiếu với căn cứ

### Tốt một phần · vn-road-55-80-2

**Câu hỏi**  
Người có mặt tại nơi xảy ra tai nạn giao thông đường bộ có những trách nhiệm gì?

**Nhận xét sơ bộ**  
Nêu được việc báo tin, hỗ trợ người bị nạn và cung cấp thông tin. Dẫn sai căn cứ, thiếu trách nhiệm bảo vệ hiện trường và chưa nêu đầy đủ việc bảo vệ tài sản theo khoản 2 Điều 80.

[Trang PDF của căn cứ](https://datafiles.chinhphu.vn/cpp/files/vbpq/2026/3/55-vbhn-vpqh.pdf#page=65) · [Toàn bộ dự đoán](baseline_predictions.csv)

### Tốt một phần · vn-road-65-66-1

**Câu hỏi**  
Hệ thống quản lý, điều hành giao thông đường cao tốc được thiết lập để hỗ trợ những hoạt động nào?

**Nhận xét sơ bộ**  
Nhận diện được chức năng quản lý, điều hành và cung cấp thông tin giao thông của hệ thống cao tốc. Bỏ sót vận hành khai thác, bảo trì và dẫn sai căn cứ.

[Trang PDF của căn cứ](https://datafiles.chinhphu.vn/cpp/files/vbpq/2026/8/65-vbhn-bxd.signed.pdf#page=82) · [Toàn bộ dự đoán](baseline_predictions.csv)

### Tốt một phần · vn-road-49-7-4

**Câu hỏi**  
Luật Đường bộ nghiêm cấm những hành vi nào đối với báo hiệu đường bộ?

**Nhận xét sơ bộ**  
Nhận diện được nhóm hành vi che khuất và làm sai lệch báo hiệu. Dẫn sai văn bản, thiếu một số hành vi như tháo dỡ, di chuyển trái quy định và gắn nội dung không liên quan.

[Trang PDF của căn cứ](https://datafiles.chinhphu.vn/cpp/files/vbpq/2026/3/49-vbhn-vpqh.pdf#page=5) · [Toàn bộ dự đoán](baseline_predictions.csv)

### Chưa tốt · vn-road-55-15-1

**Câu hỏi**  
Những thao tác nào được coi là chuyển hướng xe?

**Nhận xét sơ bộ**  
Diễn giải chung về thay đổi hướng và góc quay. Bỏ ba thao tác được khoản 1 Điều 15 liệt kê là rẽ trái, rẽ phải và quay đầu xe, đồng thời viện dẫn luật năm 2008.

[Trang PDF của căn cứ](https://datafiles.chinhphu.vn/cpp/files/vbpq/2026/3/55-vbhn-vpqh.pdf#page=14) · [Toàn bộ dự đoán](baseline_predictions.csv)

### Chưa tốt · vn-road-55-26-1

**Câu hỏi**  
Xe cơ giới và xe thô sơ phải sử dụng đèn hoặc vật phát sáng thế nào trong hầm đường bộ?

**Nhận xét sơ bộ**  
Yêu cầu bật đèn pha và viện dẫn sai điều luật. Khoản 1 Điều 26 phân biệt đèn chiếu gần đối với xe cơ giới, xe máy chuyên dùng và đèn hoặc vật phát sáng đối với xe thô sơ.

[Trang PDF của căn cứ](https://datafiles.chinhphu.vn/cpp/files/vbpq/2026/3/55-vbhn-vpqh.pdf#page=19) · [Toàn bộ dự đoán](baseline_predictions.csv)

### Chưa tốt · vn-road-55-80-3

**Câu hỏi**  
Khi nào có thể dùng phương tiện liên quan đến tai nạn để chở nạn nhân đi cấp cứu, và phải bảo vệ hiện trường thế nào?

**Nhận xét sơ bộ**  
Cho phép dùng phương tiện liên quan đi cấp cứu nhưng bỏ điều kiện chỉ được dùng khi không có phương tiện khác. Thiếu yêu cầu xác định vị trí phương tiện và nạn nhân tại hiện trường.

[Trang PDF của căn cứ](https://datafiles.chinhphu.vn/cpp/files/vbpq/2026/3/55-vbhn-vpqh.pdf#page=65) · [Toàn bộ dự đoán](baseline_predictions.csv)

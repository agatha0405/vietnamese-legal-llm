# Demo hỏi đáp pháp luật giao thông

Demo dùng Gradio 6.9.0 và nạp một bản Qwen2.5-3B-Instruct NF4 cùng adapter PEFT. Base tắt adapter trong phạm vi lượt sinh. Chế độ So sánh sinh tuần tự cho hai mô hình trên cùng ngữ cảnh. Lịch sử nằm trong phiên Gradio (`list[dict]` / chuỗi văn bản) được trích xuất qua `extract_content()` và có thể xóa bằng nút xóa hội thoại. Giao diện hiển thị thời gian sinh của từng mô hình. Khi lịch sử vượt 1.024 token, các lượt cũ nhất được bỏ khỏi prompt. Câu hỏi hiện tại vượt giới hạn được yêu cầu rút gọn.

Demo tra điều khoản trong [kho nguồn đã đối chiếu](legal_sources.toml) theo hành vi, loại xe, tình tiết và ngày áp dụng (gồm 6 văn bản hiện hành với `max_article`, 13 chủ đề / 28 quy tắc phạt từ `Nghị định 168/2024/NĐ-CP` và `238/2026/NĐ-CP`, 3 quy định bổ sung từ `Thông tư 38/2024` và `Điều 57, 58 Luật 36/2024/QH15`, cùng 11 định nghĩa trọng tâm). Cách hỏi ngắn như “phạt như nào”, “phạt ra sao” và câu hỏi về tiền trong ngữ cảnh giao thông đi vào cùng quy trình; ứng dụng hỗ trợ chuỗi làm rõ nhiều lượt (hỏi loại xe, hành vi vi phạm cụ thể khi chỉ nêu hậu quả tai nạn, hoặc đổi xe ở lượt kế tiếp), cô lập ngữ cảnh khi chuyển sang chủ đề độc lập, và kiểm chứng căn cứ giả / văn bản hết hiệu lực qua `check_citation()` (ví dụ `Điều 999` hay `Nghị định 100/2019/NĐ-CP`). Với trường hợp khớp, mặc định giao diện hiển thị ngắn gọn mức phạt và điều kiện chính; điểm trừ giấy phép lái xe, điều khoản căn cứ, đường dẫn văn bản và mốc áp dụng (mặc định `14/11/2026`) được hiển thị khi người dùng hỏi thêm hoặc hỏi nguồn. Base, Fine-tuned và So sánh dùng chung câu trả lời tra cứu. Nếu đã đủ hành vi cụ thể nhưng thiếu căn cứ đáng tin cậy, demo chỉ báo chưa có căn cứ đã đối chiếu. Câu hỏi ghi rõ “hiện nay” dùng ngày hiện tại theo giờ Việt Nam và cần nguồn đã rà soát đến ngày đó.

Các câu luật chung còn lại được `LegalIndex.augment_messages()` truy xuất tối đa 2 đoạn quy phạm đã kiểm chứng từ `data/processed/legal_qa_master.jsonl` và mục định nghĩa (chỉ nối lịch sử khi có dấu hiệu hồi chỉ `ANAPHORA` và lọc đúng chủ thể định nghĩa, tránh trộn ngữ cảnh cũ khi đổi chủ đề) rồi đi qua mô hình với prompt ngắn, giới hạn 96 token mới, chặn lặp cụm 6 token cho sinh tự do và tắt chặn lặp trên prompt khi có đoạn tham chiếu kiểm chứng để giữ nguyên câu chữ pháp lý. Bộ lọc đầu ra (`filter_output()`) chặn thêm các mẫu số tiền và số điểm chưa kiểm chứng khi lượt sinh không có căn cứ, đồng thời lược trích dẫn tự phát khi người dùng không yêu cầu nguồn. Bộ nhận diện và bộ lọc đều dựa vào mẫu văn bản, nên phạm vi nhận diện hữu hạn. `--legacy` dành cho tái hiện demo ban đầu, bao gồm lỗi ở đầu ra chưa kiểm chứng. Phép đánh giá QLoRA 50 mẫu dùng các script đánh giá gốc và độc lập với demo. [Báo cáo kiểm thử demo](../reports/DEMO_TEST_REPORT.md) · [Kiểm thử tra cứu mức phạt](../outputs/evaluation/chatbot-sanctions-20261114/README.md)

## Chạy local

Sau khi clone, [cài môi trường](../README.md#cài-môi-trường-trên-windows) và [giải nén adapter](../README.md#chạy-demo-trên-windows). Từ thư mục gốc, chạy lệnh sau.

```powershell
$env:HF_HUB_CACHE = Join-Path $PWD 'artifacts/hf-cache'
.\.venv\Scripts\python.exe -X utf8 app/app.py --adapter runs/qwen3b-r8-local/final_adapter
```

Mở [giao diện local](http://127.0.0.1:7860). Có thể thêm `--offline` sau khi cache đã có đủ revision đã ghim.

Trong lượt đo RTX 3050 Ti Laptop, nhóm dùng `--cpu-threads 1`. Số luồng là thiết lập tùy chọn của tiến trình. Xem [cấu hình và kết quả hồi quy](../outputs/evaluation/chatbot-quality-20261114/) để đối chiếu.

## Chạy trong notebook

Dùng cell demo cuối [notebook huấn luyện QLoRA](../notebooks/qlora_finetuning_colab.ipynb) sau khi có adapter. Cell dùng giao diện inline, `share=False`. Nếu Colab yêu cầu liên kết công khai để truy cập giao diện, dùng lệnh local ở trên. Phép thử dòng lệnh vẫn chạy được trong notebook.

## Kịch bản kiểm thử

| Kịch bản | Cách quan sát |
| --- | --- |
| Thông thường | Hỏi về một quy tắc giao thông và kiểm tra câu trả lời có nội dung, thời gian dương. |
| Dài | Dán câu hỏi vượt 1.024 token và kiểm tra thông báo trước khi sinh. |
| Ngoài phạm vi | Hỏi về thời tiết. Ghi lại hành vi, rồi chấm theo rubric nếu có người chấm. |
| Thiếu dữ kiện | Hỏi về một tình huống chưa nêu phương tiện hoặc bối cảnh. Ghi lại việc mô hình hỏi thêm hay tự giả định. |
| Nhiều lượt | Đặt câu hỏi nối tiếp dùng đại từ, sau đó xóa lịch sử và thử lại. |
| Chuyển mô hình | Gửi cùng câu hỏi trong Base, Fine-tuned và So sánh. Kiểm tra chế độ, thứ tự và thời gian. |
| Lỗi mô hình | Kiểm tra thông báo khi adapter thiếu, hoặc mô phỏng lỗi CUDA trong test cô lập. |

Demo đã được kiểm tra trên API Gradio local với cả hai mô hình. Câu trả lời và căn cứ do mô hình sinh cần được đối chiếu với nguồn chính thức. [Giới hạn và kết quả đánh giá](../docs/technical_report.md#giới-hạn-và-đánh-giá-pháp-lý)

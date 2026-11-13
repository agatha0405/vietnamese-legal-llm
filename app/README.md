# Demo hỏi đáp pháp luật giao thông

Demo dùng Gradio 6.9.0 và nạp một bản Qwen2.5-3B-Instruct NF4 cùng adapter PEFT. Base tắt adapter trong phạm vi lượt sinh. Chế độ So sánh sinh tuần tự cho hai mô hình trên cùng ngữ cảnh. Lịch sử nằm trong phiên Gradio và có thể xóa bằng nút xóa hội thoại. Giao diện hiển thị thời gian sinh của từng mô hình. Khi lịch sử vượt 1.024 token, các lượt cũ nhất được bỏ khỏi prompt. Câu hỏi hiện tại vượt giới hạn được yêu cầu rút gọn.

## Chạy local

Sau khi clone, [cài môi trường](../README.md#cài-môi-trường-trên-windows) và [giải nén adapter](../README.md#chạy-demo-trên-windows). Từ thư mục gốc, chạy lệnh sau.

```powershell
$env:HF_HUB_CACHE = Join-Path $PWD 'artifacts/hf-cache'
.\.venv\Scripts\python.exe -X utf8 app/app.py --adapter runs/qwen3b-r8-local/final_adapter
```

Mở [giao diện local](http://127.0.0.1:7860). Có thể thêm `--offline` sau khi cache đã có đủ revision đã ghim.

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

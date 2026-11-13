# Vietnamese Legal LLM QLoRA

## Trạng thái

| Hạng mục | Trạng thái | Bằng chứng |
| --- | --- | --- |
| Kiểm tra môi trường | Hoàn thành | T4, NF4, forward/backward, optimizer và adapter roundtrip. [Báo cáo](docs/environment_setup.md) |
| Chuẩn bị dữ liệu | Hoàn thành | 500 mẫu, split 400/50/50. [Báo cáo](DATASET_REPORT.md) |
| Đánh giá baseline | Hoàn thành | 50 dự đoán thật, ROUGE-L 0,228305, Token F1 0,308649, EM 0/50. [Kết quả](outputs/baseline/baseline_metrics.json) |
| Huấn luyện QLoRA | Hoàn thành | Rank 8, 2 epoch, 100 bước. Validation loss 0,90517, loss bước cuối 0,7475. Adapter nạp lại và sinh thành công. |
| Đánh giá adapter | Hoàn thành | 50/50 ID đúng thứ tự. ROUGE-L 0,359918, Token F1 0,400965, EM 0/50. [Báo cáo](outputs/evaluation/qwen3b-r8/results/comparison_report.md) · [Phân tích 10 ca](reports/ERROR_ANALYSIS.md) |
| Demo Gradio | Hoàn thành demo nghiên cứu | API sinh được cả Base và Fine-tuned, xử lý câu nối tiếp và chặn prompt quá dài. Mô hình chưa xử lý phù hợp ca ngoài phạm vi và thiếu dữ kiện. |
| Tài liệu và tái lập | Hoàn thành tài liệu và kiểm thử local | [Hướng dẫn khôi phục](RESUME_INSTRUCTIONS.md) · [Báo cáo học thuật](reports/REPORT_DRAFT.md) |

Baseline chưa qua chấm pháp lý độc lập.

## Điểm khôi phục

Run cục bộ `runs/qwen3b-r8-local`. Đọc bước mới nhất bằng lệnh `train_qlora.py --run-dir runs/qwen3b-r8-local --status` trong [hướng dẫn](RESUME_INSTRUCTIONS.md). Mỗi checkpoint hợp lệ có `complete.json` và `trainer_state.json`. Tệp `attempt-*.json` cho biết lần chạy nào đã tiếp tục từ checkpoint nào. Nếu `final_adapter` hợp lệ, chạy đánh giá adapter.

Đã kiểm tra 50 ID, manifest SHA-256, tổng hợp chỉ số từ CSV, biểu đồ và phiếu pháp lý 100 hàng. Notebook so sánh đã chạy với kết quả thực. Adapter cuối và chỉ số train nằm trong ZIP bàn giao. [Gói adapter và log](outputs/finetuning/qwen3b-r8-step100.zip)

Đã kiểm tra tokenizer và completion mask cho 400 mẫu train, 50 validation. Chuỗi dài nhất lần lượt 807 và 509 token. Toàn bộ 504 tensor LoRA thay đổi từ checkpoint-1 đến checkpoint-5 và đều hữu hạn. Validation loss ở bước 10 là 1,04089. Run trên RTX 3050 Ti Laptop 4 GB tiếp tục với `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.75,max_split_size_mb:128`. Cấu hình này thay đổi cấp phát bộ nhớ, giữ nguyên dữ liệu và lịch train. Nếu cần chuyển Colab, dùng checkpoint mới nhất và notebook với Drive theo hướng dẫn. Kiểm tra PID hiện tại trong `.writer/owner.json` trước khi chạy tiến trình khác.

## Công việc còn lại

Đã sinh đủ 24 đáp án bổ sung cho 12 câu ngoài phạm vi. Adapter trả lời câu đăng ký kết hôn bằng cách viện dẫn Luật Đường bộ. Giữ phiếu pháp lý ở trạng thái `unverified` cho đến khi có người chấm độc lập. Rank 16 được để lại do chi phí GPU. Đường Colab đã kiểm tra cấu trúc notebook và gói nguồn, còn mount Drive và chạy lại trên Colab chưa được xác minh trong phiên này. [Phiếu ngoài phạm vi](outputs/evaluation/refusal-qwen3b-r8/results/refusal_review.csv)

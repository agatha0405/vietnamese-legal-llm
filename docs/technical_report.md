# Vietnamese Legal Question Answering với Qwen2.5-3B-Instruct

## Bài toán

Đầu vào là một câu hỏi tiếng Việt về giao thông đường bộ. Đầu ra là câu trả lời nêu kết luận, điều kiện, ngoại lệ và căn cứ đến điều khoản ở mốc 13/11/2026. Thí nghiệm đo tác động của supervised fine-tuning lên cùng mô hình nền, cùng test, cùng prompt và cùng thiết lập sinh.

## Mô hình

Qwen2.5-3B-Instruct là Transformer decoder-only sinh token tự hồi quy. Bản 3B có 36 lớp và dùng grouped-query attention với 16 đầu query và 2 đầu key/value. Tokenizer chính thức áp dụng chat template cho các lượt system, user và assistant. [Model card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

LoRA giữ trọng số nền và học hai ma trận hạng thấp để xấp xỉ phần cập nhật trọng số. QLoRA giữ các lớp tuyến tính nền ở NF4, lượng tử hóa tiếp các hằng số lượng tử hóa và lan truyền gradient qua mô hình đã đóng băng đến adapter. Run chính dùng FP16, LoRA rank 8, alpha 16, dropout 0,05 và gradient checkpointing. Loss dùng token nội dung assistant và token kết thúc lượt. [LoRA](https://arxiv.org/abs/2106.09685) · [QLoRA](https://arxiv.org/abs/2305.14314) · [TRL SFT](https://huggingface.co/docs/trl/v0.22.2/sft_trainer)

## Dữ liệu và thiết kế thực nghiệm

500 cặp hỏi đáp có căn cứ ở bốn văn bản chính thức, chia 400 train, 50 validation và 50 test theo nhóm điều khoản với seed 42. Test có 40 câu tra cứu và 10 câu viết riêng. Đối chiếu SHA-256 trước mỗi lượt. Trong suy luận chỉ đưa system prompt và câu hỏi cho mô hình. Chạy greedy, tối đa 256 token, batch 1. Chọn adapter ở bước cuối đã quy định trước. [Báo cáo dataset](../DATASET_REPORT.md) · [Cấu hình baseline](../configs/inference.toml) · [Cấu hình QLoRA](../configs/training.toml)

## Kết quả hiện có

| Mô hình | Số mẫu | ROUGE-L | Token F1 | Exact Match | Giây/câu |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-3B-Instruct NF4 | 50 | 0,228305 | 0,308649 | 0/50 | 58,29 |

Đã xác nhận 504 tensor LoRA thay đổi từ checkpoint-1 đến checkpoint-5. Sau lỗi VRAM, thay đổi allocator CUDA đã cho phép tiếp tục từ checkpoint-10. Theo dõi số liệu huấn luyện và kết quả adapter tại trang trạng thái. [Báo cáo baseline](../outputs/baseline/baseline_report.md) · [Trạng thái thực nghiệm](../PROJECT_PROGRESS.md)

## Giới hạn và đánh giá pháp lý

Ba chỉ số khớp văn bản phản ánh mức trùng với câu tham chiếu. Chấm độc lập cần đối chiếu tính đúng nội dung, đúng căn cứ, hallucination và từ chối. Bộ 50 câu hiện có chủ yếu là câu tra cứu. Khả năng từng gặp văn bản công khai trong pretraining chưa xác định. Cần giữ cùng môi trường để diễn giải chênh lệch độ trễ. [Rubric](baseline_evaluation.md)

## Tái lập

Xem [lệnh kiểm tra và tiếp tục](../RESUME_INSTRUCTIONS.md) cho các bước huấn luyện, đánh giá và demo. Mỗi checkpoint lưu trạng thái cần cho Trainer tiếp tục sau gián đoạn.

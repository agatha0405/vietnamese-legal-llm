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
| Qwen2.5-3B-Instruct + QLoRA rank 8 | 50 | 0,359918 | 0,400965 | 0/50 | 27,08 |

Adapter hoàn tất hai epoch và 100 bước với 14.966.784 tham số trainable. Validation loss cuối là 0,90517. Cả hai lượt suy luận dùng RTX 3050 Ti Laptop với cấu hình bộ cấp phát CUDA khác nhau, nên thời gian được báo riêng. [Gói adapter và số liệu huấn luyện](../outputs/finetuning/qwen3b-r8-step100.zip) · [So sánh mô hình](../outputs/evaluation/qwen3b-r8/results/comparison_report.md)

## Giới hạn và đánh giá pháp lý

Ba chỉ số khớp văn bản phản ánh mức trùng với câu tham chiếu. Adapter vẫn có dự đoán dẫn sai điều hoặc trả lời ngoài phạm vi. Bộ 50 câu chủ yếu là câu tra cứu. Khả năng từng gặp văn bản công khai trong pretraining chưa xác định. Chấm pháp lý độc lập chưa thực hiện. [Rubric](baseline_evaluation.md) · [Phân tích lỗi](../outputs/evaluation/qwen3b-r8/error_analysis.csv)

## Tái lập

Xem [hướng dẫn cài đặt và chạy](../README.md#5-hướng-dẫn-cài-đặt-và-chạy) cho các bước huấn luyện, đánh giá và demo.

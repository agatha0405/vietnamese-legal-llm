# Đánh giá baseline Qwen2.5-3B-Instruct

Baseline dùng đúng 50 mẫu test đã ghim. Script kiểm tra SHA-256 của master và cả ba split, nối bản ghi theo ID và đối chiếu prompt, câu tham chiếu và nhóm nội dung. Khi sinh, mô hình chỉ nhận system prompt và câu hỏi có sẵn. Prompt tra cứu nào đã có số điều trong test vẫn giữ nguyên.

## Chạy đánh giá

Chạy từ thư mục gốc với môi trường Python của dự án đã cài `requirements.txt`.

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/evaluate_baseline.py --check-only
$env:HF_HUB_CACHE = Join-Path $PWD 'artifacts/hf-cache'
.\.venv\Scripts\python.exe -X utf8 scripts/evaluate_baseline.py
```

Dùng `--offline` khi cache có đủ các shard tại revision đã ghim. Script lưu tiến độ sau mỗi câu và tiếp tục từ ID kế tiếp khi chạy lại với cùng cấu hình và môi trường. Dùng `--output` với thư mục mới cho một lượt đo độc lập. Trên Colab, chọn T4, mở notebook và tải ZIP mã nguồn ở cell đầu. Gói chứa cả bốn tệp dữ liệu đã ghim để tự kiểm tra split. [Notebook đánh giá baseline](../notebooks/baseline_evaluation.ipynb) · [ZIP nguồn Colab](../artifacts/baseline_colab_source.zip)

## Điều kiện so sánh với adapter

Giữ cùng tệp test, thứ tự câu hỏi, revision của Qwen và tokenizer, chat template, system prompt, NF4 double quantization FP16, SDPA, greedy decoding, batch size 1, token kết thúc và giới hạn 256 token mới. Thời gian đo đồng bộ CUDA quanh `generate()` sau một lượt warm-up và báo đơn vị giây/câu. So sánh thời gian giữa baseline và adapter trên cùng GPU và cùng bộ thư viện. Kết quả đo không bao gồm tải mô hình, token hóa và decode. Kết quả lưu GPU và phiên bản thư viện thực tế. [Cấu hình dùng lại](../configs/inference.toml) · [Model card Qwen](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

## Chỉ số tự động

Chuẩn hóa NFC và casefold. Token là chuỗi ký tự chữ hoặc số Unicode liên tiếp. Dấu tiếng Việt được giữ nguyên. ROUGE-L dùng F1 của dãy con chung dài nhất, Token F1 dùng giao đa tập token, Exact Match so sánh hai dãy token sau chuẩn hóa. Lấy trung bình trên từng mẫu, đồng thời báo riêng hai nhóm câu hỏi viết riêng và tra cứu. Các chỉ số này đo độ khớp văn bản. [ROUGE](https://aclanthology.org/W04-1013/)

## Chấm pháp lý

Hai người chấm đọc câu hỏi, câu trả lời và căn cứ chính thức tại mốc 13/11/2026, rồi ghi nhận định độc lập. Với trường hợp có tranh luận, ghi căn cứ đến điều, khoản, điểm và giải quyết bất đồng trước khi tổng hợp. Khi so sánh Base và Fine-tuned, ẩn danh mô hình và xáo thứ tự dự đoán.

| Tiêu chí | Cách chấm |
| --- | --- |
| Legal factual correctness | `2` nếu các kết luận, điều kiện và ngoại lệ trọng yếu đều đúng. `1` nếu thiếu chi tiết nhưng kết luận chính vẫn đúng. `0` nếu kết luận chính sai hoặc bỏ điều kiện làm đổi kết luận. |
| Citation correctness | `2` nếu dẫn đúng và đủ căn cứ trọng yếu. `1` nếu căn cứ chính đúng nhưng thiếu khoản, điểm hoặc thiếu một phần căn cứ. `0` nếu thiếu căn cứ hoàn toàn hoặc dẫn sai căn cứ chính. |
| Hallucination | `yes` khi có khẳng định pháp lý trọng yếu trái nguồn hoặc không tìm thấy căn cứ sau khi đối chiếu. `no` khi không phát hiện. Ghi cụm từ và chứng cứ trong phiếu. |
| Appropriate refusal | `yes` nếu từ chối hoặc yêu cầu làm rõ đúng khi câu hỏi ngoài phạm vi hoặc thiếu dữ kiện trọng yếu. `no` nếu từ chối câu hỏi đủ dữ kiện hoặc khẳng định khi cần làm rõ. `n/a` nếu trả lời trực tiếp phù hợp. |

Phiếu chấm tạo từ đầu ra thực tế. Các ô nhãn để trống và trạng thái `unverified` cho tới khi có lượt chấm độc lập. Bộ 50 mẫu chỉ gồm câu hỏi trong phạm vi nên khả năng từ chối ngoài phạm vi cần được đo riêng nếu thực hiện sau này.

## Phạm vi kiểm tra rò rỉ

Kiểm tra đủ 500 ID thuộc đúng split, không có nhóm, câu hỏi chuẩn hóa hoặc đoạn căn cứ trùng giữa các split. Phép kiểm gần trùng khi chuẩn bị dữ liệu cũng được chạy lại. Cặp gần trùng giữa hai điều đều nằm trong cùng split. Khả năng mô hình từng gặp văn bản công khai trong pretraining chưa xác định được từ thông tin công bố.

# Vietnamese Legal LLM QLoRA – BTL Xử lý ngôn ngữ tự nhiên

## 1. Giới thiệu

Đây là dự án bài tập lớn môn Xử lý ngôn ngữ tự nhiên của sinh viên Trường Công nghệ Thông tin – Đại học Phenikaa.

| MSSV | Họ và tên |
| --- | --- |
| 24106898 | Trương Việt Thành |
| 24100543 | Bùi Khánh Linh |
| 24100040 | Nguyễn Việt Bách |
| 24108222 | Nguyễn Văn Vinh |

Dự án tinh chỉnh `Qwen/Qwen2.5-3B-Instruct` bằng QLoRA để trả lời câu hỏi tiếng Việt về pháp luật giao thông đường bộ. Hệ thống hướng đến câu trả lời nêu căn cứ, điều kiện áp dụng và ngoại lệ. Dataset gồm 500 cặp hỏi đáp từ bốn văn bản chính thức, chia theo nhóm điều khoản thành 400 train, 50 validation và 50 test. Mốc pháp lý của dataset là 13/11/2026. [Báo cáo dataset](DATASET_REPORT.md) · [Báo cáo học thuật](reports/REPORT_DRAFT.md)

**Built with Qwen.** Gói trọng số trong repo chứa adapter được huấn luyện cho dự án. Mô hình nền sử dụng Qwen Research License. [Ghi nhận và giấy phép](outputs/finetuning/NOTICE.txt)

## 2. Kiến trúc tổng quát

- [scripts/prepare_legal_dataset.py](scripts/prepare_legal_dataset.py) và [scripts/legal_sources.py](scripts/legal_sources.py) tải và kiểm tra nguồn, trích điều khoản, chuẩn hóa tiếng Việt, chia tập theo nhóm và tạo completion mask. Dataset bàn giao nằm trong [data/processed/](data/processed/).
- [scripts/train_qlora.py](scripts/train_qlora.py) nạp mô hình nền NF4 4-bit với double quantization, huấn luyện adapter LoRA rank 8 và tính loss trên phần trả lời của assistant. Cấu hình và lịch lưu checkpoint nằm trong [configs/training.toml](configs/training.toml).
- [scripts/evaluate_baseline.py](scripts/evaluate_baseline.py) và [scripts/evaluate_adapter.py](scripts/evaluate_adapter.py) đánh giá Base và Fine-tuned trên cùng 50 câu test với chung prompt, chat template và thiết lập sinh. [Cấu hình suy luận](configs/inference.toml)
- [app/app.py](app/app.py) cung cấp giao diện Gradio. Ứng dụng nạp một mô hình nền và bật hoặc tắt adapter theo chế độ đã chọn.
- [notebooks/](notebooks/) trình bày các bước thực nghiệm. [outputs/](outputs/) chứa gói adapter, dự đoán, chỉ số và biểu đồ. [reports/](reports/) chứa báo cáo học thuật, phân tích lỗi và kiểm thử demo.

## 3. Tính năng nổi bật

- Demo có ba chế độ **Base**, **Fine-tuned** và **So sánh**. Người dùng có thể hỏi tiếp theo ngữ cảnh, xóa lịch sử và xem thời gian phản hồi của từng mô hình. [Hướng dẫn demo](app/README.md)
- Adapter đã huấn luyện đủ 2 epoch, 100 bước cập nhật với 14.966.784 tham số trainable. Validation loss cuối là 0,90517. Gói bàn giao có cấu hình, lịch sử loss và manifest SHA-256. [Gói adapter](outputs/finetuning/qwen3b-r8-step100.zip) · [Đường cong loss](outputs/evaluation/qwen3b-r8/results/training_loss.png)
- Quy trình đánh giá giữ nguyên tập test và xuất dự đoán, chỉ số cùng phiếu chấm pháp lý ẩn tên mô hình. [Kết quả so sánh](outputs/evaluation/qwen3b-r8/results/comparison_report.md)

| Chỉ số trên 50 câu test | Base | Fine-tuned |
| --- | ---: | ---: |
| ROUGE-L | 0,228305 | 0,359918 |
| Token-level F1 | 0,308649 | 0,400965 |
| Exact Match | 0/50 | 0/50 |
| Thời gian sinh trung bình | 58,29 giây | 27,08 giây |

Hai lượt suy luận dùng cùng RTX 3050 Ti Laptop với tùy chọn cấp phát CUDA khác nhau, nên thời gian được báo riêng. Adapter cải thiện độ khớp văn bản, nhưng vẫn có câu dẫn sai điều và trả lời ngoài phạm vi. Độ chính xác pháp lý đang chờ chấm độc lập. [Phân tích 10 ca lỗi](reports/ERROR_ANALYSIS.md) · [Kiểm thử demo](reports/DEMO_TEST_REPORT.md)

## 4. Yêu cầu môi trường

- Python 3.12 hoặc 3.13 64-bit và PowerShell để chạy các lệnh Windows bên dưới.
- GPU NVIDIA với driver tương thích CUDA 12.6 để suy luận NF4. Khi tái lập huấn luyện trên Colab, dùng GPU T4 16 GB hoặc tương đương.
- PyTorch 2.8.0+cu126, Transformers 4.57.6, PEFT 0.17.1, TRL 0.22.2 và Gradio 6.9.0. [PyTorch CUDA](requirements-torch.txt) · [Phụ thuộc chính](requirements.txt) · [Phụ thuộc xử lý dữ liệu và biểu đồ](requirements-analysis.txt)
- Kết nối mạng để cài thư viện và tải mô hình nền ở lần chạy đầu.

## 5. Hướng dẫn cài đặt và chạy

Môi trường Python và các gói bên dưới dùng chung cho demo Gradio, xử lý dữ liệu, huấn luyện và đánh giá. Nếu chỉ muốn sử dụng chatbot đã huấn luyện, cài môi trường rồi chạy phần **Chạy demo trên Windows**. Các lệnh kiểm tra và tái lập dành cho người muốn kiểm chứng hoặc chạy lại thực nghiệm.

### Cài môi trường trên Windows

Mở PowerShell tại thư mục gốc của dự án sau khi clone hoặc giải nén bản nộp.

```powershell
python -m venv .venv
$py = '.\.venv\Scripts\python.exe'
& $py -m pip install -r requirements-torch.txt
& $py -m pip install -r requirements.txt
& $py -m pip install -r requirements-analysis.txt
& $py -m pip check
& $py -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
```

### Chạy demo trên Windows

Giải nén gói adapter nếu chưa có thư mục run, rồi mở Gradio. Demo dùng adapter đã huấn luyện trong gói bàn giao.

```powershell
$run = 'runs/qwen3b-r8-local'
$adapter = Join-Path $run 'final_adapter'
if (-not (Test-Path -LiteralPath $run)) {
    Expand-Archive -LiteralPath 'outputs/finetuning/qwen3b-r8-step100.zip' -DestinationPath $run
}
if (-not (Test-Path -LiteralPath (Join-Path $adapter 'complete.json'))) {
    throw 'Thư mục run đã tồn tại. Hãy giải nén ZIP adapter vào thư mục mới và đổi biến $adapter.'
}
$env:HF_HUB_CACHE = Join-Path $PWD 'artifacts/hf-cache'
& $py -X utf8 app/app.py --adapter $adapter
```

Mở [giao diện Gradio](http://127.0.0.1:7860) và thử câu “Những thao tác nào được coi là chuyển hướng xe?”. Sau khi cache có đủ trọng số, thêm `--offline` vào lệnh chạy để dùng bản tải cục bộ. Nhấn `Ctrl+C` để dừng.

### Kiểm tra kết quả và tái lập

Sau khi giải nén adapter, dùng các lệnh sau để kiểm tra dataset, adapter bước 100 và kết quả 50 câu đã lưu.

```powershell
& $py -X utf8 scripts/evaluate_baseline.py --check-only
& $py -X utf8 scripts/train_qlora.py --run-dir $run --status
& $py -X utf8 scripts/evaluate_adapter.py --adapter $adapter --run-dir outputs/evaluation/qwen3b-r8 --status
```

Để chạy đánh giá mới hoặc dựng lại dataset từ PDF nguồn, dùng thư mục đầu ra riêng.

```powershell
& $py -X utf8 scripts/evaluate_adapter.py --adapter $adapter --run-dir outputs/evaluation/reproduction
& $py -X utf8 scripts/prepare_legal_dataset.py --download --output data/rebuilt
```

### Huấn luyện trên Colab

Chọn GPU T4, mở [notebook huấn luyện QLoRA](notebooks/qlora_finetuning_colab.ipynb) và đưa [gói nguồn huấn luyện](artifacts/qlora_colab_source.zip) vào `MyDrive/vietnamese-legal-qlora/`. Để dùng adapter có sẵn, giải nén gói adapter vào thư mục `runs/qwen3b-r8-local/` bên dưới đường dẫn đó. Để huấn luyện mới, đặt `RUN_NAME` thành tên khác và chép `run_config.json` từ ZIP vào thư mục run mới. Notebook lưu checkpoint theo bước trên Drive. [Hướng dẫn khôi phục](RESUME_INSTRUCTIONS.md) · [Notebook so sánh](notebooks/model_comparison.ipynb)

## 6. Định hướng phát triển

- Bổ sung câu hỏi tình huống cần kết hợp nhiều điều khoản, có ngoại lệ hoặc thiếu dữ kiện.
- Chấm độc lập tính đúng pháp lý, tính đúng của trích dẫn và khả năng từ chối câu hỏi ngoài phạm vi.
- Thử nghiệm truy xuất văn bản chính thức theo mốc hiệu lực và kiểm tra điều khoản trước khi hiển thị câu trả lời.
- So sánh LoRA rank 8 và 16 với cùng dữ liệu, lịch huấn luyện và quy trình đánh giá.

## 7. Liên hệ

- Email liên hệ [24106898@st.phenikaa-uni.edu.vn](mailto:24106898@st.phenikaa-uni.edu.vn)

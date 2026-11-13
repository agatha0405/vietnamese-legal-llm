# Thiết lập và kiểm thử môi trường

## Mục tiêu

Chuẩn bị một quy trình cài đặt và kiểm tra có thể chạy lại trên Google Colab với T4 16 GB hoặc GPU tương đương. Notebook kiểm tra thư viện, kernel NF4, tải và sinh văn bản từ Qwen, một bước tối ưu SFT và khả năng nạp lại adapter. Mô hình chính và dự phòng dùng hai revision cố định để truy nguyên trọng số.

## Kiến trúc và lựa chọn kỹ thuật

- **Qwen2.5-3B-Instruct.** Mô hình ngôn ngữ tự hồi quy dự đoán token kế tiếp, gồm 3,09 tỷ tham số và 36 lớp Transformer decoder. Mô hình dùng RoPE, SwiGLU, RMSNorm, chia sẻ trọng số embedding với lớp đầu ra và dùng grouped-query attention với 16 query head, 2 key/value head. Checkpoint Instruct cung cấp sẵn chat template. [Model card Qwen 3B](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- **LoRA và QLoRA.** LoRA giữ cố định ma trận trọng số nền \(W\) và học hai ma trận hạng thấp \(B, A\) để thêm \(\Delta W = (\alpha/r)BA\). Với QLoRA, các lớp tuyến tính của mô hình nền được lượng tử hóa về NF4 4 bit. Double quantization nén các hằng số lượng tử hóa. Backpropagation truyền gradient qua mô hình nền đã đóng băng tới adapter. Phép thử dùng `all-linear` với rank 8, alpha 16 và gradient checkpointing ở chế độ `use_reentrant=False`. Rank 8 được dùng để kiểm tra môi trường. [LoRA](https://arxiv.org/abs/2106.09685) · [QLoRA](https://arxiv.org/abs/2305.14314) · [Hướng dẫn lượng tử hóa PEFT](https://github.com/huggingface/peft/blob/v0.17.1/docs/source/developer_guides/quantization.md)
- **SFT.** TRL nhận cặp hội thoại `prompt` và `completion`. Loss dự đoán token áp dụng lên phần completion thông qua `completion_only_loss=True`. Script đối chiếu mask với labels trước khi huấn luyện, rồi xác minh ít nhất một trọng số LoRA thực sự thay đổi và trọng số nạp lại bằng trọng số đã lưu. Mẫu thử là câu chào tiếng Việt trong bộ nhớ. [Tài liệu SFTTrainer 0.22.2](https://huggingface.co/docs/trl/v0.22.2/en/sft_trainer)
- **Python, CUDA và precision.** Chỉ mục PyTorch có wheel 2.8.0+cu126 cho Python 3.12 và 3.13 trên Linux x86-64 và Windows x86-64. Wheel bitsandbytes 0.48.2 hỗ trợ CUDA 12.6 và SM 7.5 của T4. Script dùng BF16 nếu GPU hỗ trợ phần cứng, hoặc FP16 trên T4. Notebook cài vào venv để tránh xung đột với thư viện cài sẵn trong Colab. [Chỉ mục wheel PyTorch CUDA 12.6](https://download.pytorch.org/whl/cu126/torch/) · [Hướng dẫn cài bitsandbytes 0.48.2](https://huggingface.co/docs/bitsandbytes/v0.48.2/en/installation)
- **Ràng buộc phiên bản.** Metadata TRL 0.22.2 yêu cầu Transformers ≥4.55.0, Accelerate ≥1.4.0 và Datasets ≥3.0.0. Các pin đã chọn thỏa các ràng buộc trực tiếp này. `huggingface-hub==0.36.0` nằm trong giao của các khoảng phiên bản do Transformers, Datasets, PEFT và Gradio khai báo. [TRL trên PyPI](https://pypi.org/project/trl/0.22.2/) · [Transformers trên PyPI](https://pypi.org/project/transformers/4.57.6/) · [Gradio trên PyPI](https://pypi.org/project/gradio/6.9.0/)

## Tệp và lệnh chạy

| Tệp | Vai trò |
| --- | --- |
| `README.md` | Phát biểu bài toán, cài môi trường và hướng dẫn chạy. |
| `.gitignore` | Loại môi trường ảo, dữ liệu làm việc, trọng số và đầu ra cục bộ khỏi Git. |
| `requirements-torch.txt`, `requirements.txt` | Cố định wheel PyTorch CUDA và các thư viện dự án. |
| `configs/environment.toml` | ID, revision và tham số của phép thử. |
| `scripts/check_environment.py` | Lệnh `environment` và `qlora`. |
| `notebooks/environment_setup_colab.ipynb` | Các cell cài đặt, kiểm thử và tải báo cáo. |
| `artifacts/environment_colab_source.zip` | Tệp mã nguồn tải lên từ notebook. |

Mở [notebook thiết lập trên Colab](../notebooks/environment_setup_colab.ipynb) và tải lên [gói nguồn kiểm tra môi trường](../artifacts/environment_colab_source.zip) khi được yêu cầu. Trên Windows, dùng [hướng dẫn cài môi trường](../README.md#cài-môi-trường-trên-windows). Kết quả kiểm tra thành công được ghi dưới dạng JSON, bao gồm GPU, danh sách thư viện đã cài và thông tin lượt chạy. Phép thử QLoRA lưu adapter vào thư mục tạm để xác minh thao tác nạp lại.

## Kiểm thử thực tế

| Kiểm tra | Kết quả |
| --- | --- |
| Workspace | Ban đầu trống. Đã khởi tạo kho Git cục bộ. |
| Máy phát triển | Windows, Python 3.13.5, NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4.096 MiB VRAM, driver 610.62 theo `nvidia-smi`. |
| Định dạng notebook | `nbformat.validate` đạt. Phân tích cú pháp Python đạt cho cả 6 cell code. |
| Script, cấu hình và gói nguồn | Phân tích cú pháp Python và TOML đạt. Ruff đạt. Cả 4 mục trong ZIP trùng nội dung tệp nguồn. |
| PyTorch CUDA trên Windows | Wheel 2.8.0+cu126 nhận GPU SM 8.6, báo CUDA build 12.6 và tạo được tensor trên `cuda:0`. |
| Bộ thư viện đã ghim trên Windows | PyTorch được cài bằng pip và các thư viện dự án bằng `uv pip install` vào cùng venv. `pip check` đạt. Các API chính import thành công và kernel NF4 chạy được forward/backward với BF16. [Bằng chứng JSON](../artifacts/environment_checks/environment-local.json) |
| Giải phụ thuộc cho Linux | `uv pip compile` đạt cho x86-64 Python 3.12 và 3.13. Xem [kết quả 3.12](../artifacts/environment_checks/resolved-linux-py312.txt) và [kết quả 3.13](../artifacts/environment_checks/resolved-linux-py313.txt) tại workspace. |
| Revision và chat template | Đã tải cấu hình và tokenizer tại đúng revision của cả hai mô hình, xác nhận kiến trúc `qwen2` và mã hóa được prompt hội thoại tiếng Việt. Phép kiểm tra này dùng Transformers 5.15.0 có sẵn trên máy. |
| QLoRA 1.5B trên RTX 3050 Ti | Đạt ở FP16 và BF16. Mỗi lượt nạp 196 lớp NF4, sinh 8 token, thực hiện một bước SFT với 9.232.384 tham số LoRA được phép cập nhật và xác minh toàn bộ trọng số adapter nạp lại khớp bản lưu. [Bằng chứng FP16](../artifacts/environment_checks/qlora-fallback-fp16-local.json) · [Bằng chứng BF16](../artifacts/environment_checks/qlora-fallback-bf16-local.json) |
| QLoRA 3B trên Colab Tesla T4 | Đạt ở FP16 với Python 3.13.16 và CUDA 12.6. Qwen2.5-3B-Instruct đúng revision, nạp 252 lớp NF4, sinh 8 token và thực hiện một bước SFT với 14.966.784 tham số LoRA được phép cập nhật (0,483% tổng số tham số). Adapter nạp lại khớp bản lưu. [ZIP kết quả Colab](../artifacts/environment_checks/colab-smoke-test-20261112T165816438401Z.zip) |

Lượt SFT FP16 ban đầu phát hiện lỗi tích hợp với TRL 0.22.2. Khi nhận mô hình 4-bit đã gắn adapter, TRL chuẩn bị lại mô hình cho k-bit training và làm đóng băng cả adapter. Script đã chuyển sang đưa mô hình nền cùng `peft_config` vào `SFTTrainer` và kiểm tra danh sách tham số trainable sau khi khởi tạo. Sau thay đổi này, phép thử với loss scale mặc định vẫn chưa cập nhật trọng số FP16. Cấu hình phép thử đặt loss scale khởi tạo ở 256 và xác nhận trực tiếp optimizer đã chạy và trọng số LoRA đã thay đổi.

Lượt FP16 cục bộ với 1.5B ghi nhận loss 3,7384 và mức cấp phát CUDA đỉnh 2.175 MiB. Lượt Colab với 3B ghi nhận loss 4,8469 và mức cấp phát CUDA đỉnh 3.358,4 MiB. PyTorch báo tổng VRAM của GPU Colab là 14.912,7 MiB. Loss được đo trên một câu chào tiếng Việt và dùng để kiểm tra phép tối ưu.

## Kết luận kiểm thử

**Hoàn thành.** Lượt Colab kết thúc lúc 00:00 ngày 13/11/2026 (giờ Việt Nam). Hai báo cáo JSON trong ZIP có cùng thông tin môi trường. Danh sách phiên bản thư viện, revision 3B và tham số phép thử khớp cấu hình dự án. Kết quả trên câu hỏi pháp luật được trình bày trong [báo cáo so sánh mô hình](../outputs/evaluation/qwen3b-r8/results/comparison_report.md).

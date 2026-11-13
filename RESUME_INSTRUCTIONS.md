# Khôi phục dự án sau khi ngắt phiên

Đọc [PROJECT_PROGRESS.md](PROJECT_PROGRESS.md) trước. Sau khi clone, cài môi trường và giải nén adapter theo [README](README.md#chạy-demo-trên-windows). Để tiếp tục một run đã bắt đầu, dùng đúng thư mục run và các tùy chọn cũ.

## Xác định checkpoint trên máy hiện tại

```powershell
$env:HF_HUB_CACHE = Join-Path $PWD 'artifacts/hf-cache'
$py = '.\.venv\Scripts\python.exe'
$run = 'runs/qwen3b-r8-local'
& $py -X utf8 scripts/train_qlora.py --run-dir $run --status
```

`--status` báo bước hợp lệ cuối cùng hoặc adapter đã hoàn tất. Mỗi checkpoint có SHA-256 cho adapter, Trainer state, optimizer, scheduler, scaler và RNG. Lệnh huấn luyện tự truyền checkpoint hợp lệ mới nhất vào `resume_from_checkpoint`. Ba checkpoint hợp lệ gần nhất nằm trong thư mục run. Các bản cũ hơn và bản ghi lỗi được chuyển nguyên trạng vào `history/` để giữ dữ liệu. Thư mục bắt đầu bằng `.checkpoint-` hoặc `.final-` là bản lưu dở. `attempt-*.json` lưu môi trường và checkpoint bắt đầu của từng lần chạy.

Nếu thấy `.writer`, xem tiến trình sở hữu trước. Nếu nó còn chạy, theo dõi tiến trình đó. Nếu runtime cũ chắc chắn đã kết thúc, đổi tên khóa để giữ bằng chứng rồi chạy lại.

```powershell
Get-Content (Join-Path $run '.writer/owner.json')
# Chỉ tra PID trên đúng máy ghi trong owner.json
$owner = Get-Content (Join-Path $run '.writer/owner.json') | ConvertFrom-Json
Get-Process -Id $owner.pid
# Chỉ đổi tên sau khi xác nhận tiến trình cũ đã dừng
Rename-Item -LiteralPath (Join-Path $run '.writer') -NewName ('.writer-stale-' + (Get-Date -Format yyyyMMddHHmmss))
```

`save_steps=5` tính theo bước cập nhật optimizer. Một lần ngắt giữa hai checkpoint có thể làm chạy lại tối đa phần cập nhật chưa lưu kể từ checkpoint cuối. Dùng `--pause-at-step N` khi muốn dừng có kiểm soát và lưu sớm.

RTX 3050 Ti 4 GB từng hết VRAM sau bước 12. Cấu hình allocator bên dưới đã tiếp tục từ bước 10 tới checkpoint-15. Trên T4 có thể dùng allocator mặc định.

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF = 'garbage_collection_threshold:0.75,max_split_size_mb:128'
& $py -X utf8 scripts/resume_pipeline.py --offline
```

## Huấn luyện QLoRA trên Colab với Google Drive

Mở `notebooks/qlora_finetuning_colab.ipynb` trong Colab. Chọn GPU T4, kết nối runtime, cấp quyền mount Google Drive và tải ZIP nguồn lên Drive theo hướng dẫn của notebook. Các thao tác chọn GPU, cấp quyền Drive và nhấn chạy cell cần làm thủ công nếu phiên Codex không có quyền vào Colab.

Notebook hiện tại dùng [gói nguồn huấn luyện](artifacts/qlora_colab_source.zip). Sau khi tiến trình Windows đã dừng, tạo ZIP checkpoint mới nhất bằng `& $py -X utf8 scripts/package_colab_run.py --run-dir $run`. Lệnh giữ nguyên các ZIP có sẵn.

Notebook cài môi trường trong `/content`, còn các checkpoint được ghi vào thư mục tương ứng với `RUN_NAME` trong `/content/drive/MyDrive/vietnamese-legal-qlora/runs/`. Khi Colab ngắt, chọn GPU và mount lại Drive, rồi chạy cùng notebook. Nếu khóa cũ còn trên Drive, xác minh runtime cũ đã kết thúc trước khi chạy lệnh đổi tên trong notebook. Để tiếp tục một run đã bắt đầu trên Windows ở Colab, chép thư mục run của Windows lên Drive, đặt `RUN_NAME` đúng tên đã chép, giữ cùng model, rank và cấu hình. Có thể có sai khác số học khi đổi GPU.

Bản chuyển checkpoint-10 được lưu riêng trên máy thực nghiệm tại `artifacts/qwen3b-r8-resume-step10.zip`. Kiểm tra trạng thái trên máy trước khi chuyển để chọn checkpoint mới nhất.

## Đánh giá và chạy demo

Sau khi huấn luyện trên Colab, chép thư mục `final_adapter` vào thư mục run cục bộ. Sau đó lệnh dưới đây chạy tiếp các bước trên máy hiện tại. Script bỏ qua adapter, biểu đồ, đánh giá và phép thử demo đã qua xác minh.

```powershell
& $py -X utf8 scripts/resume_pipeline.py --offline
```

Sau khi `--status` báo `Complete`, dùng adapter cuối đã kiểm tra SHA-256.

Run local hiện đã hoàn tất. Các lệnh đánh giá và tóm tắt sẽ xác minh đầu ra đang có và trả về. Dùng lệnh cuối để mở lại giao diện.

```powershell
$adapter = Join-Path $run 'final_adapter'
& $py -X utf8 scripts/summarize_training.py --adapter $adapter
& $py -X utf8 scripts/evaluate_adapter.py --adapter $adapter --run-dir outputs/evaluation/qwen3b-r8 --status
& $py -X utf8 scripts/evaluate_adapter.py --adapter $adapter --run-dir outputs/evaluation/qwen3b-r8 --offline
& $py -X utf8 scripts/demo.py --adapter $adapter --offline --check
& $py -X utf8 app/app.py --adapter $adapter --offline --port 7860
```

Script đánh giá lưu từng dự đoán vào tệp riêng và tiếp tục theo ID kế tiếp. Nếu đánh giá bị ngắt và còn khóa cũ, áp dụng quy trình kiểm tra và đổi tên khóa trong thư mục đánh giá. Kết quả cuối nằm ở `outputs/evaluation/qwen3b-r8/results/`. Phiếu chấm pháp lý được xáo thứ tự và có bảng giải mã riêng. Sao chép phiếu ra tệp của người chấm trước khi nhập nhãn để giữ nguyên SHA-256 của kết quả. Mở demo tại [localhost](http://127.0.0.1:7860).

Kiểm thử bổ sung trên 12 câu ngoài phạm vi lưu vào thư mục riêng, cần người chấm đọc dự đoán để ghi nhãn từ chối.

```powershell
& $py -X utf8 scripts/evaluate_refusal.py --adapter $adapter --run-dir outputs/evaluation/refusal-qwen3b-r8 --offline
```

## Cập nhật tài liệu

Cập nhật `PROJECT_PROGRESS.md`, báo cáo kỹ thuật và README theo checkpoint, log, kết quả đánh giá và kiểm thử demo thực tế. Nếu có tài nguyên, chạy lại huấn luyện với `--rank 16` trong một thư mục run riêng rồi dùng thư mục đánh giá riêng khi đánh giá. Phương án dự phòng `--fallback` cũng cần thư mục riêng và cần một baseline 1.5B tương ứng để so sánh.

# Hướng dẫn cho AI Agent

Cập nhật 2026-11-06. Áp dụng cho Codex, subagent và Antigravity.
Đọc file này; chỉ mở thêm tài liệu đúng nhiệm vụ. Không bắt đầu từ kế hoạch cũ.

Phạm vi mới được chủ project giao: triển khai B trên nhánh dev-caitien từ commit
baseline 0f65e4cc4c67b042998b84c5ac068abeeca09719. Không sửa checkout dev_tt.
Baseline đã hoàn tất: MLflow 3aaa5b8d882f4555a17d2c139e38513c, 5411 train/589 val,
1 epoch/22 global steps; final val loss 0.5565038323402405. Nhánh này là pilot
weighted loss, không phải backbone mới. Không resume checkpoint 700 hoặc 22.
Đọc hướng dẫn mới này thay phần trạng thái baseline cũ trong IMPLEMENTATION_PLAN.md.
Coordinator phụ trách notebook, GitHub và Kaggle. Worker chỉ sửa các file được giao;
không tự push, cài package, mở GPU hay sửa notebook. Không đổi dependency baseline.

## Mục tiêu hiện tại

- So sánh A: QLoRA hiện tại và B: QLoRA tăng ưu tiên mẫu khó.
- Cả hai khởi tạo từ cùng Qwen2.5-VL-3B-Instruct + LoRA mới, cùng seed.
- Train mới 1 epoch trên cùng tập nhỏ; KHÔNG warm-start/resume checkpoint 700.
- Chia 90% train / 10% validation theo ẢNH NGUỒN; mọi biến thể nhiệm vụ
  của một ảnh nằm cùng phía. Giữ cùng manifest cho A/B, không dùng test làm val.
- Đánh giá/lưu theo step như workflow cũ, dự kiến mỗi 5 optimizer step.
  Chốt khoảng cách theo số bước thực tế; phải đo checkpoint cuối riêng.
- B dự kiến tăng trọng số loss cho mẫu khó TRONG TRAIN, có giới hạn.
  Giữ nguyên mẫu/số bước; chưa oversample hay thêm task-affinity/kiến trúc mới.
  Đây là giả thuyết cần thử, chưa có code cải tiến hoặc bằng chứng hiệu quả.
- Validation chỉ đo, dùng cùng loss chuẩn cho A/B; không lấy ảnh val cho model học.
- Chi tiết cấu hình và thứ tự làm: docs/IMPLEMENTATION_PLAN.md.

## Source và nơi chạy

- Repo này /home/nhat/test-unimer chứa workflow train để GitHub lưu và Kaggle clone.
- Notebook chính: uni-mumer-kaggle-dagshub-banmoi.ipynb.
- Notebook chỉ setup, chỉnh tham số, gọi script repo; giữ cell ngắn, thứ tự quen thuộc.
- Cấu hình thực chạy = train/Uni-MuMER-train.yaml + notebook overrides.
- Registry nguồn: train/dataset_info.json. scripts/prepare_baseline_data.py xuất
  registry train/val riêng trong dataset_dir; không sửa registry ví dụ của framework.
- Kaggle dùng Python 3.10 riêng và requirements project, không mượn ML packages
  của kernel. Dùng PYTHON tuyệt đối; kernel chỉ điều phối/đọc Kaggle Secrets.
- Mục tiêu 2 x T4, FP16; NF4 4-bit, LoRA rank64/alpha16/dropout0.05.
  Hai GPU không gộp VRAM. Không tự đổi batch/LR/ảnh/chuỗi để né lỗi.
- Inference/benchmark ở /home/nhat/Uni-MuMER-project trên RTX 3070.
  Không tự sửa repo hoặc môi trường local đó trong nhiệm vụ train.

## Khi nhận việc

1. Đọc git status và các file liên quan; giữ nguyên thay đổi có sẵn.
2. Làm đúng phạm vi được giao. Sửa nhỏ trên workflow cũ; không tự xây lại pipeline.
3. Backup notebook trước khi sửa. Kiểm thử nhẹ, báo file đổi và việc chưa kiểm chứng.
4. Không nhét nhiều cờ review/setup, commit/hash bắt buộc hoặc kiểm tra lặp vào cell.
   Không thêm helper chỉ để di chuyển sự phức tạp nếu workflow hiện có đã đáp ứng.
5. Nhiều agent: mỗi file một người sửa, giao đầu ra rõ ràng. Auto-accept theo phạm vi,
   không phải quyền sửa mọi file/chạy mọi lệnh. Không gọi lại worker đã xong chỉ để gửi luật.

## Trước GitHub hoặc Kaggle

- Quy tắc thực hiện: docs/KAGGLE_RUNBOOK.md. GitHub phải có đủ helper/config đi kèm.
- Không tự commit/push, cài thư viện, tải model/dataset hoặc chạy GPU khi chưa được giao.
  Khi đã được giao một run rõ cấu hình/ngân sách, không hỏi lại từng thao tác nhỏ.
- Trước MỌI sửa Kaggle: kiểm tra job đang chạy, export backup đọc được, ghi URL/version.
  Không tự dừng job hoặc đổi source/dependency giữa run.
- Không lộ token, .env hay URL có mật khẩu. Không xóa/ghi đè checkpoint hoặc kết quả cũ.
- Lưu actual MLflow run ID + nhãn baseline/improved, commit, split và cấu hình cuối.
  RUN_UUID là tag, không phải MLflow run ID. Upload lỗi không được báo thành công.
- Tách rõ sửa local / đã push / đã cập nhật Kaggle / đã chạy thật. Test CPU không chứng
  minh Kaggle đã chạy. Không đánh dấu kế hoạch là đã triển khai.

## Việc chưa làm và tài liệu cũ

Notebook local đã cấu hình baseline 1 epoch, 1.000 dòng/nguồn và val 10% theo nhóm ảnh.
Cell 5 (Thông số baseline) là nơi chỉnh mẫu/epoch/val; helper chuẩn bị subset rồi
xuất YAML runtime, không cap/chia lần hai. Đã thử draft Kaggle 2 dòng/nguồn và
1 optimizer step thành công. Đã kiểm chứng thêm train/val/checkpoint/upload DagsHub
và tải adapter về local khớp SHA256 ở run 79065aa1f6d34262a90122f7f7b25060.
ĐÃ chạy baseline 1.000 dòng/nguồn; CHƯA inference/benchmark local.
Helper đọc ParquetFile theo batch và đóng reader rõ ràng để tránh lỗi thoát Python.
Train theo epoch phải dùng max_steps=-1, KHÔNG ghi null (Trainer so sánh với số 0).
Nhánh dev-caitien đã tạo trong checkout riêng. Bằng chứng baseline chỉ đọc:
/home/nhat/Uni-MuMER-project/data/dagshub_downloads/3aaa5b8d882f4555a17d2c139e38513c/validation_review/.
Các thư viện sync/timeout mới chưa nối đầy đủ vào đường chạy notebook.
Việc tiếp theo là code và kiểm thử loss cải tiến, rồi coordinator triển khai Kaggle.
RUN_TRAINING mặc định False; không tự bật để kiểm thử local. Validation vẫn loss chuẩn.
Checkpoint 700 chỉ là kết quả lịch sử, KHÔNG phải điều kiện chặn thí nghiệm mới.
docs/archive/ và docs/reports/ là snapshot tham khảo, không phải lệnh thực thi.
Nếu tài liệu cũ mâu thuẫn, theo file này và IMPLEMENTATION_PLAN.md hiện tại.

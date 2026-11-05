# Baseline dev_tt: thay đổi local ngày 2026-11-06

## Đã sửa

- Notebook `uni-mumer-kaggle-dagshub-banmoi.ipynb`: giữ 12 cell; chỉ đổi cell
  tiêu đề, cấu hình chung, clone nhánh và thông số baseline (index 0, 1, 3, 6).
- Clone rõ `dev_tt`. Cell `# 5. Thông số baseline` là nơi chỉnh sample/val/epoch.
- Train mới từ Qwen2.5-VL-3B-Instruct + LoRA mới, 1 epoch; không resume/adapter 700.
- Tối đa 1.000 dòng mỗi nguồn trong 6 nguồn cũ, seed 42; FP16 cho 2 x T4.
- Giữ batch/GPU 2, tích lũy 64, LR 1e-4, NF4 4-bit, rank64/alpha16/dropout0.05.
- Eval/save mỗi 5 optimizer step. `load_best_model_at_end=False` để phép eval sau
  train đo trọng số cuối; best validation vẫn được ghi riêng, không gọi final là best.

`scripts/prepare_baseline_data.py` lấy subset trước rồi chia 90/10 theo nhóm ảnh.
Nhóm dùng cùng RGB pixel hash hoặc cùng ID trong họ nguồn CROHME; biến thể nhiệm vụ
cùng ảnh ở cùng phía. Không phải bộ phát hiện trùng ảnh đã crop/resize bất kỳ.
Helper xuất `train.json`, `val.json`, ảnh PNG, `dataset_info.json`, `split_manifest.json`
vào `/kaggle/working/baseline-data/n1000-v0.1-s42`, không ghi vào source train gốc.
Manifest có source revision, membership, count từng nguồn, nhóm ảnh và hash file.
Cache khác cấu hình/hỏng sẽ báo lỗi, không ghi đè. Bản build lỗi được giữ để kiểm tra.

YAML thực chạy ở `/kaggle/working/runs/<RUN_UUID>/runtime.yaml` trỏ registry mới,
`dataset=baseline_train`, `eval_dataset=baseline_val`, `val_size=0`, bỏ cap lần hai.
`val_size=0` ở runtime KHÔNG bỏ validation: tập val đã tách riêng trong manifest.
Tỷ lệ 10% là nhóm ảnh, không nhất thiết đúng 10% số dòng đa nhiệm.
YAML nền `train/Uni-MuMER-train.yaml` và registry nguồn `train/dataset_info.json` giữ nguyên.

`scripts/dagshub_logger.py` và `scripts/train_with_checkpoint_sync.py` ghi full UUID,
actual MLflow run ID, nhãn baseline/split và best/final; chặn UUID tìm thấy nhiều run.
Wrapper chờ watcher kết thúc trước final sync và trả lỗi khi final sync thất bại.
Final upload kiểm kê file/kích thước remote đệ quy: **sizes_only, không xác minh SHA256
remote**, không chứng minh chống hỏng cùng kích thước. State checkpoint nền ghi rõ
chưa remote-verified. Không nối toàn bộ thư viện sync/timeout mới vào notebook.

## Kiểm thử đã chạy

- Python local 3.10: `python -B -m unittest discover -s tests -q`: **251 test đạt**.
- Thêm test subset/grouping/cache, notebook/runtime và logging/sync bằng mock.
- Notebook JSON đọc được, 12 cell, không có output; Python AST và `bash -n` đạt.
- `git diff --check` đạt. Kiểm thử dùng dữ liệu ảnh nhỏ tổng hợp, không train model.
- Backup trước sửa: `/home/nhat/Uni-MuMER-project/data/kaggle_backups/2026-11-06-baseline-001`.

## Chưa làm / còn kiểm chứng

- Chưa commit/push GitHub, sửa Kaggle, cài thư viện, tải model/dataset lớn hoặc chạy GPU.
- Chưa tạo subset thật từ 6 nguồn, chưa có actual count/step/budget hay kết quả baseline mới.
- Lần đầu chuẩn bị subset cần mạng Hugging Face. Streaming hạn chế số dòng đọc, không
  cam kết số byte tải đúng bằng lượng ảnh đã chọn. Cần kiểm tra ID/schema nguồn thực tế.
- Chưa xác nhận cài requirements và MLflow trên Kaggle. Watcher join còn không có
  deadline tổng; không tuyên bố pipeline tự động qua đêm đã được kiểm chứng.
- `RUN_TRAINING=False` giữ mặc định an toàn. Bật True trong bản run được giao sau khi
  đủ source trên GitHub và đã backup notebook/job Kaggle hiện tại.
- Chưa tạo `dev-caitien`, chưa triển khai weighted loss. Khi làm B, dùng lại đúng
  manifest/source revision của A, không tự lấy bản dataset mới và chia lại.

Bước tiếp: duyệt diff -> commit/push đúng file được giao -> backup Kaggle ->
chạy baseline từ source đó -> kiểm tra actual split, validation, checkpoint và run DagsHub.

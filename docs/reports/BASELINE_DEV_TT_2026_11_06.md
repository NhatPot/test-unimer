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

## Sửa dependency sau Version 12

Version 12 (scriptVersionId 366053614) dừng trước train sau 749,7 giây ở cell cài
thư viện: pip check báo Prometheus 8.1.0 cần Starlette >=1 nhưng bản cài là 0.52.1,
và decord 0.6.0 không được hỗ trợ theo kiểm tra platform của pip. Không suy ra
đây là lỗi CUDA hay thư viện có sẵn trong kernel Kaggle.

- `requirements.txt`: giữ vllm 0.8.5, thêm Prometheus 7.1.0 (chấp nhận Starlette
  >=0.30,<1); đổi qwen-vl-utils[decord] thành qwen-vl-utils cùng version 0.0.11.
  Baseline ảnh không cần extra đọc video decord; không tuyên bố sửa hỗ trợ video.
- Cell cài dependency: dùng một lệnh pip với cả `-r requirements.txt` và
  `-e train/LLaMA-Factory` để resolver xét hai bộ yêu cầu cùng lúc. Giữ pip check,
  Python riêng và thông số train; không sửa framework vendored hoặc YAML.
- Thêm hai test chống tách lệnh cài/bỏ kiểm tra và chống dependency drift.
  253 CPU tests, notebook Python/Bash syntax và git diff check đạt.
- Đã đối chiếu metadata chính thức:
  [Prometheus 7.1.0](https://pypi.org/pypi/prometheus-fastapi-instrumentator/7.1.0/json),
  [Gradio 5.45.0](https://pypi.org/pypi/gradio/5.45.0/json),
  [Qwen VL Utils 0.0.11](https://pypi.org/pypi/qwen-vl-utils/0.0.11/json).
- Bản sửa dependency mới chỉ ở local dev_tt, chưa push/upload hoặc chạy lại Kaggle.
  Test CPU/metadata không thay thế kiểm chứng cài toàn bộ dependency trên Kaggle.

## Kết quả thử draft và sửa reader

Phần này cập nhật trạng thái mới hơn các ghi nhận trước ở trên:

- `8734ab0` đã push dev_tt; cài sạch trên Kaggle đạt pip check, nhận đủ 2 Tesla T4.
- Tạo subset bằng datasets streaming cũ ghi được file nhưng crash lúc Python
  finalizing (`PyGILState_Release`, exit -6). Không bỏ qua lỗi hoặc dùng cache để che.
- `19d2836` đổi riêng helper sang ParquetFile.iter_batches(use_threads=False),
  giới hạn đọc và đóng generator/file bằng context manager. Giữ datasets 3.2.0,
  không monkey-patch thư viện cài đặt hay đổi model/hyperparameter.
- 256 test CPU đạt, có subprocess thoát sạch sau dừng đọc sớm.
- Hai lần tạo subset THẬT ở hai thư mục mới trên Kaggle đều thành công:
  10 train/2 val, 8 ảnh, 7 nhóm train/1 val, giao nhóm ảnh bằng 0.
- Draft train 1 optimizer step thành công trên 2 T4, FP16/NF4/LoRA, 10 dòng train.
  Trainer báo train_runtime=6.303 giây, train_loss=0.0636883; đây là smoke kỹ thuật,
  KHÔNG phải metric chất lượng hoặc bằng chứng cải thiện (1 bước có warmup).
- Test dùng YAML riêng để chuỗi "no" không bị CLI/OmegaConf hiểu thành boolean.
  Không sửa parser framework hoặc mang lệnh smoke vào baseline.
- Không Save Version/Save & Run All mới, không upload DagsHub. Trainer vẫn tự lưu
  adapter cuối ở working/smoke-one-step; giữ riêng, không dùng làm khởi tạo baseline.
- Đã export notebook kết quả rồi khôi phục draft về source baseline: 1.000 dòng/nguồn,
  wrapper chuẩn, RUN_TRAINING=False; xóa cell chẩn đoán/lệnh smoke và clear outputs.
- Bằng chứng local: data/kaggle_backups/2026-11-06-baseline-001/ ở repo
  Uni-MuMER-project, file kaggle-draft-one-step-passed.ipynb và DRAFT_SMOKE.md.
- Còn chưa kiểm chứng: baseline lớn hơn, validation trong run train đầy đủ,
  checkpoint sync/resume và upload DagsHub. Không tuyên bố end-to-end hoàn tất.

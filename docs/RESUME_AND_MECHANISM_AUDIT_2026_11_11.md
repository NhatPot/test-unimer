# Rà soát notebook và khôi phục resume

Ngày 2026-11-11. Chỉ sửa local nhánh dev-caitien, không push hoặc sửa/chạy Kaggle.
Không thay checkpoint, trọng số, dữ liệu, dependency hay kết quả cũ.

## Cập nhật được duyệt ngày 2026-11-12

- Chủ project đã giao triển khai B full resume 2 -> tổng 4 epoch, bật lại chọn best
  theo eval_loss, giữ val/save 50 bước, NF4 + FP16 và split cũ. Không bật benchmark.
- Notebook template vẫn không tự train; bản triển khai bật RUN_TRAINING và resume.
- load_best_model_at_end nay True. Bảng dưới là snapshot trước quyết định này.
- Checkpoint-338 còn trỏ best checkpoint-300 ở đường dẫn phiên Kaggle cũ.
  Helper tải/kiểm hash cả hai, đổi đường dẫn best trong bản sao Trainer state,
  giữ nguyên optimizer/scheduler/RNG/counter và byte nguồn. Wrapper đưa best cũ
  vào output mới trước train để vẫn được bảo vệ, sync/upload và dùng ở lần sau.
- File nhị phân bất biến dùng hardlink khi cùng filesystem để tránh tốn gấp đôi
  ổ Kaggle 20 GB; trainer_state.json luôn sao chép riêng trước khi đổi đường dẫn.
- CPU tests sau bổ sung: 181 PASS; vẫn phải kiểm chứng full resume trên Kaggle.

## Bản đối chiếu

- V5 trên main: 6255423321c3c2e59454ad49bd4808d491343c2c, ngày 01/07/2026;
  có tải checkpoint DagsHub và truyền resume_from_checkpoint.
- Backup v8 trên main: b16d323a24fd30c2eb69d3626db797e88b5c109d, ngày 26/06/2026;
  cũ hơn v5, không có cell resume. Không suy thứ tự từ tên v5/v8.
- B trước sửa: dev-caitien f99bb689282128ab23034e68668d65375f209fa6.
- Backup nguyên bản trước lần sửa này:
  /home/nhat/Uni-MuMER-project/data/kaggle_backups/2026-11-11-restore-resume-001/.

## Các cơ chế đã đối chiếu

| Cơ chế | Bản cũ và B trước sửa | Xử lý lần này |
| --- | --- | --- |
| Resume DagsHub | V5 có; B mất cell, helper ép train mới | Khôi phục công tắc, full checkpoint, không tạo adapter mới khi resume |
| Train mới | V5 tắt resume để train mới; B chỉ train mới | Giữ nguyên; resume OFF, RUN_TRAINING=False mặc định |
| Full checkpoint | V5/B đều save_only_model=False | Giữ; kiểm tra thêm scaler FP16 và RNG của hai GPU khi resume |
| Sync trong train | V5/B đều gọi train_with_checkpoint_sync.py | Giữ wrapper, chu kỳ và final sync |
| Upload model/config/log | V5/B đều gọi dagshub_logger.py upload | Giữ; thêm tag run cha và checkpoint nguồn |
| Python riêng/source GitHub | Cả hai có | Giữ env riêng, requirements, clone đúng dev-caitien; không đổi package |
| Chỉnh thông số | Cả hai có runtime YAML | Giữ cell thông số và YAML gốc, không đổi batch/LR/loss |
| Công tắc bỏ runtime YAML | V5 có USE_RUNTIME_YAML_OVERRIDE; B không còn | Ghi nhận; chưa tự thêm đường bỏ qua split 90/10 hiện tại |
| Nạp best khi train xong | V5 True; B False | Giữ B đo model cuối; tùy chọn vẫn trong cell, không mất chức năng framework |
| Validation | V5 dùng cấu hình cũ; B chia 90/10 theo nhóm ảnh | Giữ phương án đã chốt, validation loss chuẩn |
| Benchmark trên Kaggle | V8 bật; v5 đã comment; B vẫn giữ cell comment | Không xóa cell/script, không tự bật; hiện test trên RTX 3070 |
| Upload kaggle_test_results riêng | V5 có; B không còn đoạn riêng | Chưa tự thêm: đoạn cũ dùng nhầm RUN_UUID làm MLflow run ID, có thể lấy nhầm kết quả cũ |
| Tải ảnh test | V8 bật; v5/B comment | Giữ, không tự tải hoặc xóa |
| Debug model/registry trong v8 | Không có trong B | Ghi nhận; là cell điều tra thủ công, chưa tự chèn lại/chạy load model |

Khác biệt không đồng nghĩa đã chứng minh tất cả đều bị xóa nhầm. Bật lại benchmark/
upload test, bỏ runtime YAML hoặc đổi best/final phải xin chủ project xác nhận trước.
Không phục hồi nguyên đoạn upload có lỗi RUN_UUID/run ID chỉ để giống bản cũ.

## Dùng resume trong B

1. Cell `4.5 Resume từ DagsHub` có USE_DAGSHUB_RESUME, RESUME_RUN_ID,
   RESUME_ARTIFACT_PATH, RESUME_TOTAL_EPOCHS như v5. Mặc định resume OFF,
   RUN_TRAINING=False; không tự tải checkpoint hoặc train.
2. Khi được giao chạy, chọn actual MLflow run ID/artifact. Gợi ý hiện có là B
   619dd8beecbf4eb5a533f4c6618409b7, training/checkpoint-338, epoch 2.
   Dùng checkpoint huấn luyện, không dùng model merge cho inference.
3. Tải vào RUN_DIR/resume-source/<run_id>/, tách OUTPUT_DIR mới. Kiểm tra đủ
   adapter/config, trainer_state, optimizer, scheduler, scaler, rng_state_0/1;
   so kích thước/SHA256 với training/artifact_manifest.json của run nguồn.
   Đây là kiểm tra tính toàn vẹn theo manifest được chọn, không chứng minh model tốt.
   Helper vẫn cho run lịch sử không truyền manifest nếu chưa có; không âm thầm bỏ
   kiểm tra manifest của B hiện tại khi có lỗi.
4. Helper data giữ split; đặt resume_from_checkpoint và adapter_name_or_path cùng
   checkpoint, create_new_adapter=False, max_steps=-1. Train mới mới xóa hai đường dẫn.
   Đích tổng epoch phải lớn hơn epoch đã học. Resume lỗi thì dừng, không fallback train mới.
5. OUTPUT_DIR mới phải chưa tồn tại. Không tải nguồn vào output để vượt chặn ghi đè.
   Ghi tag parent_run_id, resume_artifact_path, initial_global_step, initial_epoch.
   Giữ nguyên run nguồn và kết quả cũ.

## Điều kiện trước khi chạy thật

- Chốt scheduler khi kéo B hoàn tất 2 lên 4 epoch. Không tự reset scheduler/optimizer:
  cosine cho đích 2 khác run đặt đích 4 từ đầu. Lần sửa này không thay thuật toán/lịch LR.
  Phải kiểm tra đúng Transformers của Kaggle và thông báo LR sau resume trước phiên dài.
- Giữ code loss B, model revision, split, batch/accumulation, seed và hai GPU như nguồn.
  Không lén đổi subset/phương pháp trong resume vì sẽ sai ý nghĩa bộ đếm/đối chứng.
- Tính đủ 12 giờ gồm setup/eval/lưu/upload, chừa ít nhất 1 giờ dự phòng.
- Log thật phải chứng minh đọc state và tiếp tục sau step 338, không quay lại 0.
  Cấu hình không đổi thì đích 4 dự kiến khoảng 676 bước; phải kiểm tra Trainer thực tế.
- Backup bản Kaggle trước mọi sửa sau khi được duyệt triển khai.
- Chưa push/cập nhật Kaggle, chưa tải thêm weights hoặc chạy train thật trong lần này.

## Kiểm thử và giới hạn

Toàn bộ unittest trong checkout B, CUDA_VISIBLE_DEVICES rỗng, -B: 178 tests PASS.
Test downloader dùng file giả/mock DagsHub: thiếu state, hash sai, cache khác run,
không fallback. Test runtime kiểm tra fresh/resume, tổng epoch, split, loss và tag cha.
Suite có test mô hình đồ chơi CPU rất nhỏ; không tải/train Qwen, không dùng GPU.
Đây chưa phải bằng chứng full resume đã chạy được trên Kaggle.

AGENTS.md ở dev_tt/dev-caitien đã yêu cầu hỏi trước khi thay/xóa/vô hiệu hóa lâu dài
hoặc đổi mặc định công nghệ/cơ chế. Train mới chỉ tắt resume, không được xóa tính năng.
Không sửa code/notebook dev_tt trong lần này vì có thay đổi riêng chưa commit;
chỉ bổ sung quy tắc AGENTS.md, không giả định baseline đã nhận tính năng mới.

# Hướng dẫn cho AI Agent

Cập nhật 2026-11-15. Áp dụng cho Codex, subagent và Antigravity.
Đọc file này; chỉ mở thêm tài liệu đúng nhiệm vụ. Không bắt đầu từ kế hoạch cũ.

## Phải hỏi trước khi đổi cơ chế

- Trước khi xóa, thay thế, vô hiệu hóa lâu dài hoặc đổi mặc định của công nghệ/cơ chế,
  giải thích thay đổi, lý do và ảnh hưởng rồi chờ chủ project xác nhận rõ ràng.
  Lệnh chung như "sửa cho chạy được", "dọn code" hoặc "làm tiếp" không phải quyền bỏ cơ chế.
- Áp dụng cho resume, checkpoint, sync/upload, chọn best/final model, train/validation,
  inference/chấm điểm, backbone, loss, dependency, precision/quantization và scheduler.
- Train mới chỉ TẮT resume bằng tùy chọn; phải giữ khả năng resume cho lần sau.
  Không âm thầm đổi full resume thành chỉ nạp adapter hoặc tạo adapter mới.
- Backup notebook trước sửa, đối chiếu các chức năng trước/sau; báo rõ phần giữ nguyên,
  phần đã được duyệt và phần còn cần hỏi. Không tự sửa run hay kết quả đã hoàn tất.
- Quy tắc này áp dụng cả coordinator, Antigravity và subagent; phải đưa vào brief giao việc.
- Lịch sử: ngày 2026-11-12 chủ project đã giao triển khai và chạy B resume epoch 2 -> tổng 4,
  bật lại chọn best theo eval_loss như v5, giữ eval/save mỗi 50 bước, NF4 + FP16.
  Được push đúng dev-caitien, backup rồi cập nhật notebook Kaggle và Save & Run All.
  Không reset optimizer/scheduler; dùng full resume chuẩn Trainer với tổng đích mới,
  ghi rõ khác run đặt 4 epoch ngay từ đầu. Không tự đổi cơ chế khác hoặc bật benchmark.
- Phiên đó đã lỗi hết dung lượng khi lưu bước 500. Lượt sửa hiện tại chỉ sửa local
  và kiểm thử nhẹ, KHÔNG tự dùng quyền của phiên cũ để push hoặc chạy Kaggle lại.
- Sau kiểm thử, chủ project đã giao mới: "ok bắt đầu lại từ epoch 2-4 đi".
  Được push bản sửa cache lên dev-caitien, backup/cập nhật Kaggle và Save & Run All
  B từ parent 619dd8beecbf4eb5a533f4c6618409b7, checkpoint-338 (epoch 2) tới tổng 4.
  Giữ best-300 và toàn bộ cấu hình đã chốt; dùng run/output mới, không lấy 450.
  Đây là triển khai được giao, chưa phải bằng chứng job đã train thành công.
- Ngày 2026-11-15 chủ project giao: "cho chay train tu epoch 4-6 di".
  Được triển khai B full resume từ training/checkpoint-676 của run
  51c046b18b5140a080c387416d4828e8 lên TỔNG 6 epoch, giữ best-500 và thông số cũ.
  Không reset optimizer/scheduler/scaler/RNG hoặc nạp model merge thay checkpoint.
  Được push đúng dev-caitien, backup notebook Kaggle rồi Save & Run All một run mới.
  Giữ cách mở rộng lịch train như phiên 2 -> 4; không coi giống train 6 epoch từ đầu.

## Trạng thái và quy trình hiện hành

- Nhánh dev-caitien là B: cùng backbone Qwen2.5-VL-3B-Instruct, thêm loss ưu tiên mẫu khó.
  B gần nhất đã hoàn tất 4 epoch/676 bước, MLflow 51c046b18b5140a080c387416d4828e8.
  Best theo validation là checkpoint-500, eval_loss=0.08452466130256653.
  Phiên 2 -> 4 dùng bản sửa cache đã hoàn tất trên Kaggle Version 6 (368728517),
  10 giờ 58 phút gồm setup/train/eval/upload. Parent epoch 2 là run 619dd8beecbf4eb5a533f4c6618409b7.
- Phiên resume 75ca700978c840898f9504d17006421c chưa hoàn tất 4 epoch:
  lỗi ổ đĩa khi lưu bước 500; DagsHub có checkpoints/checkpoint-450.
  Mới kiểm tra danh sách file/state của 450, chưa tải và xác minh toàn bộ trọng số.
  Không tự đổi nguồn resume từ 338 sang 450 trước khi xác minh và chốt run tiếp.
- Train mới mặc định 2 epoch, max_steps=-1; resume là lựa chọn riêng khi được giao.
  RESUME_TOTAL_EPOCHS là TỔNG đích: từ epoch 4 lên 6 nghĩa là học thêm khoảng 2 epoch.
- Giữ 8.000 dòng/nguồn trước split; 43.242 train/4.758 val, cùng split manifest A/B.
  Validation dùng loss chuẩn. Giữ batch, seed, backbone/revision, FP16/NF4 và loss B.
- Resume dùng checkpoint huấn luyện đầy đủ, không dùng model merge: adapter/config,
  optimizer, scheduler, trainer_state, scaler FP16 và RNG cho cả hai GPU.
  Tải nguồn vào vùng chuẩn bị riêng; lưu bằng RUN_UUID/output mới, ghi run cha.
- Khi bật load_best_model_at_end, phải giữ cả best checkpoint cũ (B hiện tại là 500),
  không chỉ tải checkpoint cuối 676. Giữ nguyên artifact nguồn trên DagsHub và lưu
  trainer_state gốc làm bằng chứng; chỉ đổi đường dẫn best trong bản tải cho run mới.
  Sau xác minh, DI CHUYỂN best/latest vào output mới trước train để Trainer quản lý
  retention. Không giữ thêm bản nặng/hardlink ở cache ngoài output; không đụng run cũ.
- Giữ save_total_limit=2 và load_best_model_at_end=True: best để chọn inference,
  latest full state để resume. Trainer lưu bản mới TRƯỚC khi dọn bản cũ, cần chỗ cho
  checkpoint thứ ba tạm thời. Kiểm tra dung lượng lúc bắt đầu và trước mỗi lần lưu,
  cộng 1 GiB dự phòng; thiếu thì báo lỗi, không tự xóa checkpoint hay đổi thông số.
- Bản sửa dung lượng và giới hạn kiểm chứng: docs/CHECKPOINT_STORAGE_FIX_2026_11_12.md.
- Khi kéo run đã hoàn tất lên tổng epoch cao hơn, phải trình bày và xin chốt cách xử lý
  lịch cosine; không tự reset scheduler/optimizer hoặc tuyên bố giống train 4 epoch từ đầu.
- Trước khi chạy thật: kiểm tra cấu hình cuối, split, full state và log tải checkpoint;
  sau step 676 phải tiếp tục bộ đếm, không bắt đầu lại từ 0. Kiểm thử CPU không chứng
  minh optimizer/RNG đã được khôi phục trong phiên Kaggle.
- Phiên Kaggle tối đa 12 giờ theo ngân sách đã chốt, gồm setup/eval/upload và ít nhất
  1 giờ dự phòng. Không tự giảm sample/epoch hoặc đổi batch/LR khi thiếu thời gian.
- Cơ chế được rà soát: docs/RESUME_AND_MECHANISM_AUDIT_2026_11_11.md.

## Snapshot ngày 2026-11-06 (chỉ tham khảo)

Đoạn trạng thái cũ dưới đây không phải cấu hình hay lệnh thực thi hiện tại.
Các mục "Trạng thái và quy trình hiện hành" và "Phải hỏi trước khi đổi cơ chế" ưu tiên.

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

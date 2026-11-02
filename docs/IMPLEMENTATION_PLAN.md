# Kế hoạch sửa và cải tiến

## Cách dùng

Làm theo thứ tự P0 -> P1 -> P2 -> P3 -> P4. Mỗi phần cần kiểm tra đạt trước khi chuyển bước. Tài liệu không tự cấp quyền sửa code, chạy GPU, tải artifact hay push Git; thực hiện theo yêu cầu cụ thể của chủ project.

Quy tắc xuyên suốt: không xóa dữ liệu cũ, không tự đổi notebook chính, không dùng thư viện ML cài sẵn của Kaggle, không công bố kết quả chưa đo được. Đọc [runbook](KAGGLE_RUNBOOK.md) trước khi thực hiện các lệnh trên Kaggle.

## P0. Ổn định cách chạy

Trạng thái: **chưa triển khai**.

| Việc | File liên quan hiện có | Kiểm tra để tránh sai |
| --- | --- | --- |
| Chốt notebook chính và provenance | `uni-mumer-kaggle-dagshub-banmoi.ipynb`, `scripts/dagshub_logger.py` | `NOTEBOOK_PATH` đúng; lưu hash/source thực chạy; notebook được Git theo dõi trước clone |
| Python riêng cho toàn pipeline | Notebook, `scripts/runtime_yaml.py`, `requirements.txt` | Không import helper YAML/ML trong kernel Kaggle; subprocess dùng interpreter riêng; import paths đúng env |
| Dependency tái lập | `requirements.txt`, framework đang vendored | Giải quyết cảnh báo dependency, `pip check`, cài fresh-session; khóa phiên bản đã đo, không tùy tiện nâng torch |
| Cấu hình T4 | Notebook, `train/Uni-MuMER-train.yaml`, `scripts/kaggle_test_adapter.py` | FP16 cho train/infer; kiểm tra GPU thực; không coi 2 GPU là VRAM gộp |
| Chỉ một cấu hình cuối cùng | `scripts/runtime_yaml.py`, notebook | Base + overrides resolve đúng; `None` không xóa giá trị; kiểm tra khóa mới và ghi diff cấu hình |
| Tách từng thí nghiệm | Notebook, logger, train wrapper | UUID ổn định trong một run; output/config/results/state riêng; rerun cell không trộn run |
| Kiểm tra trước train | Helper mới, chỉ thêm khi triển khai | Dừng sớm khi sai môi trường, thiếu file, sai model/revision hoặc config không hợp lệ |

Giữ model, rank/alpha, NF4, giới hạn ảnh/chuỗi làm baseline. Không vừa sửa pipeline vừa thay phương pháp nghiên cứu. `scripts/kaggle_config.py` tồn tại nhưng notebook hiện tự cấu hình; nếu tái sử dụng helper này thì phải loại bỏ nguồn cấu hình trùng lặp có chủ đích.

### Kiểm thử P0

- [ ] Notebook JSON hợp lệ, cell không bị dính dòng; kiểm tra cú pháp Python/Bash mà không thực thi training.
- [ ] Unit test resolve YAML: override đúng, `None`, khóa sai, đường dẫn tuyệt đối/tương đối, chế độ không override, giá trị resume cũ.
- [ ] Profile T4 không thể resolve thành BF16; GPU khác không âm thầm chạy profile T4.
- [ ] Kiểm thử đường dẫn interpreter/CLI và việc báo lỗi dependency; không in secrets.
- [ ] Diff chỉ chứa thay đổi dự kiến; không làm mất thay đổi cục bộ trước đó.

## P1. Không mất hoặc nhầm checkpoint

Trạng thái: **chưa triển khai**. File chính: `scripts/train_with_checkpoint_sync.py`, `scripts/dagshub_checkpoint.py`, `scripts/dagshub_logger.py` và notebook.

1. Thêm ba chế độ rõ ràng: train mới, full resume, warm start adapter. Mỗi chế độ cần metadata nguồn và hành vi optimizer/scheduler được định nghĩa.
2. Tái dùng cơ chế tìm MLflow run theo tag trong logger cho upload test. Không dùng `RUN_UUID` tự sinh làm MLflow run ID.
3. Tách sync state theo run ID; hiện trạng chỉ nhớ tên checkpoint có thể khiến run mới bị bỏ qua checkpoint cùng tên.
4. Kiểm tra checkpoint đủ file, đúng base/revision/rank; full resume cần trainer/optimizer/scheduler và RNG/scaler khi áp dụng. Không tự dùng bản local trùng tên từ run khác.
5. Xử lý checkpoint đang ghi, đang upload, retry, retention và final sync. Chỉ báo upload thành công sau khi xác minh manifest remote; không cam kết an toàn chỉ nhờ chờ file đứng yên 20 giây.
6. Log và lưu cấu hình ngay đầu run, không chờ đến upload cuối. Khi job lỗi phải báo rõ trạng thái train và trạng thái bảo toàn checkpoint.

### Kiểm thử P1

- [ ] Mock MLflow: tìm đúng run, không tìm thấy, nhiều run trùng tag, lỗi upload/retry.
- [ ] Hai run cùng có `checkpoint-50` không dùng nhầm state/cache.
- [ ] Checkpoint thiếu config/weights/state bị từ chối đúng chế độ; không fallback train mới.
- [ ] Không có hai writer ghi sync state đồng thời; gián đoạn upload không tạo trạng thái thành công giả.
- [ ] Tải lại vào thư mục mới, đối chiếu hash và thử resume có giới hạn sau khi được phép.

**Cổng kiểm tra kỹ thuật:** sau P0/P1, thực hiện smoke test 5-10 bước và 1-3 ảnh theo runbook nếu được phép. Không mở training dài khi smoke test chưa qua.

## P2. Dữ liệu và đánh giá đáng tin

Trạng thái: **chưa triển khai**. File liên quan: `train/dataset_info.json`, YAML, `scripts/kaggle_test_adapter.py`, `scripts/kaggle_full_test.py`, `scripts/eval_metrics_calculator.py`.

1. Tạo manifest ảnh gốc, dataset/revision và nhãn chuẩn. Ghép các phiên bản HMER/Tree-CoT/EDL/Counting bằng ID/hash ảnh; lấy LaTeX đúng, không lấy nhãn lỗi EDL làm ground truth cấu trúc.
2. Chia train/validation theo ảnh trước khi tạo/trộn nhiệm vụ. Kiểm tra ảnh trùng giữa nguồn nếu có; giữ test riêng. Chỉ tăng `val_size` không giải quyết rò rỉ cùng ảnh.
3. Dùng validation cố định qua `eval_dataset`, `val_size: 0`; đếm mẫu và phân bố nhóm. Khi dùng checkpoint cũ, kiểm tra validation có từng được checkpoint đó học không.
4. Chuyển benchmark thành có kiểm tra coverage, lỗi rõ ràng và chế độ subset có nhãn riêng. Thêm giới hạn mẫu để thử nhỏ; số liệu tổng hợp lấy từ các bản ghi thực chấm.
5. Generation cố định, cùng prompt/preprocessing; giữ cả prediction thô và phiên bản chuẩn hóa. Đo EM biểu thức, CER đúng đơn vị; phân biệt token accuracy và EM.
6. Chọn checkpoint bằng validation/đường dẫn xác nhận trong Trainer state, không theo thời gian sửa thư mục. Tách tham khảo `example_data` khỏi baseline tự chạy.
7. Lưu metrics dạng JSON có schema và provenance; bảng text chỉ là cách trình bày. Phân biệt micro/macro khi tổng hợp và không tái dùng file kết quả cũ sau lần chạy lỗi.

### Kiểm thử P2

- [ ] Manifest train/validation không giao ảnh; cùng ảnh ở hai nhiệm vụ không bị tách qua hai phía.
- [ ] Ảnh thiếu, ID trùng/thừa, nhãn thiếu, JSON sai và lỗi inference hạ tầng đều được phát hiện.
- [ ] Dự đoán rỗng hợp lệ được tính sai, không bị bỏ khỏi mẫu số.
- [ ] Metrics kiểm tra bằng ví dụ nhỏ tính tay; báo đúng số mẫu và kiểu chuẩn hóa.
- [ ] Full benchmark đủ ID và đủ 986/1.147/1.199 mẫu; subset không thể giả làm full.
- [ ] Cùng dữ liệu và cấu hình cho kết quả tái lập trong sai số được ghi nhận; không tự tuyên bố mọi GPU đều bitwise-identical.

## P3. Baseline và tác dụng từng nhiệm vụ

Trạng thái: **chưa triển khai**. Chỉ bắt đầu sau P2; giữ kết quả cũ làm lịch sử, không viết đè.

1. Tạo module phân nhóm cấu trúc và đánh giá nhóm. Tên dự kiến trong `scripts/`: `structure_features.py`, `evaluate_structure.py`; các file này chưa được tạo.
2. Bản đầu dùng phân số, căn, mũ, chỉ số dưới, độ dài; kiểm tra LaTeX lồng nhau, macro không hỗ trợ và tỷ lệ parse thất bại. Không gọi số ngoặc là độ sâu cây toán học nếu chưa có parser phù hợp.
3. Một công thức có nhiều nhãn; không đếm nhân số ảnh khi tổng hợp. Báo số mẫu mỗi nhóm; "sai trong nhóm có số mũ" chưa chứng minh "sai số mũ".
4. Chốt baseline HMER và các đối chứng HMER + Tree-CoT, HMER + EDL, HMER + Counting; kiểm soát tỷ lệ CROHME/HME100K và cùng nguồn ảnh khi đo tác dụng nhiệm vụ.
5. Giữ cùng điểm xuất phát và ngân sách; ghi optimizer steps, token, giờ GPU, thời gian validation. Lặp seed khi ngân sách cho phép và báo độ dao động.
6. Ước lượng tác dụng từng nhiệm vụ trên validation. Ghi cả tác dụng âm/không chắc chắn; không gán quan hệ giúp đỡ theo trực giác.

**Đạt khi:** có bảng baseline, lỗi theo nhóm và bằng chứng về tác dụng nhiệm vụ. Không khẳng định đạt EM 75-76% trước khi đo. Các thí nghiệm khởi đầu từ checkpoint 700 chỉ là kiểm chứng nhanh nếu chưa chứng minh được validation sạch đối với lịch sử train của nó.

## P4. Adaptive sampling theo giai đoạn

Trạng thái: **chưa triển khai**. Tên module dự kiến: `task_affinity.py`, `adaptive_sampler.py`, `adaptive_train.py` trong `scripts/`; không phải file đã có.

1. Giữ loss SFT và forward pass hiện tại. Có thể tận dụng interleave của framework nhưng phải xử lý được nhóm cấu trúc, không chỉ tỷ lệ sáu dataset.
2. Bản đầu giữ bảng affinity cố định, cập nhật lỗi validation theo giai đoạn. Có sàn HMER, phần lấy mẫu nền, giới hạn thay đổi trọng số và xử lý nhóm ít mẫu/nhóm rỗng.
3. Định nghĩa rõ xử lý multi-label và nhiệm vụ không có mẫu của một nhóm. Kiểm tra xác suất hữu hạn, không âm, tổng bằng 1; log phân bố thực lấy, không chỉ trọng số yêu cầu.
4. Lưu phase, sampler state, RNG, manifest và lịch learning rate. Khi thay dataloader phải kiểm soát data skipping, optimizer/scheduler, step và số mẫu; không coi việc đổi YAML rồi resume là đã đúng.
5. So sánh fixed sampling, error-only và error + affinity; cùng ngân sách. Tính cả chi phí đo affinity và validation, không chỉ thời gian train chính.
6. Chọn phương án/checkpoint bằng validation, đánh giá test sau khi khóa thiết lập. Không dùng CROHME test làm phản hồi cho sampler.

**Đạt khi:** có kiểm thử xác suất, resume đúng trạng thái, log phân bố và đối chứng đáng tin; báo cả trường hợp không cải thiện. Chưa thêm backbone/attention mới hoặc ứng dụng chấm bài vào giai đoạn này.

## Bằng chứng và bàn giao

Mỗi bước cập nhật [trạng thái](README.md) và báo:

- File đã đổi, lý do và phạm vi; thay đổi nào của người dùng đã được giữ nguyên.
- Test/lệnh thực sự chạy, môi trường chạy và kết quả; không tính lệnh chỉ viết trong docs là đã chạy.
- Git commit/notebook hash, resolved YAML, dependency lock, dataset/split hash, GPU/world size và đường dẫn artifact (không có secret).
- Việc chưa kiểm chứng và điều kiện dừng. Chỉ ghi "chạy được trên Kaggle" khi có log fresh-session Kaggle đạt các cổng kiểm tra.

Chưa chạy train hoặc benchmark trong lần tạo bộ tài liệu này. Bước tiếp theo là triển khai P0 khi chủ project yêu cầu.

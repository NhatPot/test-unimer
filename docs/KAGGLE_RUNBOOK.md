# Chạy Kaggle: hướng dẫn ngắn

Kiến trúc: GitHub giữ source/workflow, Kaggle chỉ cấp compute.
Thiết kế A/B: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).
Đây là hướng dẫn, không tự cấp quyền push/cài đặt/tải dữ liệu/chạy GPU.
Trước khi thay/xóa cơ chế hoặc công nghệ, giải thích ảnh hưởng và chờ chủ project
xác nhận; không xem yêu cầu sửa lỗi/dọn notebook là quyền bỏ chức năng.

## Trước khi chạy

1. Chốt baseline hay cải tiến, subset/manifest, mặc định 2 epoch cho train mới, split ảnh 90/10,
   eval/save theo step, cấu hình cuối và ngân sách. Không dùng checkpoint700.
2. Sửa source local, backup notebook trước sửa; xem diff, chạy test nhẹ.
   Notebook giữ thứ tự quen thuộc: cấu hình -> setup -> clone -> thư viện -> YAML -> train -> upload.
   Chỉ thêm kiểm tra thiết yếu, không bắt điền nhiều cờ/commit/hash vào cell.
3. Khi được giao: commit/push đúng các file liên quan; source trên GitHub phải có
   đủ helper/requirements mà notebook gọi. Không stage toàn bộ thay đổi cục bộ.
   Chốt baseline commit trước khi tạo dev-caitien; không đổi source đang train.
4. Trước MỌI sửa Kaggle: kiểm tra job đang chạy, export backup đọc được, ghi URL/version.
   Không tự dừng job/ghi đè notebook đang chạy. Đối chiếu bản cập nhật với source đã chốt.

## Thực thi khi đã được giao run

- Kaggle clone đúng nhánh/commit; in commit thực tế. Không tự pull giữa job.
- Giữ Conda/Python 3.10 riêng. Mọi ML/YAML/train/upload dùng PYTHON của env,
  cài requirements project và framework; không dùng ML packages của kernel Kaggle.
- Kiểm tra pip check và GPU thật. Profile hiện tại: 2 x T4, FP16; không xem VRAM là gộp.
  Kernel notebook chỉ điều phối, Kaggle Secrets cấp token; không in token hoặc toàn bộ env.
- In/lưu resolved YAML và split. Không tin mỗi YAML nền vì notebook có override.
  Với eval_dataset đã chia theo ảnh, val_size: 0; validation vẫn là 10% ở manifest.
- Dùng output/run DagsHub mới, nhãn baseline hoặc improved. RUN_UUID là tag,
  phải lấy actual MLflow run ID; lưu source commit, split, thông số và checkpoint step.
- Chạy Save Version -> Save & Run All trong ngân sách đã giao. Không cần hỏi lại
  từng bước nhỏ trong phạm vi đó; không tự thêm run dài/thay cấu hình khi gặp lỗi.
- Job train/upload lỗi phải báo lỗi và giữ checkpoint local. Không xóa bản duy nhất
  trước khi xác minh bản đã lưu; không gọi code sync mới chỉ vì file đó tồn tại.
- Chỉ báo hoàn tất khi có log train, các mốc validation và checkpoint/artifact đúng run.
  Mốc cuối phải phân biệt với best model; nếu thiếu thì báo thiếu, không tự lấp metric.

## Resume khi được giao

- Cell 4.5 khôi phục lựa chọn resume từ v5. Train mới chỉ tắt công tắc, không xóa cell.
- Chọn đúng actual run ID/checkpoint và tổng epoch đích; không dùng model merge.
  Nguồn tải vào vùng chuẩn bị riêng; output/run mới ghi run cha, không ghi đè bản cũ.
- Quyết định 2026-11-12: bật best theo eval_loss. Giữ checkpoint cuối để resume,
  best để merge/infer. Nếu state trỏ best ở phiên cũ, tải cả best; chỉ sửa đường dẫn
  trong bản tải cho run mới, giữ JSON gốc làm bằng chứng; không reset best_metric
  hoặc optimizer/scheduler. Sau kiểm tra, wrapper di chuyển best/latest vào output
  để Trainer quản lý retention; cache không được giữ thêm weights/optimizer/hardlink.
- Giữ save_total_limit=2; dự trù checkpoint thứ ba tạm thời vì lưu xong mới rotation.
  Kiểm tra chỗ trống lúc bắt đầu và trước mỗi lần lưu, cộng 1 GiB dự phòng.
  Thiếu dung lượng thì báo lỗi trước khi lưu, không tự xóa checkpoint hoặc đổi cấu hình.
- Kiểm tra full state và manifest; không tự chuyển sang train mới khi resume lỗi.
- Kéo một run đã hoàn tất lên tổng epoch cao hơn phải xin chốt lịch scheduler trước;
  không tự reset optimizer/scheduler hoặc coi giống run dài ngay từ đầu.
- Chi tiết và các cơ chế còn cần duyệt:
  [RESUME_AND_MECHANISM_AUDIT_2026_11_11.md](RESUME_AND_MECHANISM_AUDIT_2026_11_11.md).
- Bản sửa mới chỉ kiểm thử local; chưa được coi là đã triển khai lên Kaggle.
- Chi tiết lỗi ổ đĩa và bản sửa 2026-11-12:
  [CHECKPOINT_STORAGE_FIX_2026_11_12.md](CHECKPOINT_STORAGE_FIX_2026_11_12.md).

## Khi chủ project yêu cầu phân tích

- Đọc đúng run baseline/improved từ DagsHub; kiểm tra config, sample/split, step và trạng thái.
- Phân tích loss/log trước; không tự tải weights lớn hoặc mở GPU benchmark nếu chưa được giao.
- Inference/EM/CER/benchmark chính ở Uni-MuMER-project trên RTX 3070, cùng cách chấm cho A/B.
  Validation không trùng train; CROHME test không dùng để chọn mẫu khó/chọn siêu tham số.
- Lưu kết quả riêng, không đè kết quả cũ. Không hứa luôn theo dõi nền khi chưa có cơ chế đó.

## Bàn giao và giới hạn

Báo tách rõ: sửa local / đã push GitHub / đã sửa Kaggle / đã chạy / kết quả ở đâu.
CPU tests không chứng minh Kaggle tương thích. Không tuyên bố tự động end-to-end
đã chạy khi mới có notebook và helper. Tự động triển khai/chạy còn cần kiểm chứng thực tế.

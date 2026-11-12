# Sửa dung lượng khi full resume

## Lỗi đã gặp

- Kaggle version 5, scriptVersionId `368533933`, B resume từ epoch 2 lên tổng 4.
- MLflow: `75ca700978c840898f9504d17006421c`.
- Sau khoảng 4 giờ 56 phút, ghi optimizer của checkpoint-500 thất bại;
  log có `No space left on device`. Đây là ổ đĩa, không phải GPU hết VRAM.
- Best/latest nguồn còn bị giữ ở cache ngoài output. Trainer xóa bản trong output
  nhưng hardlink ở cache vẫn giữ dung lượng. Ngoài ra, Trainer lưu checkpoint mới
  trước khi rotation, nên save_total_limit=2 vẫn cần chỗ cho bản thứ ba tạm thời.

## Bản sửa local

1. Tải và xác minh latest full checkpoint cùng best của nó, như trước.
2. Lưu trainer_state gốc nhỏ để đối chiếu; di chuyển các bản tải vào vùng chuẩn bị,
   chỉ chỉnh đường dẫn best trong state làm việc. Không sửa artifact trên DagsHub.
3. Trước train, di chuyển nguyên vùng chuẩn bị vào output mới bằng rename cùng
   filesystem; YAML runtime chỉ đổi đường dẫn resume/adapter tương ứng.
   Cache ngoài output không còn giữ weights hoặc optimizer.
4. Để Trainer giữ best/latest bằng save_total_limit=2 như cũ. Kiểm tra chỗ trống
   trước train và trước lưu, tính cả save-before-rotation và 1 GiB dự phòng.
   Thiếu chỗ thì báo lỗi, không tự xóa file hay giảm cấu hình.

File chính: scripts/checkpoint_storage.py, scripts/dagshub_checkpoint.py,
scripts/train_with_checkpoint_sync.py; notebook gọi cùng workflow này.
Trainer chỉ thêm kiểm tra dung lượng khi wrapper truyền kế hoạch storage.

## Những gì không đổi

Full resume gồm adapter, optimizer, scheduler, scaler và RNG; validation/save mỗi
50 bước, best theo eval_loss, NF4 + FP16, batch, dữ liệu/split và loss B giữ nguyên.
Best dùng để chọn merge/inference; latest full checkpoint dùng để học tiếp.
Không thêm early stopping hoặc tự đổi scheduler khi kéo dài epoch.
Không xóa checkpoint, kết quả hoặc artifact của những run trước.

## Kiểm chứng và bước sau

- 193 kiểm thử local qua, GPU đã ẩn trong lệnh kiểm thử. Có thử Trainer thật với
  mô hình rất nhỏ trên CPU: resume bước 2 lên 4, nạp optimizer/RNG, giữ best/latest.
  Có kiểm tra thiếu dung lượng, file thay đổi và output có sẵn phải dừng an toàn.
- Tại thời điểm kiểm thử bản sửa: chưa push/cập nhật/chạy lại Kaggle.
  CPU test không chứng minh full resume QLoRA hai T4 đã thành công.
- Sau đó chủ project đã giao chạy lại epoch 2 -> 4 từ checkpoint-338, giữ best-300.
  Trạng thái push/triển khai thực tế được ghi riêng trong docs/reports/, không suy
  ra từ việc đã được giao hoặc từ kết quả kiểm thử CPU.
- Run lỗi có thư mục `checkpoints/checkpoint-450`, epoch khoảng 2,663. Hiện mới
  kiểm tra danh sách file và trainer_state, chưa xác minh toàn bộ trọng số.
  Không coi run này đã hoàn tất 4 epoch hoặc có final model.
- Khi được giao chạy tiếp: xác minh đầy đủ checkpoint nguồn, chốt 338 hay 450,
  rồi push đúng dev-caitien, backup notebook Kaggle và triển khai phiên mới.
  Không tự đổi nguồn resume hoặc bỏ kiểm tra manifest vì run lỗi thiếu final upload.

Backup local trước sửa:
`/home/nhat/Uni-MuMER-project/data/kaggle_backups/2026-11-12-storage-fix-54I2zf/`.

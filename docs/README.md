# Tài liệu vận hành và cải tiến Uni-MuMER

## Đọc theo thứ tự

1. [AGENTS.md](../AGENTS.md): quy tắc bắt buộc cho agent khi làm việc trong repo.
2. [Kế hoạch triển khai](IMPLEMENTATION_PLAN.md): sửa gì trước, kiểm tra gì, khi nào được chuyển bước.
3. [Hướng dẫn Kaggle](KAGGLE_RUNBOOK.md): môi trường riêng, cấu hình T4, checkpoint và điều kiện chạy.

Các tài liệu này là kế hoạch và tiêu chí nghiệm thu, không phải thông báo rằng code đã được sửa hoặc đã chạy thành công trên Kaggle.

## Project nào làm việc gì?

| Vị trí | Vai trò |
| --- | --- |
| `/home/nhat/test-unimer` | Repo đang phát triển, chuẩn bị thí nghiệm và chạy Kaggle |
| `uni-mumer-kaggle-dagshub-banmoi.ipynb` | Notebook chính do chủ project xác nhận |
| `train/Uni-MuMER-train.yaml` | Cấu hình nền; notebook có thể ghi đè |
| `train/dataset_info.json` | Đăng ký dataset đang được YAML sử dụng |
| `scripts/` | Train, đồng bộ checkpoint, tải checkpoint, inference và đánh giá |
| `/home/nhat/Uni-MuMER-project` | Kết quả/model cũ để lưu trữ và đối chiếu; không tự sửa |

Không dùng bản trong `_output_` hay notebook v5/v8 thay cho notebook chính chỉ vì tên có vẻ mới hơn. Một số tài liệu ablation cũ vẫn còn, nhưng các script/config mà chúng nhắc đến không còn trên nhánh hiện tại.

## Mốc kiểm tra ban đầu

Ngày ghi nhận: 2026-11-02. Git HEAD của `test-unimer`: `72274d5afd420054fb2e0478aa931ce3ccc09a88`.

- Notebook `banmoi` đang là file untracked tại mốc này; bản đang chạy cần được lưu vào Git trước khi tái lập qua clone.
- `requirements.txt` đã có thay đổi cục bộ của người dùng; không được ghi đè để làm sạch Git.
- Có các script/bảng thống kê và notebook untracked khác; không xóa hoặc gộp vào commit ngoài phạm vi.
- Cấu hình sau override của notebook: 6 nguồn, `max_samples: 8000` mỗi nguồn, 2 epoch, batch/GPU 2, accumulation 64, LoRA rank 64/alpha 16, NF4 4-bit.
- Notebook không ghi đè BF16/FP16 nên hiện vẫn kế thừa `bf16: true`, `fp16: false`. Đây chưa phải cấu hình chấp nhận cho profile T4.
- Notebook còn khai báo `NOTEBOOK_PATH` là v5, chưa nối bước tải checkpoint để resume, và đang tắt tải ảnh test/benchmark.
- Notebook đang import helper YAML trong kernel Kaggle. Cần chuyển phần này sang Python riêng, không chỉ chuyển lệnh train.
- Output đã lưu không có bằng chứng cell train của chính file này hoàn tất. Không dùng output cũ để xác nhận phiên bản code hiện tại chạy được.
- Chưa có kiểm chứng fresh-session Kaggle cho các thay đổi được đề xuất trong tài liệu.

## Kết quả cũ có thiếu mẫu không?

Kiểm tra chỉ đọc trong phiên làm việc này đã đối chiếu 24 file dự đoán thuộc 8 nhóm với bộ prompts tại `Uni-MuMER-project`:

- `data/CROHME/results` và `results_checkpoint450`.
- Các nhóm `results_checkpoint450_dagshub`, `700_dagshub`, `736_dagshub`, `800_dagshub`, `844_dagshub` (đều có tiền tố `results_checkpoint`).
- `_output_/test-unimer/kaggle_test_results`.

Mỗi nhóm có đủ 986 mẫu CROHME 2014, 1.147 mẫu 2016 và 1.199 mẫu 2019, tổng 3.332. Không thiếu/trùng/thừa ID; ground truth khớp; không có dự đoán rỗng; số mẫu trong báo cáo khớp JSON.

Đây là kiểm chứng độ đầy đủ của các file đang lưu, không phải chạy lại model, chứng minh validation không rò rỉ, hay xác nhận toàn bộ phương pháp đánh giá. Không cần chạy lại chỉ vì cảnh báo code có thể bỏ qua ảnh thiếu.

## Trạng thái công việc

- [x] Xác định notebook chính và đọc các script liên quan.
- [x] Kiểm tra độ đầy đủ của các kết quả cũ nêu trên.
- [x] Viết quy tắc agent, kế hoạch và checklist Kaggle.
- [ ] Sửa môi trường/cấu hình và luồng lưu kết quả.
- [ ] Hoàn thiện checkpoint/resume và kiểm thử không GPU.
- [ ] Chạy smoke test được cho phép trên Kaggle sạch.
- [ ] Chuẩn hóa split và đánh giá để làm baseline mới.
- [ ] Chạy ablation, rồi mới triển khai adaptive sampling.

Chỉ đánh dấu hoàn thành khi có bằng chứng. Mọi lần train dài, cài đặt, tải dữ liệu/model hay thao tác DagsHub phải nằm trong phạm vi được người dùng cho phép.

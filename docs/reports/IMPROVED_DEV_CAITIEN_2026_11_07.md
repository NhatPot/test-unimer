# Pilot cải tiến dev-caitien

## Đã chuẩn bị local

- Nhánh `dev-caitien`, checkout riêng `/home/nhat/orca/workspaces/test-unimer/dev-caitien`.
- Tạo từ đúng commit baseline `0f65e4cc4c67b042998b84c5ac068abeeca09719`.
- Baseline A: MLflow `3aaa5b8d882f4555a17d2c139e38513c`, 5411 train/589 val,
  1 epoch/22 global steps, final val loss `0.5565038323402405`.
- B giữ Qwen2.5-VL-3B/QLoRA, không thêm backbone hoặc resume adapter cũ.
- Train loss tăng ưu tiên mẫu có mean answer-token CE cao hơn microbatch mean:
  `weight = clamp(1 + 0.5 * relu(difficulty / mean_difficulty - 1), 1, 1.5)`.
  Difficulty và weight không nhận gradient. Chia loss theo tổng token đã nhân trọng số.
- Validation/prediction luôn dùng CE gốc. `use_hard_example_loss` mặc định tắt
  trong framework, notebook nhánh B bật rõ tại cell 5.
- Antigravity triển khai loss và 18 kiểm thử; coordinator review và nối runtime/notebook.
- 163 kiểm thử CPU qua, gồm test loss/gradient, trainer thật với model giả nhỏ,
  notebook/runtime và chặn split/revision sai. Không suy ra Kaggle đã chạy từ CPU tests.

## Những gì giữ nguyên

- 1000 dòng mỗi nguồn trong 6 nguồn, split nhóm ảnh 10%, seed42, 1 epoch.
- 2xT4, FP16, NF4 4-bit, rank64/alpha16/dropout0.05, LR1e-4,
  batch2/GPU, accumulation64, cutoff2048, image_max_pixels262144.
- Eval/save mỗi5 bước, load_best_model_at_end=false, đo cuối riêng, output/run mới.
- Model revision `66285546d2b821cf421d4f5eb2576359d3770cd3`
  được xác nhận từ cache path trong log baseline.
- Dataset revision `8fcc66de1298524c9e47f542a726090c44532e6e`.
- Split SHA256 bắt buộc khớp A:
  `739df01842b5ccfc9c9c7d94eb1d8a826212a25053558671bb1453ee50a86590`.
- Source helper nhận pin revision, chặn runtime nếu split hash khác. Không đổi registry
  hay cách chia/lấy subset; runtime không cap/chia lần hai.

## Notebook và triển khai

- File vẫn `uni-mumer-kaggle-dagshub-banmoi.ipynb`, nhưng trên nhánh `dev-caitien`.
- Clone rõ `dev-caitien`; giữ setup Python3.10 riêng và dependency baseline.
- Nhãn DagsHub `variant=improved`, `experiment_variant=improved`.
- Không thêm thuật toán vào notebook; chỉ thêm tham số và định danh A/B.
- RUN_TRAINING=False trong source; bản triển khai Kaggle bật có chủ đích.
- Backup trước sửa:
  `/home/nhat/Uni-MuMER-project/data/kaggle_backups/2026-11-07-improved-001/`.
- Trạng thái tại lúc ghi: CHƯA push, CHƯA upload Kaggle, CHƯA smoke GPU hoặc full run.
  Xem bản ghi triển khai local trong thư mục backup để biết các bước thực tế sau đó.

## Giới hạn phải báo khi so sánh

- Độ khó chỉ tương đối trên microbatch 2 mẫu/GPU; không chuẩn hóa theo task.
  Task vốn loss cao có thể được ưu tiên. Đây là pilot, không bảo đảm tốt hơn A.
- Train weighted loss của B không so trực tiếp train loss chuẩn A; dùng val CE chuẩn
  cùng step/cuối epoch, token accuracy và thời gian. Chưa chấm ExpRate/CER/test.
- Baseline A có NaN grad_norm bước1; giữ nguyên precision để không thay thêm yếu tố.
  Theo dõi xem có lặp lại; không gọi global_step là số update thành công đã xác minh.
- Không đổi/xóa kết quả A, không tự sửa dependency hay budget khi gặp lỗi B.

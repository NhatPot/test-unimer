# Bắt đầu ở đây

## Chỉ cần đọc

1. [AGENTS.md](../AGENTS.md): quy tắc cho mọi AI agent.
2. [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md): phương án thí nghiệm hiện tại.
3. [KAGGLE_RUNBOOK.md](KAGGLE_RUNBOOK.md): chỉ đọc khi chuẩn bị GitHub/Kaggle.

Không cần đọc hết lịch sử để bắt đầu một việc nhỏ.

## Phương án đã chốt

Train mới A: QLoRA cũ và B: QLoRA ưu tiên mẫu khó từ cùng Qwen base + LoRA mới.
Cùng tập nhỏ, 1 epoch, train/validation 90/10 theo ảnh nguồn; đánh giá theo step.
Checkpoint 700 giữ làm lịch sử, không dùng khởi tạo. Nhánh B dự kiến dev-caitien.

```text
Local sửa source/config -> GitHub -> Kaggle clone
 -> Python riêng + requirements -> train/validation -> DagsHub
 -> AI lấy đúng run về phân tích khi được yêu cầu
 -> inference/benchmark trên RTX 3070 local khi cần
```

Notebook là bảng điều khiển ngắn; workflow nằm trong source, không nằm rải rác
trong các cell dài hoặc chỉnh tay ngoài Git.

## Trạng thái thật

- Baseline dev_tt đã sửa local: 1 epoch, tối đa 1.000 dòng/nguồn, chia val 10% theo ảnh.
- Có helper xuất subset/manifest/registry và YAML runtime; chưa tạo subset thật, chưa train A/B.
- Chưa triển khai cơ chế loss ưu tiên mẫu khó; chưa tạo/push dev-caitien.
- Chưa kiểm chứng bản hiện tại trên Kaggle; tự động triển khai/chạy/lấy kết quả chưa hoàn tất.
- Việc tiếp theo: duyệt diff, đưa đủ source lên GitHub khi được giao, rồi chạy baseline Kaggle.
- File đổi, kiểm thử và giới hạn: [báo cáo baseline](reports/BASELINE_DEV_TT_2026_11_06.md).

## Lịch sử, không phải kế hoạch hiện hành

- [Bản hướng dẫn trước khi chốt A/B](archive/2026-11-05-before-ab-plan/INDEX.md).
- [Các báo cáo kỹ thuật](reports/README.md): chỉ mở báo cáo liên quan khi cần tra cứu.
- [Ghi nhận nhóm agent](TEAM_EXECUTION_2026_11_05.md).

Không lấy checklist cũ “khôi phục 700 rồi warm-start” làm việc tiếp theo.

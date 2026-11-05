# Kế hoạch hiện tại: A/B trên tập nhỏ

Chốt lại 2026-11-06. Code baseline đã sửa local; chưa push hoặc chạy Kaggle.
Đọc AGENTS.md trước. Không dùng kế hoạch checkpoint 700 làm thứ tự triển khai.

## Phạm vi sửa ngay: baseline dev_tt

main là bản lịch sử. CHỈ chuẩn bị baseline; chưa tạo dev-caitien, chưa viết cải tiến.

| Phần | Sửa tối thiểu |
| --- | --- |
| Notebook cấu hình | 1 epoch, giới hạn dự kiến 1.000 bản ghi/nguồn, tỷ lệ val 10%, eval/save dự kiến mỗi 5 step; giữ bảng tham số dễ chỉnh |
| Source dữ liệu | Chuẩn bị subset và split 90/10 theo ảnh, cố định seed/danh sách; không nhúng logic này vào notebook |
| Registry và YAML runtime | Đăng ký train/val, dùng eval_dataset và val_size: 0; không để loader cắt/chia lại manifest đã chuẩn bị |
| Notebook clone | Chọn rõ dev_tt, in branch/commit thực tế; không tự pull hoặc đổi checkout đang có thay đổi |
| Kết quả và upload | Ghi nhãn baseline/split/run ID; tách metric best và cuối run, báo riêng lỗi train/upload |
| Kiểm tra | Backup trước sửa, test CPU/cú pháp/config; chưa push, cài đặt hoặc chạy Kaggle trong bước chuẩn bị |

Thông số được chỉnh tại cell cấu hình notebook, không sửa đồng thời cả notebook
và YAML nền để tránh hai giá trị mâu thuẫn. Helper xuất YAML thực chạy từ cấu hình đó.
Giới hạn sample dùng khi chuẩn bị subset; runtime không được áp cap lần hai.
10% nằm ở manifest theo ảnh, nên val_size runtime bằng 0 không có nghĩa bỏ validation.
Giữ Python riêng, FP16, QLoRA và các thông số nền khác; không nối toàn bộ thư viện
sync/timeout mới hoặc viết lại pipeline nếu sửa hẹp workflow cũ đã đáp ứng.

## Thiết kế

| Nội dung | A: baseline | B: dev-caitien |
| --- | --- | --- |
| Khởi tạo | Qwen2.5-VL-3B-Instruct + LoRA mới | Cùng base/revision, cùng seed và khởi tạo |
| Cách học | Loss QLoRA hiện tại | Tăng trọng số loss mẫu khó trong train, có giới hạn |
| Thời lượng | 1 epoch | 1 epoch |
| Dữ liệu | Cùng danh sách train/validation cố định | Không đổi danh sách hay số lượt học |
| Validation | 10% ảnh nguồn, dùng loss chuẩn | Cùng phép chấm chuẩn, không dùng weighted loss để so A |
| Đánh giá/lưu | Theo step, dự kiến mỗi 5 bước, thêm mốc cuối | Cùng lịch và cách chọn best |
| Model/QLoRA | NF4 4-bit, rank64/alpha16/dropout0.05 | Giữ nguyên |

Tập nhỏ dự kiến: tối đa 1.000 bản ghi mỗi nguồn trong 6 nguồn hiện có.
Chưa coi đây là 6.000 ảnh độc lập; xác nhận danh sách/count sau kiểm kê trước chạy.
Giữ batch/LR/giới hạn ảnh/chuỗi baseline; chốt tổng optimizer step và GPU budget thực tế.
Giữ test CROHME riêng. Không dùng checkpoint 700 làm đầu vào hoặc điểm so A/B trực tiếp.

## Lộ trình tổng thể (B và so sánh để sau)

1. **Dữ liệu dùng chung:** kiểm kê subset, nhận diện ảnh nguồn/ảnh trùng; chia khoảng
   90/10 theo nhóm ảnh, mọi nhiệm vụ của một ảnh cùng phía. Lưu manifest + seed,
   đếm ảnh và bản ghi từng nguồn, kiểm tra giao ảnh bằng 0. Dữ liệu không rõ ID
   phải được ghi rõ, không tự tuyên bố split sạch. Không tải dataset lớn khi chưa được giao.
2. **Chuẩn bị và chạy A:** backup notebook, đặt 1 epoch, chọn subset/manifest đã chốt,
   lịch eval/save theo step và đo checkpoint cuối trước khi nhầm với best nạp lại.
   Test local, chuẩn bị source GitHub đầy đủ; khi được phép thì backup Kaggle,
   cập nhật và Save & Run All. Lưu actual MLflow run ID, config và split.
3. **Chuẩn bị và chạy B:** từ đúng commit baseline đã chốt, tạo dev-caitien khi được giao.
   Chỉ thêm loss ưu tiên mẫu khó, không đổi backbone/sampler hoặc thêm epoch.
   Độ khó dựa trên token đáp án hợp lệ, tránh chỉ ưu tiên chuỗi dài/nhiệm vụ vốn loss cao.
   Chặn trọng số cực đoan; trọng số bằng 1 phải tái hiện loss baseline.
   Dùng cùng dữ liệu, seed, budget và lịch đánh giá; lưu run DagsHub riêng.
4. **So sánh:** cùng mốc step, validation loss chuẩn, thời gian/VRAM và lượng token đã học.
   Nếu cần EM/CER thì sinh dự đoán trên cùng validation ở repo local.
   Khóa phương án trước benchmark test; tập nhỏ chỉ sàng lọc, không bảo đảm cải thiện.
   Báo cả trường hợp không tốt hơn. Muốn kết luận chắc hơn thì tăng dữ liệu/lặp seed sau.

## Tránh làm sai

- 10% là tỷ lệ ảnh nguồn; số dòng validation có thể khác 10% vì số nhiệm vụ mỗi ảnh khác.
- Chỉ đặt val_size: 0.1 trên dữ liệu đa nhiệm KHÔNG đủ bảo đảm không trùng ảnh.
  Khi dùng eval_dataset từ manifest riêng, đặt val_size: 0 để không chia thêm lần nữa.
- max_samples hiện lấy các hàng đầu TỪNG nguồn. Không cap/chia lại ngẫu nhiên sau
  manifest khiến A/B mất mẫu hoặc đổi split; hai nhánh phải đọc đúng manifest đã khóa.
- Validation chỉ đo; không đưa ảnh/nhãn validation vào train để “học bài khó”.
- Một step là optimizer update, không phải một ảnh. Tập nhỏ có thể kết thúc trước step50.
- Đánh giá cuối run phải gắn đúng checkpoint cuối, không gán metric của best model cho nó.
- Thử kỹ thuật 5–10 bước nếu cần là run riêng, KHÔNG thay cho baseline 1 epoch.
- Nhánh baseline hiện là dev_tt nhưng working tree còn thay đổi; chưa có baseline commit
  đã khóa. Không đổi nhánh/reset/stash hay gộp hết file bẩn chỉ để tạo dev-caitien.

## Trạng thái và bàn giao

Đã có helper chuẩn bị subset/split theo ảnh, notebook cấu hình 1 epoch/1.000 dòng mỗi
nguồn/val 10%, eval/save mỗi 5 step. Chưa tạo subset thật nên chưa chốt actual count/step.
YAML nền và registry nguồn giữ nguyên; runtime trỏ registry train/val được helper sinh.
load_best_model_at_end=False để eval sau train đo đúng trọng số cuối; best vẫn theo
validation loss tại các mốc đã lưu, ghi riêng trong trainer_state/metadata/metric upload.
Chi tiết file/test/giới hạn: [báo cáo baseline](reports/BASELINE_DEV_TT_2026_11_06.md).
Nhánh dev-caitien và weighted-loss code chưa tồn tại trong quy trình hiện tại.
Các helper sync/timeout có test CPU nhưng chưa tích hợp đầy đủ, không mặc định gọi chúng.
Mỗi lần bàn giao chỉ cần: file đổi, test đã chạy, run/commit nếu có, việc còn thiếu.
Đọc KAGGLE_RUNBOOK.md khi sắp triển khai; không biến kế hoạch này thành nhiều gate trong notebook.

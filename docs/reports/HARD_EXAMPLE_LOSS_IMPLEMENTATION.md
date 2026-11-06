# Hard-Example-Weighted Training Loss Pilot (`dev-caitien`)

Cập nhật: 2026-11-07
Nhánh: `dev-caitien` (từ baseline commit `0f65e4cc4c67b042998b84c5ac068abeeca09719`)
Baseline tham chiếu (A): MLflow run `3aaa5b8d882f4555a17d2c139e38513c` (`5411` train / `589` val, `1` epoch / `22` global steps, final `eval_loss = 0.5565038323402405`).

---

## 1. Phạm vi và các file thay đổi

Chỉ sửa và thêm các file trong phạm vi source + test CPU được giao (giữ nguyên `AGENTS.md`, notebook, `scripts/runtime_yaml.py`, `train/Uni-MuMER-train.yaml` và checkout `dev_tt`):

1. `train/LLaMA-Factory/src/llamafactory/hparams/finetuning_args.py`
   - Thêm 3 trường cấu hình vào `FinetuningArguments` và kiểm tra hợp lệ trong `__post_init__`.
2. `train/LLaMA-Factory/src/llamafactory/train/sft/hard_example_loss.py`
   - Helper tính toán `compute_hard_example_weighted_loss`, `validate_hard_example_loss_config` và `HardExampleLossDetails`.
3. `train/LLaMA-Factory/src/llamafactory/train/sft/trainer.py`
   - Tích hợp nhánh hard-example loss vào `CustomSeq2SeqTrainer.__init__` và `CustomSeq2SeqTrainer.compute_loss`.
4. `tests/test_hard_example_loss.py`
   - Bộ kiểm thử CPU tách rõ giữa **pure helper unit tests** (`10` test methods) và **actual `CustomSeq2SeqTrainer` integration tests** (`8` test methods).
5. `docs/reports/HARD_EXAMPLE_LOSS_IMPLEMENTATION.md`
   - Báo cáo chi tiết thiết kế, cấu hình, kết quả kiểm thử CPU và giới hạn còn lại.

---

## 2. Cấu hình và kiểm tra hợp lệ (`FinetuningArguments`)

| Tên trường trong `FinetuningArguments` | Kiểu | Mặc định | Quy tắc kiểm tra hợp lệ |
| --- | --- | --- | --- |
| `use_hard_example_loss` | `bool` | `False` | Phải là `bool`. Mặc định `False` giữ tương thích ngược hoàn toàn với baseline A. Khi bật `True`, chỉ cho phép `stage == "sft"`. |
| `hard_example_loss_alpha` | `float` | `0.5` | Phải là số thực hữu hạn (`int`/`float`, không phải `bool`), `>= 0.0`. |
| `hard_example_loss_max_weight` | `float` | `1.5` | Phải là số thực hữu hạn (`int`/`float`, không phải `bool`), `>= 1.0`. |

### Chặn các chế độ loss không tương thích

`FinetuningArguments.__post_init__` và `CustomSeq2SeqTrainer._validate_hard_example_loss_setup` từ chối rõ ràng (`ValueError`) thay vì âm thầm kết hợp khi `use_hard_example_loss=True`:

- `stage != "sft"`
- `use_dft_loss=True`
- `compute_loss_func is not None`
- `label_smoother is not None` hoặc `training_args.label_smoothing_factor != 0`
- `model_args.enable_liger_kernel=True` hoặc `training_args.use_liger_kernel=True`
- Thiếu `"labels"` trong `inputs` khi train hoặc `outputs` của model không có `"logits"` (ví dụ kernel fused-CE ẩn logits).

---

## 3. Thuật toán và tối ưu bộ nhớ (Memory Engineering cho 2xT4 15GB)

### 3.1. Công thức toán học

Với mỗi mẫu $i \in \{0, \dots, B-1\}$ trong microbatch:

1. **Causal shift & mask**:
   - Dùng `logits[i, :-1, :]` dự đoán `shift_labels_i = labels[i, 1:]`.
   - Chỉ lấy các vị trí đáp án hợp lệ $t$ có `shift_labels_i[t] != IGNORE_INDEX` (`-100`). Các token hệ thống/prompt/ảnh/padding và vị trí `labels[i, 0]` hoặc `logits[i, -1, :]` không đóng góp vào loss và có gradient bằng `0`.
2. **Tổng CE hợp lệ và độ khó mẫu ($d_i$)**:
   - Với mẫu hợp lệ ($n_i = \text{count\_valid\_tokens}_i > 0$):
     $$\text{sum\_valid\_token\_ce}_i = \sum_{t \in \text{valid}(i)} \text{CE}_{\text{FP32}}\!\left(\text{logits}_{i, t}, \text{shift\_labels}_{i, t}\right)$$
     $$d_i = \text{detach}\!\left(\frac{\text{sum\_valid\_token\_ce}_i}{n_i}\right)$$
   - Độ khó $d_i$ là trung bình CE trên mỗi token đáp án hợp lệ của mẫu $i$ (tránh ưu tiên mẫu chỉ vì chuỗi đáp án dài) và được **detach hoàn toàn khỏi đồ thị gradient**.
3. **Độ khó tương đối và trọng số có trần ($w_i$)**:
   - Trung bình độ khó microbatch chỉ tính trên các mẫu hợp lệ ($n_i > 0$):
     $$\bar{d} = \frac{1}{|\{i : n_i > 0\}|} \sum_{i : n_i > 0} d_i$$
   - Trọng số mỗi mẫu hợp lệ (trong khối `torch.no_grad()`):
     $$w_i = \text{clamp}\!\left(1 + \alpha \cdot \text{relu}\!\left(\frac{d_i}{\bar{d}} - 1\right),\, 1.0,\, \text{max\_weight}\right)$$
   - Nếu $\bar{d} \le 0$ (mọi token hợp lệ có CE bằng `0`), đặt $w_i = 1.0$ để tránh chia `0/0`. Không dùng `nan_to_num` để che giấu `NaN`/`Inf` thật từ `logits`.
4. **Loss huấn luyện có trọng số**:
   $$\mathcal{L}_{\text{train}} = \frac{\sum_{i} w_i \cdot \text{sum\_valid\_token\_ce}_i}{\sum_{i} w_i \cdot n_i}$$
   - Khi $\alpha = 0$ hoặc $\text{max\_weight} = 1.0$ (tức $w_i \equiv 1$), công thức rút gọn về đúng $\frac{\sum_i \text{sum\_valid\_token\_ce}_i}{\sum_i n_i}$, khớp hoàn toàn với `ForCausalLMLoss` chuẩn cả về giá trị loss và gradient.
   - Nếu toàn bộ microbatch không có token hợp lệ nào ($\sum_i n_i = 0$), trả về scalar `0.0` khả vi (`logits.reshape(-1)[:0].float().sum()`) để không chia cho `0` và vẫn giữ graph hợp lệ.

### 3.2. Tránh nhân đôi đồ thị CE và giảm đỉnh VRAM

- Trong `Qwen2_5_VLForConditionalGeneration.forward`, nếu truyền `labels` vào `model(**inputs)`, model sẽ tự gọi `ForCausalLMLoss` và ép kiểu toàn bộ `logits` `(B * S, V)` sang FP32, giữ đồ thị backward của `outputs.loss` trong bộ nhớ.
- Trong `CustomSeq2SeqTrainer.compute_loss`, khi ở nhánh train có `use_hard_example_loss=True`:
  1. Tạo bản sao nông `model_inputs` không chứa `"labels"` khi gọi `outputs = model(**model_inputs)` (tương tự cơ chế `Trainer.compute_loss` khi dùng `compute_loss_func`), nhờ đó model chỉ chạy đúng **1 forward pass** để lấy `logits` và **không tạo đồ thị FP32 CE gốc dư thừa**.
  2. Trong `compute_hard_example_weighted_loss`, lặp qua từng mẫu $i$ và chỉ trích xuất các vị trí token hợp lệ `logits[i, :-1, :][valid_mask_i].float()` trước khi gọi `F.cross_entropy(..., reduction="sum")`. Cách này chỉ cấp phát tensor `log_softmax` FP32 cho các token đáp án hợp lệ thay vì toàn bộ chiều dài chuỗi `(B, S, V)`.

### 3.3. Nhánh Train vs Eval/Predict và Gradient Accumulation

- Nhánh tính loss được điều khiển bởi `model.training` (không dựa vào `torch.is_grad_enabled()`):
  - Khi `not model.training` (`trainer.evaluate()`, `trainer.predict()`, `prediction_step`), `CustomSeq2SeqTrainer.compute_loss` luôn gọi `super().compute_loss(...)` với loss chuẩn không trọng số.
- Khi `use_hard_example_loss=True`, `self.model_accepts_loss_kwargs` được đặt `False` trong train (nhất quán với xử lý `if processor is not None:` sẵn có của `CustomSeq2SeqTrainer`) và `self.compute_loss_func` giữ `None`. Nhờ vậy:
  - `compute_loss` trả về loss trung bình theo microbatch.
  - `Trainer.training_step` chia `loss` cho `gradient_accumulation_steps` đúng **một lần**, không bị chia hai lần và không làm hỏng `return_outputs` hay `past_index`.

---

## 4. Kiểm thử CPU đã chạy

Phân biệt rõ hai nhóm kiểm thử trong `tests/test_hard_example_loss.py`:

1. **Pure Helper Unit Tests (`HardExampleLossHelperTests` — 10 test methods)**:
   - Kiểm tra trực tiếp hàm `compute_hard_example_weighted_loss` và `validate_hard_example_loss_config`:
     - Giá trị mặc định (`alpha=0.5`, `max_weight=1.5`, `use_hard_example_loss=False`).
     - Trọng số đồng nhất (`alpha=0` hoặc `max_weight=1`) khớp `ForCausalLMLoss` về cả giá trị scalar và gradient trên các câu trả lời độ dài biến thiên có prompt/padding `-100`.
     - Token prompt, padding, `labels[:, 0]` và `logits[:, -1, :]` có gradient đúng bằng `0`.
     - Kiểm tra đúng chiều dịch nhân quả (`logits[:, t]` dự đoán `labels[:, t+1]`).
     - Trọng số bị chặn trong `[1.0, max_weight]`, mẫu khó hơn nhận trọng số lớn hơn (kể cả khi mẫu khó có ít token hơn mẫu dễ).
     - `difficulty` và `weights` được detach hoàn toàn khỏi autograd (khớp tuyệt đối với gradient khi coi $w_i$ là hằng số cố định, và khác với gradient khi để đạo hàm đi qua $w_i$).
     - Xử lý an toàn microbatch có một số mẫu toàn `-100`, toàn bộ microbatch toàn `-100`, `seq_len=1`, `batch_size=0`, và trường hợp `mean_difficulty == 0.0`.
     - Input `torch.float16` trên CPU được nâng lên FP32 khi tính CE, trả về loss FP32 hữu hạn và gradient FP16 hữu hạn.
     - Không dùng `nan_to_num` che giấu `NaN` thật trong `logits`.
     - Từ chối mọi cấu hình/shape không hợp lệ.
2. **Actual Framework Trainer Integration Tests (`CustomSeq2SeqTrainerHardExampleIntegrationTests` — 8 test methods)**:
   - Khởi tạo `CustomSeq2SeqTrainer` thật với `Seq2SeqTrainingArguments(use_cpu=True)` và `TinyCausalPreTrainedModel` nội bộ (không tải model/dữ liệu từ mạng):
     - Kiểm tra `HfArgumentParser` parse đúng `use_hard_example_loss`, `hard_example_loss_alpha`, `hard_example_loss_max_weight`.
     - Từ chối các cấu hình xung đột (`stage != "sft"`, `use_dft_loss`, `label_smoothing_factor > 0`, `compute_loss_func`, `enable_liger_kernel`, `use_liger_kernel`, thiếu `labels`, thiếu `logits`).
     - Khi `use_hard_example_loss=False`: tái hiện chính xác `Seq2SeqTrainer.compute_loss` trong cả `train()` và `eval()`.
     - Khi `use_hard_example_loss=True`: `model.eval()` (kể cả dưới `torch.enable_grad()`) vẫn dùng loss gốc của superclass; `model.train()` (kể cả dưới `torch.no_grad()`) dùng nhánh hard-example loss và không truyền `labels` vào `model.forward`.
     - Bảo toàn hợp đồng `return_outputs=True` (gồm `outputs["loss"]`, `outputs.loss`, `outputs[0]` cho cả `ModelOutput` và `tuple`), `past_index`, và không sửa đổi dict `inputs` của hàm gọi.
     - Kiểm tra `trainer.training_step(..., num_items_in_batch=...)` với `gradient_accumulation_steps=4` (cả khi có và không có `processor`): chỉ chia cho `gradient_accumulation_steps` đúng 1 lần.
     - Chạy end-to-end `trainer.evaluate()`, `trainer.predict()`, và `trainer.train()` (1 optimizer step trên CPU).

### Lệnh đã chạy và kết quả

```bash
# 1. Chạy riêng bộ test mới cho hard-example loss
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
  /home/nhat/miniconda3/envs/unimumer/bin/python -B -m unittest -v tests/test_hard_example_loss.py
# -> Ran 18 tests in 0.44s: OK

# 2. Chạy toàn bộ test suite CPU của repo
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
  /home/nhat/miniconda3/envs/unimumer/bin/python -B -m unittest discover -s tests -v
# -> Ran 160 tests in 4.19s: OK (142 existing tests + 18 new tests in test_hard_example_loss.py)

# 3. Kiểm tra định dạng diff
git diff --check
# -> Exit code 0 (không có lỗi khoảng trắng/conflict)
```

---

## 5. Giới hạn còn lại và lưu ý bàn giao cho Coordinator

1. **Giới hạn thuật toán (Residual Algorithmic Limitation)**:
   - Trên cấu hình mục tiêu 2xT4 với `per_device_train_batch_size: 2`, độ khó trung bình $\bar{d}$ chỉ được tính cục bộ trên tối đa **2 mẫu mỗi GPU trong một microbatch** (không đồng bộ qua DDP và không chuẩn hóa theo từng loại nhiệm vụ).
   - Trên tập đa nhiệm (HMER, Counting, Tree-CoT, EDL), các nhiệm vụ vốn có cross-entropy tự nhiên cao hơn có thể nhận trọng số lớn hơn khi ghép chung microbatch với nhiệm vụ có CE thấp hơn. Đây là pilot kiểm chứng giả thuyết ở phạm vi hẹp, **không tuyên bố chuẩn hóa độ khó theo nhiệm vụ (task-normalized difficulty)** và **chưa khẳng định cải thiện chất lượng thực tế** trước khi chạy A/B trên cùng manifest.
2. **Việc chưa kiểm chứng (Missing Verification)**:
   - Toàn bộ kiểm thử hiện tại chỉ chạy trên **CPU local** với model nhỏ tổng hợp. Kết quả này **không chứng minh** đã chạy thành công trên Kaggle 2xT4 QLoRA (Qwen2.5-VL-3B-Instruct 4-bit NF4) hay cải thiện `eval_loss`/CER/EM.
3. **Lưu ý cấu hình cho Coordinator khi tích hợp Notebook / Runtime YAML**:
   - Theo phân quyền, worker không sửa `scripts/runtime_yaml.py`, `train/Uni-MuMER-train.yaml` hay `uni-mumer-kaggle-dagshub-banmoi.ipynb`.
   - `scripts/runtime_yaml.py` hiện kiểm tra `strict_keys=True` dựa trên tập khóa của `train/Uni-MuMER-train.yaml` hợp với `EXTRA_OVERRIDE_KEYS`. Trước khi truyền `use_hard_example_loss`, `hard_example_loss_alpha`, `hard_example_loss_max_weight` qua `YAML_OVERRIDES` trong notebook, Coordinator cần thêm 3 khóa này vào `EXTRA_OVERRIDE_KEYS` trong `scripts/runtime_yaml.py` (hoặc khai báo mặc định trong `train/Uni-MuMER-train.yaml`).

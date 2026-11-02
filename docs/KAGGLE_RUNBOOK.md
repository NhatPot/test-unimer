# Hướng dẫn chạy Kaggle có kiểm soát

## 1. Nguyên tắc môi trường

Toàn bộ thư viện phục vụ project phải được cài trong môi trường Python riêng. Không dùng torch, transformers, bitsandbytes, datasets, MLflow hoặc PyYAML cài sẵn trong kernel Kaggle để chạy một phần pipeline.

Kaggle vẫn cung cấp máy, GPU, driver và giao diện notebook. `kaggle_secrets` chỉ là cầu nối lấy token của nền tảng; không phải môi trường ML của project. Token đi qua biến môi trường, không in ra log hoặc lưu trong notebook. Nếu không muốn dùng cầu nối này, phải cấu hình một kênh secrets tương đương, không hard-code token.

Giữ Python 3.10 riêng theo pipeline hiện tại. Mọi helper có dependency, kể cả `prepare_runtime_yaml`, phải chạy qua interpreter này; notebook chỉ truyền cấu hình và nhận JSON/đường dẫn qua subprocess. Không mặc định rằng `!pip`, `%pip` hoặc `!python` trỏ vào môi trường đúng.

## 2. Khóa nguồn code trước khi chạy

- Dùng `uni-mumer-kaggle-dagshub-banmoi.ipynb`; bảo đảm file được Git theo dõi và đúng bản đang mở trên Kaggle.
- Chốt branch/commit được dùng, cùng revision của base model và dataset. Lưu notebook thực tế, không chỉ một file cùng tên trong repo.
- Cell clone hiện tại không cập nhật repo đã tồn tại. Khi chạy lại session, kiểm tra commit; không tự `pull`, reset hoặc đổi code giữa run.
- Đối với repo private, dùng cơ chế xác thực an toàn; không ghi token vào URL clone hay notebook output.
- Không dùng cache không rõ phiên bản hoặc checkpoint chỉ trùng tên nhưng thuộc run khác.

## 3. Tạo và cài môi trường riêng

Các lệnh dưới là hướng dẫn cho Kaggle, không được tự chạy khi chỉ đọc tài liệu hoặc sửa docs. Cài đặt chỉ sau khi được cho phép. Chúng chưa phải bằng chứng requirements hiện tại tương thích.

1. Cài Miniconda riêng tại `/kaggle/working/miniconda`; cố định phiên bản bộ cài và kiểm tra SHA256 từ nguồn chính thức. Không phụ thuộc link `latest` mà không lưu phiên bản.
2. Tạo env `unimumer` với Python 3.10, không kế thừa system site-packages. Ghi lại phiên bản Python cụ thể; không nâng phiên bản trong cùng thí nghiệm.
3. Dùng cùng bootstrap môi trường cho mọi cell shell, vì `export` bên trong một cell Bash không tự tồn tại trong cell tiếp theo.

```bash
set -euo pipefail
export PROJECT_DIR=/kaggle/working/test-unimer
export ENV_DIR=/kaggle/working/miniconda/envs/unimumer
export PYTHON="$ENV_DIR/bin/python"
export PATH="$ENV_DIR/bin:$PATH"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
export PYTHONPATH="$PROJECT_DIR"
cd "$PROJECT_DIR"
test -x "$PYTHON"
"$PYTHON" --version
"$PYTHON" -m pip --version
```

`PYTHONPATH` ở đây chỉ chứa source repo, không chứa site-packages của Kaggle. Framework cài editable có thể trỏ về source `train/LLaMA-Factory`, đó là ngoại lệ có chủ đích.

4. Cài dependency và framework trong cùng env. Bước khởi đầu để kiểm tra bộ requirements hiện tại là:

```bash
"$PYTHON" -m pip install -r "$PROJECT_DIR/requirements.txt"
"$PYTHON" -m pip install -e "$PROJECT_DIR/train/LLaMA-Factory"
"$PYTHON" -m pip check
```

Phải có bootstrap ở trên trong cùng shell. Nếu resolver báo xung đột hoặc `pip check` thất bại, dừng và xử lý rồi cài lại trong env sạch. Không che lỗi bằng `|| true`, tự nâng torch, hay `--no-deps` không có kiểm chứng. Notebook cũ đã ghi cảnh báo xung đột `prometheus-fastapi-instrumentator`/`starlette`; không coi cảnh báo đó là vô hại chỉ vì import torch thành công.

5. Sau khi tìm được bộ phiên bản chạy tốt, khóa dependency cùng thông tin CUDA/wheel index và commit framework. Kiểm chứng cài lại từ đầu; `pip freeze` là bản ghi môi trường, chưa đủ để chứng minh tái lập được.
6. Có thể tách vLLM thành môi trường inference riêng nếu gây xung đột, vì đường test adapter hiện mặc định dùng Transformers + PEFT. Đây là thay đổi cần kiểm thử, không phải tính năng đã có. Không chuyển sang vLLM/kernel Kaggle để lách lỗi cài đặt.

## 4. Kiểm tra môi trường trước GPU job

Sau khi cài xong, dùng đúng bootstrap để kiểm tra nhẹ:

```bash
"$PYTHON" -m pip check
command -v llamafactory-cli
command -v torchrun
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$PYTHON" - <<'PY'
import importlib
import site
import sys
from pathlib import Path

expected = Path('/kaggle/working/miniconda/envs/unimumer').resolve()
assert Path(sys.prefix).resolve() == expected, sys.prefix
assert not site.ENABLE_USER_SITE, 'User site-packages must be disabled'
print('Python:', sys.executable)
print('Prefix:', sys.prefix)
for name in ('torch', 'torchvision', 'transformers', 'accelerate', 'peft',
             'bitsandbytes', 'datasets', 'mlflow', 'yaml'):
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    assert path.is_relative_to(expected), (name, str(path))
    print(name, getattr(module, '__version__', 'see pip freeze'), str(path))

import torch
assert torch.cuda.is_available(), 'CUDA unavailable'
print('Torch CUDA:', torch.version.cuda)
print('Visible GPUs:', torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    device = torch.cuda.get_device_properties(index)
    print(index, device.name, round(device.total_memory / 2**30, 2), 'GiB',
          torch.cuda.get_device_capability(index))
PY
```

Chỉ kiểm tra import/GPU, chưa nạp model. Xác nhận đường dẫn CLI thuộc `$ENV_DIR/bin`; nếu khác thì dừng. Không in toàn bộ environment vì có secrets. `pip check` kiểm tra metadata dependency, không thay thế kiểm tra kernel CUDA/quantization khi smoke test. Xem [pip check](https://pip.pypa.io/en/stable/cli/pip_check/) và [PYTHONNOUSERSITE](https://docs.python.org/3.10/using/cmdline.html#envvar-PYTHONNOUSERSITE).

## 5. Profile phần cứng và cấu hình

| Hạng mục | Quy tắc |
| --- | --- |
| GPU mục tiêu | 2 x NVIDIA T4, khoảng 16 GB/GPU; đối chiếu log thật mỗi session |
| Precision T4 | `bf16: false`, `fp16: true`; dtype tính toán inference là FP16 |
| Model | `Qwen/Qwen2.5-VL-3B-Instruct`, ghi revision |
| QLoRA | NF4 4-bit, rank 64, alpha 16, dropout 0.05 |
| Giới hạn đầu vào nền | `image_max_pixels: 262144`, `cutoff_len: 2048`; ghi lại mọi thay đổi |
| Batch nền hiện tại | 2/GPU x accumulation 64 x world size thực tế |
| Nếu đủ 2 rank | Effective batch là 256; nếu chỉ 1 rank thì là 128 |
| Resume | `save_only_model: false`; checkpoint phải đủ trạng thái phù hợp |
| Chọn best hiện tại | `eval_loss`, không phải EM; chưa có chọn best theo generation EM |

T4 dùng FP16 theo [hướng dẫn mixed precision của Hugging Face](https://huggingface.co/docs/transformers/mixed_precision_training). GPU khác cần profile đã kiểm tra, không âm thầm dùng lại thông số. Hai GPU DDP giữ bản model riêng; không xem VRAM như một vùng nhớ gộp 32 GB. Nếu chỉ có một GPU hoặc dung lượng khác, dừng đối chiếu cấu hình trước khi tiếp tục.

Batch 2 là cấu hình nền, không phải bảo đảm không OOM. Khi thử kỹ thuật, có thể bắt đầu batch/GPU 1; nếu muốn giữ effective batch 256 trên 2 GPU thì accumulation cần 128. Không thay learning rate, độ phân giải, chiều dài chuỗi hoặc batch trong thí nghiệm đối chứng mà không ghi nhận.

## 6. Cấu hình cuối cùng và đường dẫn

- `train/Uni-MuMER-train.yaml` là nền; notebook hiện ghi đè `max_samples: 8000`, `num_train_epochs: 2`. Dùng YAML đã resolve, không suy luận từ YAML nền.
- `None` trong helper hiện có nghĩa là giữ giá trị nền, không phải tự chọn giá trị an toàn. Muốn xóa một giá trị như resume path cần cơ chế rõ ràng, không giả định `None` sẽ xóa.
- `strict_keys=True` không cho thêm khóa chưa có trong YAML nền. Khi thêm `max_steps`, `eval_dataset`, seed hay tham số sampler, phải sửa hợp đồng override có kiểm thử, không bỏ strict mode cho qua lỗi.
- Với 6 nguồn, 8.000 là giới hạn từng nguồn, tối đa 48.000 bản ghi trước các bước xử lý, không phải 48.000 ảnh độc lập. Đếm thực tế và ghi distribution; loader hiện lấy các hàng đầu, không phải lấy ngẫu nhiên theo cấu trúc.
- Dùng registry `train/dataset_info.json`. Tách validation theo ảnh; nếu dùng `eval_dataset` thì đặt `val_size: 0` theo hợp đồng loader.
- Output, runtime YAML, sync state và kết quả phải tách theo run. Quy ước đề xuất, chưa triển khai: `/kaggle/working/runs/<RUN_UUID>/{runtime.yaml,checkpoints,checkpoint_sync_state.json,results}`.
- Luôn log commit, notebook hash, resolved YAML, model/dataset revision, split hash, seed, world size, GPU và phiên bản dependency trước khi train; không lưu token.

## 7. Checkpoint và DagsHub

1. Phân biệt train mới, full resume giữ trạng thái, và warm start từ adapter mở thí nghiệm mới. Không tự đổi chế độ nếu resume thất bại.
2. Ghi đúng cặp base model/revision và adapter. Với LoRA cần ít nhất `adapter_config.json` và weights; full resume còn cần `trainer_state.json`, optimizer, scheduler, RNG của các rank và scaler khi áp dụng. Phân tích JSON, kiểm tra file không rỗng và thử load; không chỉ nhìn tên file.
3. `RUN_UUID` tự sinh là tag truy vết. Tra `run.info.run_id` thật trước khi upload, không truyền UUID đó trực tiếp làm run ID. Xem [MLflow start_run](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html#mlflow.start_run).
4. Không dùng một file sync state cho nhiều run. Ghi run ID, step, danh sách file, kích thước/hash và trạng thái upload hoàn tất. Không đánh dấu thành công khi upload dở.
5. Với `save_total_limit: 2`, Trainer có thể dọn checkpoint trong khi watcher đang upload. Thiết kế cơ chế giữ bản đang upload hoặc retention an toàn; không xóa bản local duy nhất trước khi xác minh remote.
6. Không để background sync và final sync ghi state/upload đồng thời. Khi kết thúc hoặc lỗi, kiểm tra checkpoint mới nhất và báo trạng thái remote thật.
7. Kiểm chứng tải lại checkpoint vào thư mục mới, so sánh manifest/hash và resume ngắn. Cache chỉ trùng tên `checkpoint-700` chưa chứng minh đúng run nguồn.

## 8. Smoke test có giới hạn

Chỉ chạy sau khi người dùng cho phép GPU test và các kiểm tra trên đã qua. Profile smoke là cấu hình thử riêng, không thay YAML baseline và không trộn kết quả với benchmark.

- Dùng tập nhỏ cố định, đủ cho batch và validation; đặt `max_steps: 5` hoặc `10`, batch/GPU 1, accumulation nhỏ để giới hạn chi phí.
- Trong profile smoke, đặt save/eval cùng chu kỳ, ví dụ mỗi 5 bước, để thật sự tạo checkpoint. Không giữ chu kỳ 50 khi chỉ chạy 5 bước.
- Kiểm tra loss hữu hạn, không NaN/OOM, đúng world size, đúng Python và dtype.
- Xác minh upload, tải lại, rồi resume thêm 1-2 optimizer step với giới hạn tổng step phù hợp. Kiểm tra step/scheduler không bị reset ngoài ý muốn.
- Inference 1-3 ảnh bằng base + adapter, batch 1, decode cố định; không bắt buộc merge model để smoke test.
- Lưu log và thời gian; có smoke pass mới cân nhắc job dài. Smoke pass không chứng minh hội tụ hay đạt EM mục tiêu.

## 9. Đánh giá và điều kiện dừng

- Full CROHME cần đủ 986/1.147/1.199 mẫu tương ứng 2014/2016/2019. Đối chiếu ID, ảnh, nhãn, dự đoán và số mẫu chấm thực tế; không chỉ in số cố định trong bảng.
- Thiếu ảnh, ID trùng/thừa, JSON không hợp lệ hoặc inference lỗi hạ tầng: dừng full benchmark. Không bỏ mẫu hoặc dùng kết quả cũ để lấp chỗ trống.
- Nếu model chạy thành công nhưng trả chuỗi rỗng, giữ mẫu và tính là nhận dạng sai. Tách trạng thái này khỏi CUDA/file error.
- Chế độ subset/smoke phải ghi rõ số mẫu và không được gắn nhãn full benchmark.
- Cố định prompt, preprocessing, generation và chuẩn hóa LaTeX; lưu dự đoán thô. Không dùng stochastic decoding không kiểm soát để kết luận một phương pháp tốt hơn.
- Chọn checkpoint trên validation, không dùng CROHME test để cập nhật sampler hoặc chọn best. Token accuracy khi teacher forcing không phải expression EM.
- Kết quả có sẵn trong `example_data` là tham khảo chưa xác minh provenance cho baseline của run mới. Không gắn nhãn "base model" nếu chưa xác nhận model/config tạo ra chúng.
- Nếu gộp ba bộ test, phân biệt macro average với EM gộp theo tổng số mẫu; không gọi trung bình ba tỷ lệ là EM gộp.

## 10. Điều kiện được báo sẵn sàng

- [ ] Cài lại thành công trong Kaggle session sạch bằng dependency của project.
- [ ] `pip check`, import provenance và kiểm tra GPU/CLI đều đạt.
- [ ] Runtime YAML đúng profile thực tế và được lưu cùng run.
- [ ] Smoke train, upload/download/resume và inference nhỏ đều có log đạt.
- [ ] Split, coverage, checkpoint và kết quả đều truy vết được.
- [ ] Notebook được chạy tuần tự không cần sửa tay môi trường ngoài các cell đã ghi nhận.

Nếu chưa chạy được trên Kaggle thật, ghi đúng: "đã kiểm tra tĩnh/local; chưa xác minh Kaggle". Không thể bảo đảm môi trường chạy được chỉ bằng việc viết tài liệu.

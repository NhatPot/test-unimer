# Kaggle Testing Scripts

Scripts để test model trên Kaggle sau khi training, theo đúng logic của tác giả Uni-MuMER.

## 📁 Files

- **`kaggle_test_adapter.py`**: Test model trên 1 dataset
- **`kaggle_full_test.py`**: Test model trên cả 3 CROHME datasets và tạo bảng so sánh

## 🎯 Cách sử dụng

### 1. Test một dataset

```bash
python scripts/kaggle_test_adapter.py \
  --base-model Qwen/Qwen2.5-VL-3B-Instruct \
  --adapter-path saves/qwen2.5_vl-3b/qlora/sft/standred/uni-mumer_qlora/checkpoint-20 \
  --test-data example_data/backup/crohme_2014.json \
  --output-dir test_results \
  --batch-size 4
```

### 2. Test full benchmark (3 CROHME datasets)

```bash
python scripts/kaggle_full_test.py \
  --base-model Qwen/Qwen2.5-VL-3B-Instruct \
  --adapter-path saves/qwen2.5_vl-3b/qlora/sft/standred/uni-mumer_qlora/checkpoint-20 \
  --test-datasets crohme_2014 crohme_2016 crohme_2019 \
  --backup-dir example_data/backup \
  --base-results-dir example_data/CROHME/results \
  --output-dir kaggle_test_results \
  --batch-size 2
```

## 📊 Output

### kaggle_test_adapter.py outputs:

```
output_dir/
├── crohme_2014_prompts.json   # Converted prompts format
├── crohme_2014_pred.json      # Predictions
└── crohme_2014_results.txt    # Metrics
```

### kaggle_full_test.py outputs:

```
kaggle_test_results/
├── crohme_2014_prompts.json
├── crohme_2014_pred.json
├── crohme_2014_results.txt
├── crohme_2016_prompts.json
├── crohme_2016_pred.json
├── crohme_2016_results.txt
├── crohme_2019_prompts.json
├── crohme_2019_pred.json
├── crohme_2019_results.txt
└── comparison_table.txt        # Bảng so sánh base vs fine-tuned vs paper
```

## 🔧 Công nghệ

### Inference Engine

**Default: Transformers + PEFT**
- Load base model với 4-bit quantization
- Load LoRA adapter từ checkpoint
- Inference từng batch
- VRAM usage: ~4-5GB trên 1 GPU

**Optional: vLLM** (chưa hỗ trợ đầy đủ)
- Cần cài `pip install vllm`
- Nhanh hơn nhưng có thể không tương thích với Kaggle environment

### Evaluation

Sử dụng `scripts/eval_metrics_calculator.py` có sẵn:
- Mean Edit Score
- BLEU-4 Score
- Character Error Rate (CER)
- Exact Match Rate
- Error Threshold Analysis (≤1, ≤2, ≤3)

## ⏱️ Thời gian ước tính

| Dataset | Samples | Inference | Total |
|---------|---------|-----------|-------|
| CROHME 2014 | 986 | ~5-7 phút | ~6-8 phút |
| CROHME 2016 | 1,147 | ~6-8 phút | ~7-9 phút |
| CROHME 2019 | 1,199 | ~6-8 phút | ~7-9 phút |
| **Tổng (3 datasets)** | **3,332** | **~17-23 phút** | **~20-26 phút** |

## 📝 Lưu ý

### Input format

Scripts tự động convert từ backup format sang prompts format:

**Backup format** (example_data/backup/):
```json
[
  {
    "image": "relative/path.png",
    "latex": "x^2 + y^2"
  }
]
```

**Prompts format** (tác giả):
```json
[
  {
    "images": ["absolute/path.png"],
    "messages": [
      {"from": "human", "value": "<image>Prompt..."},
      {"from": "gpt", "value": "x^2 + y^2"}
    ]
  }
]
```

### Batch size

- Default: 4 cho test đơn, 2 cho full test
- Giảm nếu gặp OOM error
- Tăng nếu VRAM còn dư (tối đa 8-16)

### Adapter vs Merged model

Scripts load **adapter trực tiếp** (không cần merge):
- ✅ Tiết kiệm thời gian (không cần merge 5-10 phút)
- ✅ Tiết kiệm disk (~6-12GB)
- ⚠️ Inference chậm hơn ~10-20% so với merged model
- ✅ Đủ nhanh cho testing trên Kaggle

## 🎯 So sánh với tác giả

| Aspect | Tác giả | Kaggle Scripts |
|--------|---------|----------------|
| Input format | Prompts JSON | Backup JSON → Auto-convert |
| Model | Merged model | Base + Adapter (no merge) |
| Inference | vLLM | Transformers (default) |
| Evaluation | eval_metrics_calculator.py | **Giống hệt** ✅ |
| Output | Predictions + Metrics | **Giống hệt** ✅ |

## 🐛 Troubleshooting

### OOM Error
```bash
# Giảm batch size
--batch-size 1
```

### Image not found
```bash
# Check project_dir
--project-dir /kaggle/working/test-unimer
```

### Slow inference
```bash
# Tăng batch size nếu VRAM cho phép
--batch-size 8
```

## 📚 Tham khảo

- Original paper: [arXiv:2505.23566](https://arxiv.org/abs/2505.23566)
- Original repo: https://github.com/BFlameSwift/Uni-MuMER
- Tác giả test script: `scripts/vllm_infer.py`
- Eval script: `scripts/eval_metrics_calculator.py`

# Ablation Study Guide for Uni-MuMER

This guide explains how to run fair ablation experiments to measure the contribution of each component (Tree-CoT, EDL, Symbol Counting).

## 📋 Overview

We created 5 fixed ablation datasets with controlled sampling ratios:

| Config | Components | Composition |
|--------|-----------|-------------|
| **Baseline** | Img→LaTeX only | 8000 crohme_train |
| **Tree-CoT** | Baseline + Tree-CoT | 4000 crohme_train + 4000 tree |
| **EDL** | Baseline + Error Detection | 4000 crohme_train + 2000 error_find + 2000 error_fix |
| **Counting** | Baseline + Symbol Counting | 4000 crohme_train + 4000 can |
| **Full** | All components | 4000 crohme_train + 1000 tree + 1000 can + 1000 error_find + 1000 error_fix |

**All configs:**
- ✅ Same total samples: 8000
- ✅ Same base model: Qwen/Qwen2.5-VL-3B-Instruct
- ✅ Same LoRA config: rank=64, alpha=16, dropout=0.05
- ✅ Same hyperparameters: lr=1e-4, batch_size=2, grad_accum=64
- ✅ Same seed: 42
- ✅ Same test set: CROHME 2014, 2016, 2019

**Only difference:** Training data composition

---

## 🛠️ Step 1: Create Ablation Datasets (One-time setup)

Run this script **once** to create fixed ablation datasets:

```bash
python scripts/create_ablation_datasets.py --seed 42 --output-dir data/ablation
```

**Output:**
```
data/ablation/
├── ablation_baseline_8000.parquet
├── ablation_tree_8000.parquet
├── ablation_edl_8000.parquet
├── ablation_counting_8000.parquet
└── ablation_full_8000.parquet
```

**Note:** These datasets are already registered in `train/LLaMA-Factory/data/dataset_info.json`

---

## 🚀 Step 2: Train Each Configuration

### Option A: Local Training

```bash
# Train each config sequentially
llamafactory-cli train train/ablation/ablation_baseline_8000.yaml
llamafactory-cli train train/ablation/ablation_tree_8000.yaml
llamafactory-cli train train/ablation/ablation_edl_8000.yaml
llamafactory-cli train train/ablation/ablation_counting_8000.yaml
llamafactory-cli train train/ablation/ablation_full_8000.yaml
```

### Option B: Kaggle Notebook (Quick Test)

1. Open `uni-mumer-kaggle-dagshub v7.ipynb`
2. In **Cell 1 (Config)**, set:
   ```python
   BASE_YAML_CONFIG = "train/ablation/ablation_tree_8000.yaml"
   ```
3. In **Cell 5 (Runtime YAML Override)**:
   ```python
   USE_RUNTIME_YAML_OVERRIDE = True
   
   YAML_OVERRIDES = {
       # DO NOT override dataset or max_samples!
       # Only override hyperparameters for quick testing:
       "num_train_epochs": 1,  # Quick test with 1 epoch
       "per_device_train_batch_size": 2,
       "gradient_accumulation_steps": 64,
       "learning_rate": 1.0e-4,
       "logging_steps": 1,
       "save_steps": 20,
   }
   ```
4. Run notebook → trains with ablation dataset

**Repeat for each config** by changing `BASE_YAML_CONFIG`.

---

## 📊 Step 3: Test All Configs on Same Test Set

After training all 5 configs, test them on **same CROHME test sets**:

```bash
# Test each checkpoint
for config in baseline tree edl counting full; do
  python scripts/kaggle_full_test.py \
    --base-model Qwen/Qwen2.5-VL-3B-Instruct \
    --adapter-path saves/ablation/${config}_8000/checkpoint-XXX \
    --test-datasets crohme_2014 crohme_2016 crohme_2019 \
    --backup-dir example_data/backup \
    --images-base-dir data/CROHME \
    --output-dir results/ablation/${config}_8000 \
    --project-dir . \
    --batch-size 2
done
```

---

## 📈 Step 4: Compare Results

Results will be saved in:
```
results/ablation/
├── baseline_8000/
│   ├── crohme_2014_results.txt
│   ├── crohme_2016_results.txt
│   ├── crohme_2019_results.txt
│   └── comparison_table.txt
├── tree_8000/
├── edl_8000/
├── counting_8000/
└── full_8000/
```

**Compare metrics:**
- Mean Edit Score (higher is better)
- BLEU-4 Score (higher is better)
- Character Error Rate (lower is better)
- Exact Match Rate (higher is better)

---

## ⚠️ Important Notes

### ✅ DO:
- Use ablation YAML configs as-is for fair comparison
- Train all configs with same seed (42)
- Test all configs on same CROHME test sets
- Compare results after all configs are trained

### ❌ DON'T:
- Override `dataset` in `YAML_OVERRIDES` (breaks ablation)
- Override `max_samples` in `YAML_OVERRIDES` (dataset already sampled)
- Change LoRA config between runs (breaks fairness)
- Change learning rate between runs (breaks fairness)
- Test on different test sets (breaks fairness)

---

## 🔬 Expected Outcomes

If components are effective, we expect:

1. **Baseline < Tree-CoT** → Tree-CoT helps
2. **Baseline < EDL** → Error detection helps
3. **Baseline < Counting** → Symbol counting helps
4. **Full > Individual components** → Combined effect

---

## 📝 Reproducibility

All ablation datasets are created with:
- **Fixed seed:** 42
- **Fixed ratios:** Exact sample counts from each source dataset
- **Deterministic sampling:** Same dataset every time

To reproduce:
1. Run `create_ablation_datasets.py` with same seed
2. Train with same YAML configs
3. Test on same test sets
4. Compare results

---

## 🆘 Troubleshooting

**Problem:** "Dataset ablation_xxx_8000 not found"
- **Solution:** Run `create_ablation_datasets.py` first

**Problem:** "Different results on different runs"
- **Solution:** Ensure same seed (42) in all YAML configs

**Problem:** "Unfair comparison"
- **Solution:** Don't override `dataset` or `max_samples` in notebook

**Problem:** "Can't find checkpoint-XXX"
- **Solution:** Check `saves/ablation/{config}_8000/` for actual checkpoint numbers

---

## 📚 Citation

If you use this ablation framework, please cite:

```bibtex
@inproceedings{unimumer2024,
  title={Uni-MuMER: Unified Multi-Modal Error Recognition},
  author={...},
  booktitle={...},
  year={2024}
}
```

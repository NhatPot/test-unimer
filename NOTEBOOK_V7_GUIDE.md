# Notebook v7 - Ablation Study Usage Guide

## 🎯 Purpose

This notebook is designed to train and test ablation study configurations on Kaggle with 2xT4 GPUs.

## 📝 How to Use

### **For Ablation Study:**

1. **Cell 1 (Config)** - Choose ablation config:
   ```python
   # Select one of the ablation configs:
   BASE_YAML_CONFIG = "train/ablation/ablation_baseline_8000.yaml"
   # Options:
   # - train/ablation/ablation_baseline_8000.yaml
   # - train/ablation/ablation_tree_8000.yaml
   # - train/ablation/ablation_edl_8000.yaml
   # - train/ablation/ablation_counting_8000.yaml
   # - train/ablation/ablation_full_8000.yaml
   ```

2. **Cell 5 (Runtime YAML Override)** - IMPORTANT:
   ```python
   USE_RUNTIME_YAML_OVERRIDE = True
   
   YAML_OVERRIDES = {
       # ❌ DO NOT override these when using ablation configs:
       # "dataset": "...",  # Dataset is fixed in ablation YAML
       # "max_samples": ...,  # Already sampled to 8000
       
       # ✅ Only override hyperparameters for quick testing:
       "num_train_epochs": 1,  # Test with 1 epoch (original: 3)
       "per_device_train_batch_size": 2,
       "gradient_accumulation_steps": 64,
       "learning_rate": 1.0e-4,
       "logging_steps": 1,
       "save_steps": 20,
   }
   ```

3. **Run all cells** - The notebook will:
   - Setup environment
   - Clone repo
   - Install dependencies
   - Download test images
   - Train with ablation config
   - Test on CROHME 2014, 2016, 2019
   - Upload results to DagsHub

4. **Repeat for each ablation config** - Change `BASE_YAML_CONFIG` and rerun

---

### **For Normal Training (Non-Ablation):**

1. **Cell 1 (Config)**:
   ```python
   BASE_YAML_CONFIG = "train/Uni-MuMER-train.yaml"
   ```

2. **Cell 5 (Runtime YAML Override)**:
   ```python
   USE_RUNTIME_YAML_OVERRIDE = True
   
   YAML_OVERRIDES = {
       # ✅ Can override dataset for normal training
       "dataset": "parquet_crohme_train",
       "max_samples": 100,  # Quick test
       "num_train_epochs": 1,
       ...
   }
   ```

---

## ⚠️ Important Rules for Ablation

### ✅ DO:
- Use `train/ablation/*.yaml` as `BASE_YAML_CONFIG`
- Only override hyperparameters in `YAML_OVERRIDES`
- Keep same test set (CROHME 2014, 2016, 2019)
- Train all configs before comparing results

### ❌ DON'T:
- Override `dataset` when using ablation configs
- Override `max_samples` when using ablation configs
- Change LoRA config between runs
- Mix ablation and non-ablation configs

---

## 📊 Expected Results

After running all 5 ablation configs, you'll have:

```
saves/ablation/
├── baseline_8000/checkpoint-XXX/
├── tree_8000/checkpoint-XXX/
├── edl_8000/checkpoint-XXX/
├── counting_8000/checkpoint-XXX/
└── full_8000/checkpoint-XXX/
```

Test results on DagsHub:
- Each run will have test metrics for CROHME 2014, 2016, 2019
- Compare across configs to measure component contributions

---

## 🔧 Customization

### Quick Test (1 epoch):
```python
YAML_OVERRIDES = {
    "num_train_epochs": 1,
    "save_steps": 20,
}
```

### Full Training (3 epochs):
```python
YAML_OVERRIDES = {
    "num_train_epochs": 3,
    "save_steps": 100,
}
```

### Different Learning Rate:
```python
YAML_OVERRIDES = {
    "learning_rate": 5.0e-5,  # Lower LR
}
```

---

## 📈 Monitoring

- **Tensorboard:** Logs saved to `saves/ablation/{config}/runs/`
- **MLflow:** Tracked on DagsHub
- **Test Results:** Saved to `kaggle_test_results/`

---

## 🆘 Troubleshooting

**Error: "Dataset ablation_xxx not found"**
- Make sure ablation datasets exist in `data/ablation/`
- Run `scripts/create_ablation_datasets.py` first (on local machine)

**Error: "Runtime YAML override failed"**
- Check `scripts/runtime_yaml.py` for compatibility
- Ensure you're not overriding protected keys (`dataset`, `max_samples`)

**Error: "Images not found"**
- Cell 3.5 should download CROHME test images
- Check `data/CROHME/2014/images/` exists

**Error: "Model class mismatch"**
- Fixed in v6: Uses `Qwen2_5_VLForConditionalGeneration`
- Make sure latest code is pulled from GitHub

---

## 📚 Related Files

- **Ablation configs:** `train/ablation/*.yaml`
- **Ablation guide:** `ABLATION_GUIDE.md`
- **Dataset creation:** `scripts/create_ablation_datasets.py`
- **Test script:** `scripts/kaggle_full_test.py`

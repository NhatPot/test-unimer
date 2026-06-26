# Instructions: Add New Cells to uni-mumer-kaggle-dagshub v8.ipynb

## Cell 4.5: Create Ablation Datasets (thêm sau Cell #4)

**CHÚ Ý:** Cell này chỉ tạo dataset tương ứng với `BASE_YAML_CONFIG` đang chọn trong Cell #1.

```bash
%%bash
# 4.5. Create ablation datasets with validation
set -euo pipefail
cd "$PROJECT_DIR"
source "$CONDA_DIR/bin/activate" unimumer

# Option: force rebuild (set FORCE_REBUILD=true to rebuild)
FORCE_REBUILD=${FORCE_REBUILD:-false}

# Parse ablation config from BASE_YAML_CONFIG
ABLATION_CONFIG=""

if [[ "$BASE_YAML_CONFIG" == *"ablation_baseline_8000.yaml"* ]]; then
    ABLATION_CONFIG="ablation_baseline_8000"
elif [[ "$BASE_YAML_CONFIG" == *"ablation_tree_8000.yaml"* ]]; then
    ABLATION_CONFIG="ablation_tree_8000"
elif [[ "$BASE_YAML_CONFIG" == *"ablation_edl_8000.yaml"* ]]; then
    ABLATION_CONFIG="ablation_edl_8000"
elif [[ "$BASE_YAML_CONFIG" == *"ablation_counting_8000.yaml"* ]]; then
    ABLATION_CONFIG="ablation_counting_8000"
elif [[ "$BASE_YAML_CONFIG" == *"ablation_full_8000.yaml"* ]]; then
    ABLATION_CONFIG="ablation_full_8000"
else
    echo "⚠️  Not an ablation config, skip dataset creation"
    exit 0
fi

echo "📋 Detected ablation config: $ABLATION_CONFIG"
echo ""

# Required file for this config only
REQUIRED_FILE="train/ablation_data/${ABLATION_CONFIG}.parquet"
MANIFEST_FILE="train/ablation_data/ablation_manifest.json"

# Function: check if this config's dataset is valid
check_dataset_valid() {
    echo "Checking dataset: $ABLATION_CONFIG"
    
    # 1. Check file exists
    if [[ ! -f "$REQUIRED_FILE" ]]; then
        echo "✗ Missing: $REQUIRED_FILE"
        return 1
    fi
    echo "✓ File exists: $REQUIRED_FILE"
    
    # 2. Validate with Python
    python << EOF
import pandas as pd
from pathlib import Path

parquet_file = "$REQUIRED_FILE"

try:
    df = pd.read_parquet(parquet_file)
    
    # Check sample count
    if len(df) != 8000:
        print(f"✗ Expected 8000 samples, got {len(df)}")
        exit(1)
    
    # Check schema
    if "conversations" not in df.columns or "image" not in df.columns:
        print(f"✗ Missing required columns")
        exit(1)
    
    print(f"✓ {len(df)} samples, schema OK")
    exit(0)
    
except Exception as e:
    print(f"✗ Validation error: {e}")
    exit(1)
EOF
    
    return $?
}

# Check if need rebuild
NEED_REBUILD=false

if [[ "$FORCE_REBUILD" == "true" ]]; then
    echo "🔄 Force rebuild requested"
    NEED_REBUILD=true
elif ! check_dataset_valid; then
    echo "🔄 Dataset invalid or missing, need rebuild"
    NEED_REBUILD=true
else
    echo "✅ Dataset valid, skip rebuild"
fi

# Build if needed
if [[ "$NEED_REBUILD" == "true" ]]; then
    echo ""
    echo "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="
    echo "Creating ablation dataset: $ABLATION_CONFIG"
    echo "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="
    
    python scripts/create_ablation_datasets.py \
        --seed 42 \
        --output-dir train/ablation_data \
        --project-dir . \
        --dataset-info train/dataset_info.json \
        --config "$ABLATION_CONFIG"
    
    echo ""
    echo "Validating build..."
    if check_dataset_valid; then
        echo "✅ Build successful"
    else
        echo "❌ Build validation failed"
        exit 1
    fi
fi

echo ""
echo "📊 Dataset ready:"
ls -lh "$REQUIRED_FILE"
```

---

## Cell 5.5: Verify dataset_info.json (thêm sau Cell #5, trước Cell #6)

**CHÚ Ý:** Cell này chỉ verify dataset tương ứng với `BASE_YAML_CONFIG` đang chọn.

```python
# 5.5. Verify dataset_info.json has required ablation entry
import json
from pathlib import Path

print("=" * 80)
print("Verifying dataset registration for current config")
print("=" * 80)

# Parse config name from BASE_YAML_CONFIG
required_key = None

if "ablation_baseline_8000" in BASE_YAML_CONFIG:
    required_key = "ablation_baseline_8000"
elif "ablation_tree_8000" in BASE_YAML_CONFIG:
    required_key = "ablation_tree_8000"
elif "ablation_edl_8000" in BASE_YAML_CONFIG:
    required_key = "ablation_edl_8000"
elif "ablation_counting_8000" in BASE_YAML_CONFIG:
    required_key = "ablation_counting_8000"
elif "ablation_full_8000" in BASE_YAML_CONFIG:
    required_key = "ablation_full_8000"
else:
    print("⚠️  Not ablation mode, skip verification")
    print("=" * 80)
    # Exit successfully (not an error)
    import sys
    sys.exit(0)

print(f"📋 Current config: {required_key}")
print("")

# Load dataset_info.json
dataset_info_path = Path(PROJECT_DIR) / "train/dataset_info.json"

with open(dataset_info_path, 'r', encoding='utf-8') as f:
    dataset_info = json.load(f)

# Check if required key exists
if required_key in dataset_info:
    config = dataset_info[required_key]
    print(f"✓ {required_key} registered in dataset_info.json")
    print(f"    file_name: {config.get('file_name')}")
    print(f"    formatting: {config.get('formatting')}")
    print(f"    columns: {config.get('columns', {})}")
    print("")
    print("=" * 80)
    print("✅ Dataset registration verified - Ready to train")
    print("=" * 80)
else:
    print(f"✗ {required_key} NOT FOUND in dataset_info.json")
    print("")
    print("=" * 80)
    print("❌ Dataset not registered")
    print("=" * 80)
    raise RuntimeError(f"Missing dataset registration: {required_key}")
```

---

## Instructions:

1. Open `uni-mumer-kaggle-dagshub v8.ipynb` in Jupyter/Kaggle
2. **Insert Cell 4.5** after Cell #4 (Cài dependency)
   - Type: Code
   - Content: Copy bash script above
3. **Insert Cell 5.5** after Cell #5 (Runtime YAML Override), before Cell #6
   - Type: Code  
   - Content: Copy Python script above
4. Save notebook
5. Test by running Cell 4.5 → should create 5 parquet files
6. Test by running Cell 5.5 → should verify all 5 entries in dataset_info.json

---

## Force Rebuild (if needed):

To force rebuild all datasets, add this cell BEFORE Cell 4.5:

```python
import os
os.environ["FORCE_REBUILD"] = "true"
```

Then run Cell 4.5.

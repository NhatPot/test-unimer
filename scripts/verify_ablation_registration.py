#!/usr/bin/env python3
"""
Verify that the current ablation config is registered in dataset_info.json.

Automatically detects which ablation config is active from BASE_YAML_CONFIG
environment variable and checks if it's registered.

Usage:
    python scripts/verify_ablation_registration.py
"""

import json
import os
import sys
from pathlib import Path


def parse_ablation_config(base_yaml_config: str) -> str:
    """
    Parse ablation config name from BASE_YAML_CONFIG.

    Returns:
        Config name (e.g., "ablation_baseline_8000") or None if not ablation
    """
    config_mapping = {
        "ablation_baseline_8000.yaml": "ablation_baseline_8000",
        "ablation_tree_8000.yaml": "ablation_tree_8000",
        "ablation_edl_8000.yaml": "ablation_edl_8000",
        "ablation_counting_8000.yaml": "ablation_counting_8000",
        "ablation_full_8000.yaml": "ablation_full_8000",
    }

    for yaml_name, config_name in config_mapping.items():
        if yaml_name in base_yaml_config:
            return config_name

    return None


def main():
    print("=" * 80)
    print("Verifying dataset registration for current config")
    print("=" * 80)

    # Get BASE_YAML_CONFIG from environment
    base_yaml_config = os.environ.get("BASE_YAML_CONFIG", "")

    if not base_yaml_config:
        print("ERROR: BASE_YAML_CONFIG environment variable not set")
        sys.exit(1)

    # Parse config name
    required_key = parse_ablation_config(base_yaml_config)

    if not required_key:
        print("⚠️  Not ablation mode, skip verification")
        print("=" * 80)
        sys.exit(0)

    print(f"📋 Current config: {required_key}")
    print("")

    # Get project dir
    project_dir = os.environ.get("PROJECT_DIR", ".")
    dataset_info_path = Path(project_dir) / "train/dataset_info.json"

    if not dataset_info_path.exists():
        print(f"❌ dataset_info.json not found: {dataset_info_path}")
        sys.exit(1)

    # Load dataset_info.json
    with open(dataset_info_path, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)

    # Check if required key exists
    if required_key in dataset_info:
        config = dataset_info[required_key]
        print(f"✓ {required_key} registered in dataset_info.json")
        print(f"    file_name: {config.get('file_name')}")
        print(f"    formatting: {config.get('formatting')}")
        print(f"    columns: {list(config.get('columns', {}).keys())}")
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
        print(f"\nPlease add this entry to {dataset_info_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()

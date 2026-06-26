#!/usr/bin/env python3
"""
Setup ablation dataset for current training config.

Automatically detects which ablation config is active from BASE_YAML_CONFIG
environment variable and creates only the required dataset.

Usage:
    python scripts/setup_ablation_dataset.py --seed 42 [--force]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


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


def check_dataset_valid(parquet_file: Path) -> bool:
    """
    Check if dataset file exists and is valid.

    Returns:
        True if valid, False otherwise
    """
    if not parquet_file.exists():
        print(f"✗ Missing: {parquet_file}")
        return False

    try:
        df = pd.read_parquet(parquet_file)

        if len(df) != 8000:
            print(f"✗ Expected 8000 samples, got {len(df)}")
            return False

        if "conversations" not in df.columns or "image" not in df.columns:
            print(f"✗ Missing required columns")
            return False

        print(f"✓ {len(df)} samples, schema OK")
        return True

    except Exception as e:
        print(f"✗ Validation error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Setup ablation dataset for current config"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if dataset exists",
    )
    parser.add_argument(
        "--project-dir",
        type=str,
        default=".",
        help="Project root directory (default: current directory)",
    )

    args = parser.parse_args()

    # Get BASE_YAML_CONFIG from environment
    base_yaml_config = os.environ.get("BASE_YAML_CONFIG", "")

    if not base_yaml_config:
        print("ERROR: BASE_YAML_CONFIG environment variable not set")
        sys.exit(1)

    # Parse ablation config
    ablation_config = parse_ablation_config(base_yaml_config)

    if not ablation_config:
        print("⚠️  Not an ablation config, skip dataset creation")
        sys.exit(0)

    print(f"📋 Detected ablation config: {ablation_config}")
    print("")

    # Paths
    project_dir = Path(args.project_dir).resolve()
    parquet_file = project_dir / "train/ablation_data" / f"{ablation_config}.parquet"

    # Check if rebuild needed
    need_rebuild = False

    if args.force:
        print("🔄 Force rebuild requested")
        need_rebuild = True
    elif not check_dataset_valid(parquet_file):
        print("🔄 Dataset invalid or missing, need rebuild")
        need_rebuild = True
    else:
        print("✅ Dataset valid, skip rebuild")

    # Build if needed
    if need_rebuild:
        print("")
        print("=" * 80)
        print(f"Creating ablation dataset: {ablation_config}")
        print("=" * 80)

        cmd = [
            sys.executable,
            "scripts/create_ablation_datasets.py",
            "--seed", str(args.seed),
            "--output-dir", "train/ablation_data",
            "--project-dir", str(project_dir),
            "--dataset-info", "train/dataset_info.json",
            "--config", ablation_config,
        ]

        result = subprocess.run(cmd, cwd=project_dir)

        if result.returncode != 0:
            print("❌ Build failed")
            sys.exit(1)

        print("")
        print("Validating build...")
        if check_dataset_valid(parquet_file):
            print("✅ Build successful")
        else:
            print("❌ Build validation failed")
            sys.exit(1)

    print("")
    print("📊 Dataset ready:")
    file_size = parquet_file.stat().st_size / (1024 * 1024)
    print(f"   {parquet_file.name} ({file_size:.2f} MB)")


if __name__ == "__main__":
    main()

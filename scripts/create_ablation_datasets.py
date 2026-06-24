#!/usr/bin/env python3
"""
Create ablation study datasets with fixed sampling ratios and seed.

This script samples from source datasets (parquet_crohme_train, parquet_crohme_train_tree, etc.)
to create fixed ablation datasets with controlled ratios for fair comparison.

Usage:
    python scripts/create_ablation_datasets.py --seed 42 --output-dir data/ablation
"""

import argparse
import json
import logging
import random
from pathlib import Path

import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
LOGGER = logging.getLogger(__name__)


# Ablation configurations
ABLATION_CONFIGS = {
    "ablation_baseline_8000": {
        "parquet_crohme_train": 8000,
    },
    "ablation_tree_8000": {
        "parquet_crohme_train": 4000,
        "parquet_crohme_train_tree": 4000,
    },
    "ablation_edl_8000": {
        "parquet_crohme_train": 4000,
        "parquet_crohme_train_error_find": 2000,
        "parquet_crohme_train_error_fix": 2000,
    },
    "ablation_counting_8000": {
        "parquet_crohme_train": 4000,
        "parquet_crohme_train_can": 4000,
    },
    "ablation_full_8000": {
        "parquet_crohme_train": 4000,
        "parquet_crohme_train_tree": 1000,
        "parquet_crohme_train_can": 1000,
        "parquet_crohme_train_error_find": 1000,
        "parquet_crohme_train_error_fix": 1000,
    },
}


def load_dataset_info(dataset_info_path: Path) -> dict:
    """Load dataset_info.json to get file paths."""
    with open(dataset_info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_parquet_dataset(file_path: Path) -> pd.DataFrame:
    """Load a parquet dataset."""
    LOGGER.info(f"Loading dataset from {file_path}")
    df = pd.read_parquet(file_path)
    LOGGER.info(f"  Loaded {len(df)} samples")
    return df


def sample_from_dataset(df: pd.DataFrame, n_samples: int, seed: int) -> pd.DataFrame:
    """Sample n_samples from dataframe with fixed seed."""
    if len(df) < n_samples:
        LOGGER.warning(
            f"Dataset has only {len(df)} samples, but requested {n_samples}. "
            f"Using all available samples."
        )
        return df.copy()

    # Sample with fixed seed
    sampled = df.sample(n=n_samples, random_state=seed)
    return sampled.reset_index(drop=True)


def create_ablation_dataset(
    config_name: str,
    sampling_config: dict,
    dataset_info: dict,
    project_dir: Path,
    output_dir: Path,
    seed: int,
) -> Path:
    """
    Create one ablation dataset by sampling from source datasets.

    Args:
        config_name: Name of ablation config (e.g., "ablation_tree_8000")
        sampling_config: Dict mapping dataset_name -> n_samples
        dataset_info: Loaded dataset_info.json
        project_dir: Project root directory
        output_dir: Output directory for ablation datasets
        seed: Random seed for reproducibility

    Returns:
        Path to created parquet file
    """
    LOGGER.info("="*80)
    LOGGER.info(f"Creating {config_name}")
    LOGGER.info("="*80)

    all_samples = []

    for dataset_name, n_samples in sampling_config.items():
        LOGGER.info(f"\nSampling {n_samples} from {dataset_name}...")

        # Get dataset file path
        if dataset_name not in dataset_info:
            raise ValueError(f"Dataset {dataset_name} not found in dataset_info.json")

        dataset_config = dataset_info[dataset_name]
        file_path = project_dir / dataset_config["file_name"]

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Load and sample
        df = load_parquet_dataset(file_path)
        sampled_df = sample_from_dataset(df, n_samples, seed)

        LOGGER.info(f"  ✓ Sampled {len(sampled_df)} samples")
        all_samples.append(sampled_df)

    # Concatenate all samples
    LOGGER.info(f"\nMerging all samples...")
    merged_df = pd.concat(all_samples, ignore_index=True)
    LOGGER.info(f"  Total before shuffle: {len(merged_df)} samples")

    # Shuffle with fixed seed
    LOGGER.info(f"Shuffling with seed={seed}...")
    merged_df = merged_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Save to parquet
    output_path = output_dir / f"{config_name}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info(f"Saving to {output_path}...")
    merged_df.to_parquet(output_path, index=False)

    LOGGER.info(f"✓ Created {config_name}: {len(merged_df)} samples")
    LOGGER.info(f"  Saved to: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Create ablation study datasets with fixed sampling"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/ablation",
        help="Output directory for ablation datasets (default: data/ablation)",
    )
    parser.add_argument(
        "--project-dir",
        type=str,
        default=".",
        help="Project root directory (default: current directory)",
    )
    parser.add_argument(
        "--dataset-info",
        type=str,
        default="train/LLaMA-Factory/data/dataset_info.json",
        help="Path to dataset_info.json (default: train/LLaMA-Factory/data/dataset_info.json)",
    )

    args = parser.parse_args()

    # Convert to Path objects
    project_dir = Path(args.project_dir).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_dir / output_dir

    dataset_info_path = project_dir / args.dataset_info

    LOGGER.info("="*80)
    LOGGER.info("Creating Ablation Study Datasets")
    LOGGER.info("="*80)
    LOGGER.info(f"Project dir: {project_dir}")
    LOGGER.info(f"Output dir: {output_dir}")
    LOGGER.info(f"Seed: {args.seed}")
    LOGGER.info(f"Dataset info: {dataset_info_path}")
    LOGGER.info("")

    # Load dataset_info.json
    if not dataset_info_path.exists():
        raise FileNotFoundError(f"dataset_info.json not found: {dataset_info_path}")

    dataset_info = load_dataset_info(dataset_info_path)
    LOGGER.info(f"Loaded dataset_info.json with {len(dataset_info)} datasets")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create each ablation dataset
    created_files = []
    for config_name, sampling_config in ABLATION_CONFIGS.items():
        try:
            output_path = create_ablation_dataset(
                config_name=config_name,
                sampling_config=sampling_config,
                dataset_info=dataset_info,
                project_dir=project_dir,
                output_dir=output_dir,
                seed=args.seed,
            )
            created_files.append(output_path)
        except Exception as e:
            LOGGER.error(f"Failed to create {config_name}: {e}")
            raise

    # Summary
    LOGGER.info("\n" + "="*80)
    LOGGER.info("Summary")
    LOGGER.info("="*80)
    LOGGER.info(f"Successfully created {len(created_files)} ablation datasets:")
    for path in created_files:
        file_size = path.stat().st_size / (1024 * 1024)  # MB
        LOGGER.info(f"  ✓ {path.name} ({file_size:.2f} MB)")

    LOGGER.info("\n" + "="*80)
    LOGGER.info("Next steps:")
    LOGGER.info("="*80)
    LOGGER.info("1. Register these datasets in train/LLaMA-Factory/data/dataset_info.json")
    LOGGER.info("2. Create YAML configs in train/ablation/")
    LOGGER.info("3. Train each config and compare results")
    LOGGER.info("")


if __name__ == "__main__":
    main()

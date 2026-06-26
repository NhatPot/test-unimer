#!/usr/bin/env python3
"""
Create ablation study datasets with fixed sampling ratios and seed.

This script samples from source datasets (parquet_crohme_train, parquet_crohme_train_tree, etc.)
to create fixed ablation datasets with controlled ratios for fair comparison.

Usage:
    python scripts/create_ablation_datasets.py --seed 42 --output-dir train/ablation_data
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

try:
    from datasets import Dataset, concatenate_datasets, load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
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


def load_hf_dataset_subset(dataset_name: str, dataset_info: dict, project_dir: Path):
    """
    Load dataset from HuggingFace or local parquet using datasets library.

    Returns:
        Dataset object from HuggingFace datasets library
    """
    if not HF_DATASETS_AVAILABLE:
        raise ImportError("HuggingFace datasets library required. Install: pip install datasets")

    config = dataset_info[dataset_name]

    # Try HuggingFace Hub first
    if "hf_hub_url" in config:
        hf_url = config["hf_hub_url"]
        subset = config.get("subset")

        LOGGER.info(f"  Loading from HuggingFace: {hf_url}, subset={subset}")
        ds = load_dataset(hf_url, subset, split="train")
        return ds

    # Fallback: local parquet
    elif "file_name" in config:
        file_path = project_dir / config["file_name"]
        LOGGER.info(f"  Loading from local: {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        ds = load_dataset("parquet", data_files=str(file_path), split="train")
        return ds

    else:
        raise ValueError(f"Dataset {dataset_name} has no hf_hub_url or file_name")


def validate_parquet(file_path: Path, expected_count: int = 8000) -> bool:
    """
    Validate parquet has correct number of samples and schema.
    Does not assert dtype to avoid false negatives.
    """
    if not HF_DATASETS_AVAILABLE:
        import pandas as pd
        df = pd.read_parquet(file_path)

        if len(df) != expected_count:
            LOGGER.error(f"Expected {expected_count} samples, got {len(df)}")
            return False

        if "conversations" not in df.columns or "image" not in df.columns:
            LOGGER.error("Missing required columns: conversations, image")
            return False

        LOGGER.info(f"  Image dtype: {df['image'].dtype}")
        LOGGER.info(f"  Conversations dtype: {df['conversations'].dtype}")
        return True

    else:
        ds = load_dataset("parquet", data_files=str(file_path), split="train")

        if len(ds) != expected_count:
            LOGGER.error(f"Expected {expected_count} samples, got {len(ds)}")
            return False

        if "conversations" not in ds.column_names or "image" not in ds.column_names:
            LOGGER.error("Missing required columns: conversations, image")
            return False

        LOGGER.info(f"  Columns: {ds.column_names}")
        return True


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
    Uses HuggingFace datasets library to preserve image encoding.

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

    if not HF_DATASETS_AVAILABLE:
        raise ImportError("HuggingFace datasets library required for this operation")

    all_datasets = []

    for dataset_name, n_samples in sampling_config.items():
        LOGGER.info(f"\nSampling {n_samples} from {dataset_name}...")

        if dataset_name not in dataset_info:
            raise ValueError(f"Dataset {dataset_name} not found in dataset_info.json")

        # Load from HF or local
        ds = load_hf_dataset_subset(dataset_name, dataset_info, project_dir)

        # Sample with seed
        if len(ds) < n_samples:
            LOGGER.warning(f"Dataset has only {len(ds)}, using all samples")
            sampled_ds = ds
        else:
            sampled_ds = ds.shuffle(seed=seed).select(range(n_samples))

        LOGGER.info(f"  ✓ Sampled {len(sampled_ds)} samples")
        all_datasets.append(sampled_ds)

    # Concatenate (no pandas conversion)
    LOGGER.info(f"\nMerging {len(all_datasets)} datasets...")
    merged_ds = concatenate_datasets(all_datasets)
    LOGGER.info(f"  Total: {len(merged_ds)} samples")

    # Shuffle
    LOGGER.info(f"Shuffling with seed={seed}...")
    merged_ds = merged_ds.shuffle(seed=seed)

    # Save to parquet using Dataset.to_parquet (preserves encoding)
    output_path = output_dir / f"{config_name}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info(f"Saving to {output_path}...")
    merged_ds.to_parquet(output_path)

    LOGGER.info(f"✓ Created {config_name}: {len(merged_ds)} samples")
    LOGGER.info(f"  Saved to: {output_path}")

    # Validate
    LOGGER.info(f"Validating {config_name}...")
    if validate_parquet(output_path, expected_count=8000):
        LOGGER.info(f"  ✓ Validation passed")
    else:
        raise RuntimeError(f"Validation failed for {config_name}")

    return output_path


def create_manifest(output_dir: Path, seed: int, created_files: list, configs_built: dict) -> Path:
    """
    Create manifest JSON file for ablation datasets audit.

    Args:
        output_dir: Output directory
        seed: Random seed used
        created_files: List of created parquet file paths
        configs_built: Dict of configs that were actually built

    Returns:
        Path to manifest JSON
    """
    manifest = {
        "seed": seed,
        "created_at": datetime.now().isoformat(),
        "total_configs": len(configs_built),
        "configs": configs_built
    }

    manifest_path = output_dir / "ablation_manifest.json"

    # Update existing manifest if exists
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                existing_manifest = json.load(f)
            # Merge configs
            existing_manifest["configs"].update(configs_built)
            existing_manifest["total_configs"] = len(existing_manifest["configs"])
            manifest = existing_manifest
        except:
            pass  # Use new manifest if can't read existing

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    LOGGER.info(f"✓ Created/updated manifest: {manifest_path}")
    return manifest_path


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
        default="train/dataset_info.json",
        help="Path to dataset_info.json (default: train/dataset_info.json)",
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=list(ABLATION_CONFIGS.keys()),
        help="Specific config to build (e.g., ablation_baseline_8000). If not specified, build all configs.",
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

    # Determine which configs to build
    if args.config:
        LOGGER.info(f"Building single config: {args.config}")
        configs_to_build = {args.config: ABLATION_CONFIGS[args.config]}
    else:
        LOGGER.info("Building all configs")
        configs_to_build = ABLATION_CONFIGS

    # Create each ablation dataset
    created_files = []
    for config_name, sampling_config in configs_to_build.items():
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

    # Create manifest
    LOGGER.info("\n" + "="*80)
    LOGGER.info("Creating manifest")
    LOGGER.info("="*80)
    manifest_path = create_manifest(output_dir, args.seed, created_files, configs_to_build)

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
    LOGGER.info("1. Register these datasets in train/dataset_info.json")
    LOGGER.info("2. Verify with Cell 5.5 in notebook")
    LOGGER.info("3. Train each config and compare results")
    LOGGER.info("")


if __name__ == "__main__":
    main()

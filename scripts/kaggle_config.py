#!/usr/bin/env python3
"""
Kaggle notebook configuration setup.
Handles environment variables and paths for Kaggle training runs.
"""

import json
import os
import sys
import uuid
from pathlib import Path

from kaggle_secrets import UserSecretsClient


def setup_kaggle_env(
    base_yaml_config="train/Uni-MuMER-train.yaml",
    notebook_path="uni-mumer-kaggle-dagshub v8.ipynb",
):
    """
    Setup all environment variables for Kaggle notebook.

    Args:
        base_yaml_config: Path to YAML config (e.g., "train/ablation/ablation_baseline_8000.yaml")
        notebook_path: Name of the notebook file

    Returns:
        Tuple of (RUN_UUID, env dict)
    """
    # Project paths
    PROJECT_DIR = "/kaggle/working/test-unimer"
    CONDA_DIR = "/kaggle/working/miniconda"
    ENV_DIR = f"{CONDA_DIR}/envs/unimumer"
    PYTHON = f"{ENV_DIR}/bin/python"

    # Runtime YAML path
    RUNTIME_YAML_CONFIG = "/kaggle/working/runtime_Uni-MuMER-train.yaml"

    # Initial config
    YAML_CONFIG = base_yaml_config
    OUTPUT_DIR = "saves/qwen2.5_vl-3b/qlora/sft/standred/uni-mumer_qlora"

    # DagsHub config
    DAGSHUB_USERNAME = "NhatPot"
    DAGSHUB_REPO = "test-unimer"
    EXPERIMENT_NAME = "Uni-MuMER-Qwen2.5-VL-3B"

    # Generate run UUID
    RUN_UUID = uuid.uuid4().hex

    # Setup Python path for imports
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    os.environ["PYTHONPATH"] = PROJECT_DIR

    # Get DagsHub token from Kaggle Secrets
    DAGSHUB_TOKEN = UserSecretsClient().get_secret("DAGSHUB_TOKEN")
    if not DAGSHUB_TOKEN:
        raise RuntimeError("Kaggle Secret DAGSHUB_TOKEN is missing")

    # Set all environment variables
    os.environ.update({
        "PROJECT_DIR": PROJECT_DIR,
        "CONDA_DIR": CONDA_DIR,
        "ENV_DIR": ENV_DIR,
        "PYTHON": PYTHON,
        "BASE_YAML_CONFIG": base_yaml_config,
        "RUNTIME_YAML_CONFIG": RUNTIME_YAML_CONFIG,
        "YAML_CONFIG": YAML_CONFIG,
        "NOTEBOOK_PATH": notebook_path,
        "OUTPUT_DIR": OUTPUT_DIR,
        "RUN_UUID": RUN_UUID,
        "MLFLOW_TRACKING_URI": f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow",
        "MLFLOW_TRACKING_USERNAME": DAGSHUB_USERNAME,
        "MLFLOW_TRACKING_PASSWORD": DAGSHUB_TOKEN,
        "MLFLOW_EXPERIMENT_NAME": EXPERIMENT_NAME,
        "MLFLOW_FLATTEN_PARAMS": "TRUE",
        "MLFLOW_TAGS": json.dumps({
            "run_uuid": RUN_UUID,
            "source": "kaggle",
            "task": "sft",
            "dataset": "parquet_crohme_train",
        }),
    })

    # Print summary
    print(f"Run UUID: {RUN_UUID}")
    print(f"MLflow: {os.environ['MLFLOW_TRACKING_URI']}")
    print(f"PROJECT_DIR: {PROJECT_DIR}")
    print(f"BASE_YAML_CONFIG: {base_yaml_config}")
    print(f"Is Ablation Mode: {'ablation' in base_yaml_config}")
    print(f"PROJECT_DIR in sys.path: {PROJECT_DIR in sys.path}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}")
    print(f"runtime_yaml.py exists: {Path(PROJECT_DIR, 'scripts/runtime_yaml.py').exists()}")

    return RUN_UUID, os.environ

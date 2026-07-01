from __future__ import annotations

import os
from pathlib import Path


MODEL_FILE_CANDIDATES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "model.safetensors",
    "pytorch_model.bin",
)

FULL_CHECKPOINT_FILES = (
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
)


def _resolve_path(project_dir: str | Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else Path(project_dir) / path


def _configure_mlflow():
    import mlflow

    required = (
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
    )
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing DagsHub credentials: {', '.join(missing)}")

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    return mlflow


def _find_run_id_by_uuid(experiment_name: str, run_uuid: str) -> str:
    mlflow = _configure_mlflow()
    from mlflow import MlflowClient

    experiment = mlflow.set_experiment(experiment_name)
    client = MlflowClient()
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.run_uuid = '{run_uuid}'",
        max_results=1,
        order_by=["attributes.start_time DESC"],
    )
    if not runs:
        raise RuntimeError(f"No MLflow run found for run_uuid={run_uuid}")
    return runs[0].info.run_id


def _validate_checkpoint(checkpoint_dir: Path, require_full_checkpoint: bool) -> None:
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    missing = [name for name in ("trainer_state.json",) if not (checkpoint_dir / name).is_file()]
    if not any((checkpoint_dir / name).is_file() for name in MODEL_FILE_CANDIDATES):
        missing.append("adapter_model.safetensors/model.safetensors")

    if require_full_checkpoint:
        missing.extend(name for name in FULL_CHECKPOINT_FILES if not (checkpoint_dir / name).is_file())

    if missing:
        raise RuntimeError(
            "Checkpoint is not resume-ready. Missing: "
            + ", ".join(dict.fromkeys(missing))
            + f"\nCheckpoint: {checkpoint_dir}"
        )


def prepare_resume_checkpoint(
    project_dir: str | Path,
    output_dir: str | Path,
    enabled: bool = False,
    run_id: str | None = None,
    run_uuid: str | None = None,
    experiment_name: str | None = None,
    artifact_path: str | None = None,
    require_full_checkpoint: bool = True,
) -> str | None:
    """Download a DagsHub MLflow checkpoint artifact for LLaMA-Factory resume."""
    print(f"DagsHub Resume: {'ON' if enabled else 'OFF'}")
    if not enabled:
        os.environ.pop("RESUME_CHECKPOINT", None)
        return None

    if not artifact_path:
        raise ValueError("artifact_path is required when DagsHub resume is enabled")

    if not run_id:
        if not run_uuid:
            raise ValueError("run_id or run_uuid is required when DagsHub resume is enabled")
        if not experiment_name:
            raise ValueError("experiment_name is required when using run_uuid")
        run_id = _find_run_id_by_uuid(experiment_name, run_uuid)

    mlflow = _configure_mlflow()
    output_path = _resolve_path(project_dir, output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint_name = Path(artifact_path).name
    expected_path = output_path / checkpoint_name
    if expected_path.is_dir():
        local_checkpoint = expected_path
        print(f"Reusing local checkpoint: {local_checkpoint}")
    else:
        local_checkpoint = Path(
            mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=artifact_path,
                dst_path=str(output_path),
            )
        )
        nested_checkpoint = local_checkpoint / checkpoint_name
        if nested_checkpoint.is_dir():
            local_checkpoint = nested_checkpoint

    _validate_checkpoint(local_checkpoint, require_full_checkpoint=require_full_checkpoint)
    os.environ["RESUME_CHECKPOINT"] = str(local_checkpoint)

    print(f"Resume run_id: {run_id}")
    print(f"Resume artifact: {artifact_path}")
    print(f"Local checkpoint: {local_checkpoint}")
    return str(local_checkpoint)

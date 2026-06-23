from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient


def _command(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _configure(experiment_name: str):
    required = (
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
    )
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing DagsHub credentials: {', '.join(missing)}")

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    experiment = mlflow.set_experiment(experiment_name)
    return MlflowClient(), experiment


def check_connection(experiment_name: str) -> None:
    client, experiment = _configure(experiment_name)
    run = client.create_run(experiment.experiment_id, tags={"purpose": "connectivity-check"})
    try:
        client.log_metric(run.info.run_id, "connection", 1.0)
        client.set_terminated(run.info.run_id)
    finally:
        try:
            client.delete_run(run.info.run_id)
        except Exception:
            pass

    print(f"DagsHub MLflow connection OK: {os.environ['MLFLOW_TRACKING_URI']}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_run_files(output_dir: Path, project_dir: Path) -> None:
    freeze = _command(sys.executable, "-m", "pip", "freeze")
    (output_dir / "requirements-lock.txt").write_text(freeze + "\n", encoding="utf-8")

    git_status = _command("git", "status", "--short", cwd=project_dir)
    metadata = {
        "git_commit": _command("git", "rev-parse", "HEAD", cwd=project_dir),
        "git_branch": _command("git", "branch", "--show-current", cwd=project_dir),
        "git_dirty": bool(git_status),
        "gpu": _command(
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ),
        "python": sys.version,
        "run_uuid": os.getenv("RUN_UUID", ""),
        "run_name": f"uni-mumer-{os.getenv('RUN_UUID', '')[:8]}",
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    files = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            files.append(
                {
                    "path": path.relative_to(output_dir).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )

    manifest = {"file_count": len(files), "files": files}
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _find_run(client: MlflowClient, experiment_id: str, run_uuid: str):
    runs = client.search_runs(
        [experiment_id],
        filter_string=f"tags.run_uuid = '{run_uuid}'",
        max_results=1,
        order_by=["attributes.start_time DESC"],
    )
    if not runs:
        raise RuntimeError(f"No MLflow run found for run_uuid={run_uuid}")
    return runs[0]


def _final_metrics(output_dir: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name in ("train_results.json", "all_results.json"):
        path = output_dir / name
        if not path.exists():
            continue
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        for key, value in data.items():
            if isinstance(value, (int, float)):
                metrics[f"final_{key}"] = float(value)
    return metrics


def upload_run(
    experiment_name: str,
    run_uuid: str,
    config_path: Path,
    output_dir: Path,
    project_dir: Path,
    notebook_path: Path,
) -> None:
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Training output not found: {output_dir}")

    if not notebook_path.is_absolute():
        notebook_path = project_dir / notebook_path
    if not notebook_path.is_file():
        raise FileNotFoundError(f"Notebook source not found: {notebook_path}")

    client, experiment = _configure(experiment_name)
    run = _find_run(client, experiment.experiment_id, run_uuid)
    _write_run_files(output_dir, project_dir)

    checkpoints = [
        path for path in output_dir.glob("checkpoint-*") if path.name.rsplit("-", 1)[-1].isdigit()
    ]
    latest = max(checkpoints, key=lambda path: int(path.name.rsplit("-", 1)[-1]), default=None)

    with mlflow.start_run(run_id=run.info.run_id):
        mlflow.set_tags(
            {
                "git.commit": _command("git", "rev-parse", "HEAD", cwd=project_dir),
                "checkpoint.latest": latest.name if latest else "output_dir",
            }
        )
        metrics = _final_metrics(output_dir)
        if metrics:
            mlflow.log_metrics(metrics)

        for path in (
            config_path,
            project_dir / "train" / "dataset_info.json",
            project_dir / "requirements.txt",
            notebook_path,
        ):
            if path.is_file():
                mlflow.log_artifact(str(path), artifact_path="configuration")

        mlflow.log_artifacts(str(output_dir), artifact_path="training")

    artifacts = client.list_artifacts(run.info.run_id)
    if not artifacts:
        raise RuntimeError("DagsHub accepted the run but no artifact was found")

    run_url = (
        f"{os.environ['MLFLOW_TRACKING_URI']}/#/experiments/"
        f"{experiment.experiment_id}/runs/{run.info.run_id}"
    )
    print(f"Uploaded {len(artifacts)} artifact groups to DagsHub")
    print(f"Run: {run_url}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DagsHub MLflow utilities for Uni-MuMER")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="Verify DagsHub write access")
    check.add_argument("--experiment", required=True)

    upload = subparsers.add_parser("upload", help="Upload and verify training artifacts")
    upload.add_argument("--experiment", required=True)
    upload.add_argument("--run-uuid", required=True)
    upload.add_argument("--config", type=Path, required=True)
    upload.add_argument("--output-dir", type=Path, required=True)
    upload.add_argument("--project-dir", type=Path, default=Path.cwd())
    upload.add_argument("--notebook", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "check":
        check_connection(args.experiment)
    else:
        upload_run(
            args.experiment,
            args.run_uuid,
            args.config,
            args.output_dir,
            args.project_dir.resolve(),
            args.notebook,
        )


if __name__ == "__main__":
    main()

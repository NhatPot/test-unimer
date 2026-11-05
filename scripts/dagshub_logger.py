from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import mlflow
from mlflow import MlflowClient
import yaml


def _command(*args: str, cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)
    except OSError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _configure(experiment_name: str, *, allow_create: bool = False):
    required = (
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
    )
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing DagsHub credentials: {', '.join(missing)}")

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None and allow_create:
        experiment = mlflow.set_experiment(experiment_name)
    if experiment is None:
        raise RuntimeError("MLflow experiment not found")
    return client, experiment


def _safe_run_url(experiment_id: str, run_id: str) -> str:
    uri = urlsplit(os.environ["MLFLOW_TRACKING_URI"])
    if uri.scheme not in ("http", "https") or not uri.hostname:
        return "(run URL unavailable)"
    host = f"[{uri.hostname}]" if ":" in uri.hostname else uri.hostname
    if uri.port:
        host += f":{uri.port}"
    return urlunsplit((uri.scheme, host, uri.path.rstrip("/") + "/", "",
                       f"/experiments/{experiment_id}/runs/{run_id}"))


def check_connection(experiment_name: str) -> None:
    client, experiment = _configure(experiment_name, allow_create=True)
    run = client.create_run(experiment.experiment_id, tags={"purpose": "connectivity-check"})
    try:
        client.log_metric(run.info.run_id, "connection", 1.0)
        client.set_terminated(run.info.run_id)
    finally:
        try:
            client.delete_run(run.info.run_id)
        except Exception:
            pass

    print("DagsHub MLflow connection OK")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_run_files(
    output_dir: Path, project_dir: Path, run_metadata: dict[str, Any] | None = None,
) -> None:
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
        "run_name": f"uni-mumer-{os.getenv('RUN_UUID', '')}",
        **(run_metadata or {}),
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
    _validate_run_uuid(run_uuid)
    runs = client.search_runs(
        [experiment_id],
        filter_string=f"tags.run_uuid = '{run_uuid}'",
        max_results=2,
    )
    if not runs:
        raise RuntimeError(f"No MLflow run found for run_uuid={run_uuid}")
    if len(runs) != 1:
        raise RuntimeError("Ambiguous MLflow run_uuid: multiple runs found")
    return runs[0]


def _validate_run_uuid(run_uuid: str) -> None:
    if not re.fullmatch(r"(?:[0-9a-fA-F]{32}|[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})", run_uuid):
        raise ValueError("run_uuid must be a full hexadecimal UUID")


def _runtime_config(config_path: Path) -> dict[str, Any]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Runtime YAML must be a mapping")
    return data


def _output_role(config: dict[str, Any]) -> str:
    return "final_output" if config.get("load_best_model_at_end") is False else "best_or_reloaded_output"


def _final_metrics(output_dir: Path, config_path: Path | None = None) -> dict[str, float]:
    config = _runtime_config(config_path) if config_path is not None else {}
    role = _output_role(config)
    metrics: dict[str, float] = {}
    for name in ("train_results.json", "eval_results.json", "all_results.json"):
        path = output_dir / name
        if not path.exists():
            continue
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        for key, value in data.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[f"{role}_{key}"] = float(value)
    state_path = output_dir / "trainer_state.json"
    if state_path.is_file() and config.get("metric_for_best_model") in ("loss", "eval_loss"):
        best = json.loads(state_path.read_text(encoding="utf-8")).get("best_metric")
        if isinstance(best, (int, float)) and not isinstance(best, bool):
            metrics["best_validation_eval_loss"] = float(best)
    return metrics


def _configuration_files(config_path: Path, notebook_path: Path, project_dir: Path,
                         config: dict[str, Any]) -> list[tuple[Path, str]]:
    files = [(config_path, "configuration"), (notebook_path, "configuration")]
    for path in (config_path, notebook_path):
        if not path.is_file():
            raise FileNotFoundError("Required runtime configuration or notebook source missing")

    split = os.getenv("SPLIT_MANIFEST")
    if split:
        files.append((project_dir / split, "configuration/split_manifest"))
    dataset_dir = os.getenv("DATASET_DIR") or config.get("dataset_dir")
    if dataset_dir:
        files.append((project_dir / str(dataset_dir) / "dataset_info.json",
                      "configuration/generated_registry"))
    for path, _ in files:
        if not path.is_file():
            raise FileNotFoundError("Required split manifest or active dataset registry missing")

    for path in (project_dir / "train" / "dataset_info.json", project_dir / "requirements.txt"):
        if path.is_file():
            if all(path.resolve() != source.resolve() for source, _ in files):
                files.append((path, "configuration"))
        else:
            print(f"Skipping absent optional historical configuration: {path.name}")
    targets = [f"{target}/{path.name}" for path, target in files]
    if len(set(targets)) != len(targets):
        raise ValueError("Configuration artifact names collide")
    return files


def _verify_artifacts(client: MlflowClient, run_id: str, expected: dict[str, int]) -> None:
    remote: dict[str, int | None] = {}
    pending = [""]
    visited = set()
    while pending:
        prefix = pending.pop()
        if prefix in visited:
            continue
        visited.add(prefix)
        for entry in client.list_artifacts(run_id, path=prefix):
            if entry.is_dir:
                pending.append(entry.path)
            else:
                remote[entry.path] = entry.file_size
    mismatched = sum(remote.get(path) != size for path, size in expected.items())
    if mismatched:
        raise RuntimeError(f"Remote artifact sizes_only verification failed for {mismatched} expected file(s)")


def upload_run(
    experiment_name: str,
    run_uuid: str,
    config_path: Path,
    output_dir: Path,
    project_dir: Path,
    notebook_path: Path,
) -> None:
    _validate_run_uuid(run_uuid)
    config_path = project_dir / config_path
    output_dir = project_dir / output_dir
    config = _runtime_config(config_path)
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Training output not found: {output_dir}")

    if not notebook_path.is_absolute():
        notebook_path = project_dir / notebook_path
    if not notebook_path.is_file():
        raise FileNotFoundError(f"Notebook source not found: {notebook_path}")

    configuration = _configuration_files(config_path, notebook_path, project_dir, config)
    client, experiment = _configure(experiment_name)
    run = _find_run(client, experiment.experiment_id, run_uuid)

    checkpoints = [
        path for path in output_dir.glob("checkpoint-*")
        if path.is_dir() and path.name.rsplit("-", 1)[-1].isdigit()
    ]
    latest = max(checkpoints, key=lambda path: int(path.name.rsplit("-", 1)[-1]), default=None)

    tags = json.loads(os.getenv("MLFLOW_TAGS", "{}"))
    if not isinstance(tags, dict):
        raise ValueError("MLFLOW_TAGS must be a JSON object")
    tags.update({"run_uuid": run_uuid,
                 "experiment_variant": os.getenv("EXPERIMENT_VARIANT", "baseline")})
    for name in ("SPLIT_MANIFEST", "DATASET_DIR", "SPLIT_SEED", "SPLIT_ID"):
        if os.getenv(name):
            tags[name.lower()] = os.environ[name]
    split_file = next((path for path, target in configuration
                       if target == "configuration/split_manifest"), None)
    if split_file is not None:
        tags["split_manifest_sha256"] = _sha256(split_file)
    run_name = run.data.tags.get("mlflow.runName") or f"uni-mumer-{run_uuid}"
    if run_uuid not in run_name:
        run_name = f"{run_name}-{run_uuid}"
    role = _output_role(config)
    trainer_state_path = output_dir / "trainer_state.json"
    trainer_state = (json.loads(trainer_state_path.read_text(encoding="utf-8"))
                     if trainer_state_path.is_file() else {})
    _write_run_files(output_dir, project_dir, {
        "run_uuid": run_uuid, "run_name": run_name,
        "mlflow_run_id": run.info.run_id, "mlflow_experiment_id": experiment.experiment_id,
        "mlflow_experiment_name": experiment_name, "tags": tags,
        "runtime_config_artifact": f"configuration/{config_path.name}",
        "load_best_model_at_end": config.get("load_best_model_at_end"),
        "output_artifact_path": "training", "output_model_role": role,
        "latest_checkpoint": latest.name if latest else None,
        "best_checkpoint": trainer_state.get("best_model_checkpoint"),
        "best_metric": trainer_state.get("best_metric"),
        "best_global_step": trainer_state.get("best_global_step"),
        "final_global_step": trainer_state.get("global_step"),
        "final_epoch": trainer_state.get("epoch"),
        "artifact_verification_level": "sizes_only",
    })
    expected = {f"{target}/{path.name}": path.stat().st_size for path, target in configuration}
    expected.update({f"training/{path.relative_to(output_dir).as_posix()}": path.stat().st_size
                     for path in output_dir.rglob("*") if path.is_file()})

    with mlflow.start_run(run_id=run.info.run_id):
        mlflow.set_tags(
            {
                "git.commit": _command("git", "rev-parse", "HEAD", cwd=project_dir),
                **tags,
                "mlflow.runName": run_name,
                "checkpoint.latest": latest.name if latest else "none",
                "output.model_role": role,
                "artifact.verification_level": "sizes_only",
                "artifact.verification_status": "pending",
            }
        )
        metrics = _final_metrics(output_dir, config_path)
        if metrics:
            mlflow.log_metrics(metrics)

        for path, target in configuration:
            mlflow.log_artifact(str(path), artifact_path=target)

        mlflow.log_artifacts(str(output_dir), artifact_path="training")

    _verify_artifacts(client, run.info.run_id, expected)
    client.set_tag(run.info.run_id, "artifact.verification_status", "verified")
    print(f"Verified {len(expected)} expected remote files (sizes_only; no remote SHA256 verification)")
    print(f"Run: {_safe_run_url(experiment.experiment_id, run.info.run_id)}")


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
    try:
        main()
    except Exception as exc:
        print(f"DagsHub logging failed ({type(exc).__name__}); remote exception details suppressed.", file=sys.stderr)
        sys.exit(1)

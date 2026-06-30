from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


CHECKPOINT_FILE_CANDIDATES = (
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


def _resolve_path(path: str | Path, project_dir: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else project_dir / path


def _configure_mlflow(experiment_name: str):
    import mlflow
    from mlflow import MlflowClient

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
    return MlflowClient(), experiment.experiment_id


def _find_run(client, experiment_id: str, run_uuid: str):
    runs = client.search_runs(
        [experiment_id],
        filter_string=f"tags.run_uuid = '{run_uuid}'",
        max_results=1,
        order_by=["attributes.start_time DESC"],
    )
    return runs[0] if runs else None


def _wait_for_run(
    client,
    experiment_id: str,
    run_uuid: str,
    stop_event: threading.Event,
    poll_seconds: int,
):
    while not stop_event.is_set():
        run = _find_run(client, experiment_id, run_uuid)
        if run:
            print(f"[checkpoint-sync] Found MLflow run: {run.info.run_id}", flush=True)
            return run
        print("[checkpoint-sync] Waiting for MLflow run...", flush=True)
        stop_event.wait(poll_seconds)
    return None


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[-1])
    except ValueError:
        return -1


def _file_snapshot(path: Path) -> dict[str, tuple[int, int]]:
    snapshot: dict[str, tuple[int, int]] = {}
    for item in path.rglob("*"):
        if item.is_file():
            stat = item.stat()
            snapshot[item.relative_to(path).as_posix()] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


def _checkpoint_ready(
    checkpoint_dir: Path,
    stable_seconds: int,
    require_full_checkpoint: bool,
) -> tuple[bool, str]:
    if not checkpoint_dir.is_dir():
        return False, "not a directory"

    if not (checkpoint_dir / "trainer_state.json").is_file():
        return False, "missing trainer_state.json"

    if not any((checkpoint_dir / name).is_file() for name in CHECKPOINT_FILE_CANDIDATES):
        return False, "missing model/adapter file"

    if require_full_checkpoint:
        missing = [name for name in FULL_CHECKPOINT_FILES if not (checkpoint_dir / name).is_file()]
        if missing:
            return False, f"missing full-checkpoint files: {', '.join(missing)}"

    before = _file_snapshot(checkpoint_dir)
    if not before:
        return False, "empty checkpoint"
    time.sleep(stable_seconds)
    after = _file_snapshot(checkpoint_dir)
    if before != after:
        return False, "checkpoint is still being written"

    return True, "ready"


def _load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"uploaded": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"uploaded": {}}
    if not isinstance(data, dict):
        return {"uploaded": {}}
    data.setdefault("uploaded", {})
    return data


def _save_state(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _sync_checkpoints_once(
    client,
    run_id: str,
    output_dir: Path,
    artifact_path: str,
    state_path: Path,
    stable_seconds: int,
    require_full_checkpoint: bool,
) -> int:
    import mlflow

    state = _load_state(state_path)
    uploaded: dict[str, Any] = state.setdefault("uploaded", {})
    checkpoint_dirs = sorted(
        (path for path in output_dir.glob("checkpoint-*") if _checkpoint_step(path) >= 0),
        key=_checkpoint_step,
    )

    uploaded_count = 0
    for checkpoint_dir in checkpoint_dirs:
        name = checkpoint_dir.name
        if name in uploaded:
            continue

        ready, reason = _checkpoint_ready(
            checkpoint_dir,
            stable_seconds=stable_seconds,
            require_full_checkpoint=require_full_checkpoint,
        )
        if not ready:
            print(f"[checkpoint-sync] Skip {name}: {reason}", flush=True)
            continue

        target = f"{artifact_path}/{name}".strip("/")
        print(f"[checkpoint-sync] Uploading {name} -> {target}", flush=True)
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifacts(str(checkpoint_dir), artifact_path=target)
            mlflow.set_tags(
                {
                    "checkpoint.synced.latest": name,
                    "checkpoint.synced.latest_step": str(_checkpoint_step(checkpoint_dir)),
                }
            )

        uploaded[name] = {
            "step": _checkpoint_step(checkpoint_dir),
            "artifact_path": target,
            "uploaded_at": int(time.time()),
        }
        state["latest"] = name
        _save_state(state_path, state)
        uploaded_count += 1

    return uploaded_count


def _watch_checkpoints(
    experiment_name: str,
    run_uuid: str,
    output_dir: Path,
    artifact_path: str,
    state_path: Path,
    interval_seconds: int,
    stable_seconds: int,
    require_full_checkpoint: bool,
    stop_event: threading.Event,
) -> None:
    try:
        client, experiment_id = _configure_mlflow(experiment_name)
        run = _wait_for_run(
            client=client,
            experiment_id=experiment_id,
            run_uuid=run_uuid,
            stop_event=stop_event,
            poll_seconds=max(10, min(interval_seconds, 60)),
        )
        if not run:
            return

        while not stop_event.is_set():
            try:
                count = _sync_checkpoints_once(
                    client=client,
                    run_id=run.info.run_id,
                    output_dir=output_dir,
                    artifact_path=artifact_path,
                    state_path=state_path,
                    stable_seconds=stable_seconds,
                    require_full_checkpoint=require_full_checkpoint,
                )
                if count:
                    print(f"[checkpoint-sync] Uploaded {count} checkpoint(s)", flush=True)
            except Exception as exc:
                print(f"[checkpoint-sync] Warning: {exc}", flush=True)
            stop_event.wait(interval_seconds)
    except Exception as exc:
        print(f"[checkpoint-sync] Disabled: {exc}", flush=True)


def _final_sync(
    experiment_name: str,
    run_uuid: str,
    output_dir: Path,
    artifact_path: str,
    state_path: Path,
    stable_seconds: int,
    require_full_checkpoint: bool,
) -> None:
    try:
        client, experiment_id = _configure_mlflow(experiment_name)
        run = _find_run(client, experiment_id, run_uuid)
        if not run:
            print("[checkpoint-sync] Final sync skipped: MLflow run not found", flush=True)
            return
        count = _sync_checkpoints_once(
            client=client,
            run_id=run.info.run_id,
            output_dir=output_dir,
            artifact_path=artifact_path,
            state_path=state_path,
            stable_seconds=stable_seconds,
            require_full_checkpoint=require_full_checkpoint,
        )
        print(f"[checkpoint-sync] Final sync uploaded {count} checkpoint(s)", flush=True)
    except Exception as exc:
        print(f"[checkpoint-sync] Final sync warning: {exc}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLaMA-Factory training and upload checkpoints to DagsHub periodically."
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--run-uuid", required=True)
    parser.add_argument("--yaml-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--artifact-path", default="checkpoints")
    parser.add_argument("--interval-seconds", type=int, default=180)
    parser.add_argument("--stable-seconds", type=int, default=20)
    parser.add_argument("--state-path", default="/kaggle/working/checkpoint_sync_state.json")
    parser.add_argument("--allow-model-only", action="store_true")
    parser.add_argument("train_overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    output_dir = _resolve_path(args.output_dir, project_dir)
    state_path = _resolve_path(args.state_path, project_dir)
    stop_event = threading.Event()
    process_holder: dict[str, subprocess.Popen[Any] | None] = {"process": None}

    def stop_handler(signum, _frame):
        print(f"[train-wrapper] Received signal {signum}; stopping checkpoint sync.", flush=True)
        stop_event.set()
        process = process_holder["process"]
        if process and process.poll() is None:
            try:
                process.send_signal(signum)
            except Exception:
                process.terminate()

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)

    watcher = threading.Thread(
        target=_watch_checkpoints,
        kwargs={
            "experiment_name": args.experiment,
            "run_uuid": args.run_uuid,
            "output_dir": output_dir,
            "artifact_path": args.artifact_path,
            "state_path": state_path,
            "interval_seconds": args.interval_seconds,
            "stable_seconds": args.stable_seconds,
            "require_full_checkpoint": not args.allow_model_only,
            "stop_event": stop_event,
        },
        daemon=True,
    )
    watcher.start()

    train_command = [
        "llamafactory-cli",
        "train",
        args.yaml_config,
        f"run_name={args.run_name}",
        *args.train_overrides,
    ]

    print(f"[train-wrapper] YAML_CONFIG: {args.yaml_config}", flush=True)
    print(f"[train-wrapper] OUTPUT_DIR: {output_dir}", flush=True)
    print(f"[train-wrapper] Checkpoint sync interval: {args.interval_seconds}s", flush=True)
    print(f"[train-wrapper] Command: {' '.join(train_command)}", flush=True)

    process = subprocess.Popen(train_command, cwd=project_dir)
    process_holder["process"] = process
    return_code = process.wait()
    stop_event.set()
    watcher.join(timeout=10)

    _final_sync(
        experiment_name=args.experiment,
        run_uuid=args.run_uuid,
        output_dir=output_dir,
        artifact_path=args.artifact_path,
        state_path=state_path,
        stable_seconds=1,
        require_full_checkpoint=not args.allow_model_only,
    )

    sys.exit(return_code)


if __name__ == "__main__":
    main()

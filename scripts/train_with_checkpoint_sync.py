from __future__ import annotations

import argparse
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

if __name__ == "__main__" and not __package__:
    from checkpoint_contract import sha256_file
    from dagshub_checkpoint import checkpoint_manifest_artifact
else:
    from scripts.checkpoint_contract import sha256_file
    from scripts.dagshub_checkpoint import checkpoint_manifest_artifact


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

POST_STOP_TIMEOUT_SECONDS = 90 * 60
SYNC_TIMEOUT_SECONDS = 45 * 60
TRAIN_POLL_SECONDS = 5


class SyncTimeout(TimeoutError):
    pass


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
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError("MLflow experiment not found")
    return client, experiment.experiment_id


def _find_run(client, experiment_id: str, run_uuid: str):
    _validate_run_uuid(run_uuid)
    runs = client.search_runs(
        [experiment_id],
        filter_string=f"tags.run_uuid = '{run_uuid}'",
        max_results=2,
    )
    if len(runs) > 1:
        raise RuntimeError("Ambiguous MLflow run_uuid: multiple runs found")
    return runs[0] if runs else None


def _validate_run_uuid(run_uuid: str) -> None:
    if not re.fullmatch(r"(?:[0-9a-fA-F]{32}|[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})", run_uuid):
        raise ValueError("run_uuid must be a full hexadecimal UUID")


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


def _stable_checkpoint_snapshot(
    checkpoint_dir: Path,
    stable_seconds: int,
    require_full_checkpoint: bool,
) -> tuple[dict[str, tuple[int, int]] | None, str]:
    if not checkpoint_dir.is_dir():
        return None, "not a directory"

    if not (checkpoint_dir / "trainer_state.json").is_file():
        return None, "missing trainer_state.json"

    if not any((checkpoint_dir / name).is_file() for name in CHECKPOINT_FILE_CANDIDATES):
        return None, "missing model/adapter file"

    if require_full_checkpoint:
        missing = [name for name in FULL_CHECKPOINT_FILES if not (checkpoint_dir / name).is_file()]
        if missing:
            return None, f"missing full-checkpoint files: {', '.join(missing)}"

    before = _file_snapshot(checkpoint_dir)
    if not before:
        return None, "empty checkpoint"
    time.sleep(stable_seconds)
    after = _file_snapshot(checkpoint_dir)
    if before != after:
        return None, "checkpoint is still being written"

    return after, "ready"


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
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(data, stream, indent=2, ensure_ascii=False)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _sync_checkpoints_once(
    client,
    run_id: str,
    output_dir: Path,
    artifact_path: str,
    state_path: Path,
    stable_seconds: int,
    require_full_checkpoint: bool,
    final: bool = False,
) -> int:
    state = _load_state(state_path)
    identity = {"run_id": run_id, "artifact_path": artifact_path.strip("/"),
                "output_dir": str(output_dir.resolve())}
    if state.get("identity") != identity:
        # Old acknowledgements cannot establish that this run received a checkpoint.
        state["uploaded"] = {}
        state.pop("latest", None)
    state["identity"] = identity
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

        snapshot, reason = _stable_checkpoint_snapshot(
            checkpoint_dir,
            stable_seconds=stable_seconds,
            require_full_checkpoint=require_full_checkpoint,
        )
        if snapshot is None:
            if final:
                raise RuntimeError(f"Final checkpoint is not ready: {name} ({reason})")
            print(f"[checkpoint-sync] Skip {name}: {reason}", flush=True)
            continue

        target = f"{artifact_path}/{name}".strip("/")
        manifest_artifact = checkpoint_manifest_artifact(target)
        files = [{"path": f"{name}/{relative}", "size": size,
                  "sha256": sha256_file(checkpoint_dir / relative)}
                 for relative, (size, _) in sorted(snapshot.items())]
        if snapshot != _file_snapshot(checkpoint_dir):
            raise RuntimeError(f"Checkpoint changed while hashing: {name}")
        print(f"[checkpoint-sync] Uploading {name} -> {target}", flush=True)
        # Trainer owns the run lifecycle. The manifest is published only after the files.
        client.log_artifacts(run_id, str(checkpoint_dir), artifact_path=target)
        if snapshot != _file_snapshot(checkpoint_dir):
            raise RuntimeError(f"Checkpoint changed during upload: {name}")
        with tempfile.TemporaryDirectory() as scratch:
            manifest_file = Path(scratch) / Path(manifest_artifact).name
            manifest_file.write_text(json.dumps({
                "schema_version": 1, "run_id": run_id, "artifact_path": target,
                "file_count": len(files), "files": files,
            }, indent=2), encoding="utf-8")
            client.log_artifact(run_id, str(manifest_file), artifact_path=Path(manifest_artifact).parent.as_posix())
        client.set_tag(run_id, "checkpoint.synced.latest", name)
        client.set_tag(run_id, "checkpoint.synced.latest_step", str(_checkpoint_step(checkpoint_dir)))

        uploaded[name] = {
            "step": _checkpoint_step(checkpoint_dir),
            "artifact_path": target,
            "manifest_artifact": manifest_artifact,
            "uploaded_at": int(time.time()),
            "verification": "not_verified",
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
    failures: list[dict[str, str]] | None = None,
) -> None:
    failures = failures if failures is not None else []
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
                    print(f"[checkpoint-sync] Submitted {count} checkpoint(s); remote content not verified", flush=True)
            except Exception as exc:
                failures.append({"stage": "background_upload", "error_type": type(exc).__name__})
                print(f"[checkpoint-sync] Background upload failed ({type(exc).__name__}); will retry", flush=True)
            stop_event.wait(interval_seconds)
    except Exception as exc:
        failures.append({"stage": "background_setup", "error_type": type(exc).__name__})
        print(f"[checkpoint-sync] Background sync stopped ({type(exc).__name__})", flush=True)


def _final_sync(
    experiment_name: str,
    run_uuid: str,
    output_dir: Path,
    artifact_path: str,
    state_path: Path,
    stable_seconds: int,
    require_full_checkpoint: bool,
) -> None:
    if not output_dir.is_dir():
        raise FileNotFoundError("Training output directory missing during final checkpoint sync")
    client, experiment_id = _configure_mlflow(experiment_name)
    run = _find_run(client, experiment_id, run_uuid)
    if not run:
        raise RuntimeError("MLflow run not found during final checkpoint sync")
    count = _sync_checkpoints_once(
        client=client,
        run_id=run.info.run_id,
        output_dir=output_dir,
        artifact_path=artifact_path,
        state_path=state_path,
        stable_seconds=stable_seconds,
        require_full_checkpoint=require_full_checkpoint,
        final=True,
    )
    print(f"[checkpoint-sync] Final checkpoint pass submitted {count} checkpoint(s); "
          "not remote-verified. Full output upload/verification remains with dagshub_logger.", flush=True)


def _final_sync_with_retries(failures, outcome, **kwargs) -> None:
    for attempt in range(1, 3):
        outcome["attempts"] = attempt
        try:
            _final_sync(**kwargs)
            outcome["status"] = "submitted"
            return
        except Exception as exc:
            failures.append({"stage": "final_sync", "error_type": type(exc).__name__})
            print(f"[checkpoint-sync] Final checkpoint attempt {attempt}/2 failed ({type(exc).__name__})", flush=True)


def _join_sync(thread, deadline: float, stage: str) -> None:
    thread.join(max(0, deadline - time.monotonic()))
    if thread.is_alive():
        # No other thread may start uploading or write the receipt while this writer is blocked.
        raise SyncTimeout(f"[checkpoint-sync] Timed out waiting for {stage}; exiting with code 124.")


def _terminate_training(process, signum=signal.SIGTERM, grace_seconds=35) -> None:
    # torchrun's ranks can have process groups of their own outside the launcher's group.
    groups = {process.pid}
    pending = [process.pid]
    while pending:
        pid = pending.pop()
        try:
            pending.extend(int(child) for child in Path(f"/proc/{pid}/task/{pid}/children").read_text().split())
            groups.add(os.getpgid(pid))
        except (ProcessLookupError, FileNotFoundError):
            pass

    def send_to_groups(sig):
        for group in groups:
            try:
                os.killpg(group, sig)
            except ProcessLookupError:
                pass

    send_to_groups(signum)
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    send_to_groups(signal.SIGKILL)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print("[train-wrapper] Launcher still has not exited after SIGKILL.", flush=True)


def _wait_for_training(process, stop_marker, timeout_seconds, received_signal) -> int:
    deadline = None
    while True:
        if received_signal:
            _terminate_training(process, received_signal[0])
            return 128 + received_signal[0]
        if deadline is None and stop_marker is not None and stop_marker.is_file():
            deadline = time.monotonic() + timeout_seconds
            print(f"[train-wrapper] Stop step reached; giving Trainer {timeout_seconds}s to finish.", flush=True)
        remaining = deadline - time.monotonic() if deadline is not None else TRAIN_POLL_SECONDS
        if remaining <= 0:
            print("[train-wrapper] Trainer timed out after reaching STOP_AT_GLOBAL_STEP (exit 124).", flush=True)
            _terminate_training(process)
            return 124
        try:
            return process.wait(timeout=min(TRAIN_POLL_SECONDS, remaining))
        except subprocess.TimeoutExpired:
            continue


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
    parser.add_argument("--resume-best-checkpoint", default="")
    parser.add_argument("--resume-prepared-output", default="")
    parser.add_argument("--post-stop-timeout-seconds", type=int, default=POST_STOP_TIMEOUT_SECONDS,
                        help="Maximum wait for training and final evaluation after reaching the stop step")
    parser.add_argument("--sync-timeout-seconds", type=int, default=SYNC_TIMEOUT_SECONDS,
                        help="Total checkpoint sync shutdown and final-pass budget")
    parser.add_argument("train_overrides", nargs="*")
    args = parser.parse_args()
    if min(args.post_stop_timeout_seconds, args.sync_timeout_seconds) <= 0:
        parser.error("timeout values must be positive")
    return args


def _seed_resume_best(source: Path, output_dir: Path) -> None:
    """Keep the old best available for reload, retention, sync and the next resume."""
    from scripts.dagshub_checkpoint import _copy_resume_file
    if output_dir.exists():
        raise FileExistsError("Training output already exists; refusing to seed or overwrite it")
    snapshot, reason = _stable_checkpoint_snapshot(source, stable_seconds=0, require_full_checkpoint=True)
    if snapshot is None:
        raise RuntimeError(f"Resume best checkpoint is incomplete: {reason}")
    shutil.copytree(source, output_dir / source.name, copy_function=_copy_resume_file)
    print(f"[train-wrapper] Preserved parent best checkpoint: {source.name}", flush=True)


def main() -> None:
    args = parse_args()
    _validate_run_uuid(args.run_uuid)
    project_dir = args.project_dir.resolve()
    output_dir = _resolve_path(args.output_dir, project_dir)
    state_path = _resolve_path(args.state_path, project_dir)
    storage_plan = None
    if getattr(args, "resume_prepared_output", ""):
        from scripts.checkpoint_storage import activate_resume
        storage_plan = activate_resume(args.resume_prepared_output, output_dir,
                                       _resolve_path(args.yaml_config, project_dir), project_dir)
    elif getattr(args, "resume_best_checkpoint", ""):
        _seed_resume_best(Path(args.resume_best_checkpoint).resolve(), output_dir)
    stop_event = threading.Event()
    failures: list[dict[str, str]] = []
    received_signal = []

    def stop_handler(signum, _frame):
        stop_event.set()
        received_signal.append(signum)

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)

    sync_args = {
        "experiment_name": args.experiment, "run_uuid": args.run_uuid,
        "output_dir": output_dir, "artifact_path": args.artifact_path,
        "state_path": state_path, "require_full_checkpoint": not args.allow_model_only,
    }
    watcher = threading.Thread(
        target=_watch_checkpoints,
        kwargs={
            **sync_args,
            "interval_seconds": args.interval_seconds,
            "stable_seconds": args.stable_seconds,
            "stop_event": stop_event,
            "failures": failures,
        },
        daemon=True,
    )
    run_name = args.run_name if args.run_uuid in args.run_name else f"{args.run_name}-{args.run_uuid}"
    train_command = [
        "llamafactory-cli",
        "train",
        args.yaml_config,
        *args.train_overrides,
        f"run_name={run_name}",
    ]

    print(f"[train-wrapper] YAML_CONFIG: {args.yaml_config}", flush=True)
    print(f"[train-wrapper] OUTPUT_DIR: {output_dir}", flush=True)
    print(f"[train-wrapper] Checkpoint sync interval: {args.interval_seconds}s", flush=True)
    # Never echo arbitrary overrides or SDK exceptions: either may contain credentials.
    child_env = os.environ.copy()
    child_env.pop("CHECKPOINT_STORAGE_PLAN", None)
    child_env.pop("TRAIN_STOP_MARKER", None)
    if storage_plan:
        child_env["CHECKPOINT_STORAGE_PLAN"] = storage_plan
    tags = json.loads(child_env.get("MLFLOW_TAGS", "{}"))
    if not isinstance(tags, dict):
        raise ValueError("MLFLOW_TAGS must be a JSON object")
    tags.update({"run_uuid": args.run_uuid,
                 "experiment_variant": os.getenv("EXPERIMENT_VARIANT", "baseline")})
    for name in ("SPLIT_MANIFEST", "DATASET_DIR", "SPLIT_SEED", "SPLIT_ID"):
        if os.getenv(name):
            tags[name.lower()] = os.environ[name]
    child_env["RUN_UUID"] = args.run_uuid
    child_env["MLFLOW_TAGS"] = json.dumps(tags)

    process = None
    with tempfile.TemporaryDirectory(prefix="unimer-train-") as scratch:
        stop_marker = None
        if child_env.get("STOP_AT_GLOBAL_STEP"):
            stop_marker = Path(scratch) / "stop-step-reached"
            child_env["TRAIN_STOP_MARKER"] = str(stop_marker)
        try:
            process = subprocess.Popen(train_command, cwd=project_dir, env=child_env, start_new_session=True)
            watcher.start()
            return_code = _wait_for_training(process, stop_marker, args.post_stop_timeout_seconds, received_signal)
        except OSError as exc:
            print(f"[train-wrapper] Could not start/wait for training ({type(exc).__name__})", flush=True)
            return_code = 127
            if process is not None:
                _terminate_training(process)
        finally:
            stop_event.set()

    outcome = {"status": "failed", "verification": "not_verified", "attempts": 0}
    if process is not None:
        deadline = time.monotonic() + args.sync_timeout_seconds
        _join_sync(watcher, deadline, "background checkpoint sync")
        if time.monotonic() >= deadline:
            raise SyncTimeout("[checkpoint-sync] No time left for final checkpoint sync; exiting with code 124.")
        final_worker = threading.Thread(
            target=_final_sync_with_retries,
            args=(failures, outcome),
            kwargs={**sync_args, "stable_seconds": 1},
            daemon=True,
        )
        final_worker.start()
        _join_sync(final_worker, deadline, "final checkpoint sync")

    final_ok = outcome["status"] == "submitted"
    try:
        state = _load_state(state_path)
        state["sync_failures"] = failures
        state["final_sync"] = outcome
        _save_state(state_path, state)
    except Exception as exc:
        print(f"[checkpoint-sync] Could not persist sync outcome ({type(exc).__name__})", flush=True)
        final_ok = False
    if failures:
        print(f"[checkpoint-sync] Recorded {len(failures)} sync failure(s); "
              f"final checkpoint pass {'completed' if final_ok else 'failed'}", flush=True)
    sys.exit(return_code if return_code else (0 if final_ok else 1))


if __name__ == "__main__":
    try:
        main()
    except SyncTimeout as exc:
        print(str(exc), file=sys.stderr, flush=True)
        # An SDK executor can otherwise keep the interpreter alive during shutdown.
        os._exit(124)
    except Exception as exc:
        print(f"Training wrapper failed ({type(exc).__name__}); exception details suppressed.", file=sys.stderr)
        sys.exit(1)

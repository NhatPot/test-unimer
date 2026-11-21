from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

if not __package__:
    from checkpoint_contract import (
        BASE_MODEL, read_json_object, require_hex, require_regular_path, sha256_file, validate_checkpoint,
    )
else:
    from scripts.checkpoint_contract import (
        BASE_MODEL, read_json_object, require_hex, require_regular_path, sha256_file, validate_checkpoint,
    )


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

TRAINING_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def _verify_training_adapter(directory, run_id, weights_sha256, base_revision, artifact_path):
    manifest = read_json_object(directory / "artifact_manifest.json")
    files = manifest["files"]
    if manifest["file_count"] != len(files):
        raise ValueError("Adapter artifact manifest file count mismatch")
    prefix = ""
    if artifact_path != "training":
        if (manifest.get("schema_version"), manifest.get("run_id"), manifest.get("artifact_path")) != (
            1, run_id, artifact_path,
        ):
            raise ValueError("Adapter checkpoint manifest does not match the requested run and artifact")
        prefix = PurePosixPath(artifact_path).name + "/"
    expected_paths = {prefix + name for name in TRAINING_ADAPTER_FILES}
    entries = [entry for entry in files if entry["path"] in expected_paths]
    if len(entries) != len(expected_paths) or {entry["path"] for entry in entries} != expected_paths:
        raise ValueError("Artifact manifest must cover each root adapter file exactly once")
    adapter_files = {entry["path"][len(prefix):]: entry for entry in entries}
    if adapter_files["adapter_model.safetensors"]["sha256"] != weights_sha256:
        raise ValueError("Training adapter manifest does not match the pinned weights SHA256")
    return validate_checkpoint(
        directory / "adapter",
        {"files": adapter_files, "base_model": BASE_MODEL, "run_id": run_id,
         "artifact_path": artifact_path, "global_step": None, "base_revision": base_revision},
        mode="warm-start", base_revision=base_revision,
    )["checkpoint"]


def prepare_training_adapter(project_dir, output_dir, run_id, weights_sha256, base_revision,
                             artifact_path="training"):
    """Fetch and hash only the LoRA weights/config from the selected MLflow artifact."""
    require_hex(run_id, 32, "MLflow run ID")
    require_hex(weights_sha256, 64, "adapter weights SHA256")
    require_hex(base_revision, 40, "adapter base revision")
    artifact = PurePosixPath(artifact_path)
    if (artifact.is_absolute() or ".." in artifact.parts or artifact.as_posix() != artifact_path
            or (artifact_path != "training" and not re.fullmatch(r"checkpoint-[0-9]+", artifact.name))):
        raise ValueError("Adapter source must name training or a checkpoint directory")
    destination = _resolve_path(project_dir, output_dir) / run_id
    manifest_artifact = "training/artifact_manifest.json"
    if artifact_path != "training":
        # Keep the checkpoint cache separate from the root-adapter cache for the same run.
        destination = destination.with_name(f"{run_id}-checkpoints") / artifact
        manifest_artifact = checkpoint_manifest_artifact(artifact_path)
    destination = require_regular_path(destination)
    if destination.exists():
        return _verify_training_adapter(destination, run_id, weights_sha256, base_revision, artifact_path)

    mlflow = _configure_mlflow()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".adapter-", dir=destination.parent) as temp:
        staging = Path(temp) / "verified"
        adapter = staging / "adapter"
        adapter.mkdir(parents=True)
        sources = {staging / "artifact_manifest.json": manifest_artifact}
        sources.update({adapter / name: f"{artifact_path}/{name}" for name in TRAINING_ADAPTER_FILES})
        for local_path, remote_path in sources.items():
            downloaded = Path(mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=remote_path, dst_path=temp,
            ))
            shutil.copyfile(require_regular_path(downloaded), local_path)
        _verify_training_adapter(staging, run_id, weights_sha256, base_revision, artifact_path)
        if destination.exists():
            return _verify_training_adapter(destination, run_id, weights_sha256, base_revision, artifact_path)
        try:
            staging.rename(destination)
        except OSError:
            if not destination.is_dir():
                raise
            return _verify_training_adapter(destination, run_id, weights_sha256, base_revision, artifact_path)
    return str(destination / "adapter")


def checkpoint_manifest_artifact(artifact_path: str) -> str:
    checkpoint = PurePosixPath(artifact_path)
    parent = checkpoint.parent
    directory = "checkpoint_manifests" if parent.name in ("", "checkpoints") else f"{parent.name}_manifests"
    return (parent.parent / directory / f"{checkpoint.name}.json").as_posix()


def _manifest_exists(client, run_id: str, artifact_path: str) -> bool:
    path = PurePosixPath(artifact_path)
    if path.is_absolute() or '..' in path.parts or not path.name:
        raise ValueError("Manifest must name a relative artifact file")
    parent = None
    for index, _ in enumerate(path.parts, start=1):
        child = '/'.join(path.parts[:index])
        match = next((entry for entry in client.list_artifacts(run_id, parent) if entry.path == child), None)
        if match is None:
            return False
        if match.is_dir != (index < len(path.parts)):
            raise ValueError(f"Unexpected artifact type: {child}")
        parent = child
    return True


def _checkpoint_entries(client, run_id: str, artifact_path: str) -> dict[str, int]:
    files = {}
    pending = [artifact_path]
    while pending:
        parent = pending.pop()
        for entry in client.list_artifacts(run_id, parent):
            path = PurePosixPath(entry.path)
            if path.parent.as_posix() != parent or path.name == '..' or path.as_posix() != entry.path:
                raise ValueError("Invalid path in MLflow artifact listing")
            if entry.is_dir:
                pending.append(entry.path)
            else:
                if type(entry.file_size) is not int or entry.file_size < 0:
                    raise ValueError(f"MLflow did not report a valid size for {entry.path}")
                files[path.relative_to(artifact_path).as_posix()] = entry.file_size
    return files


def _verify_checkpoint_download(mlflow, run_id, artifact_path, local_checkpoint, output_path, legacy_manifest):
    client = mlflow.MlflowClient()
    specific_manifest = checkpoint_manifest_artifact(artifact_path)
    local_files = {}
    if local_checkpoint.is_symlink():
        raise ValueError("Downloaded checkpoint must not contain symlinks")
    for path in local_checkpoint.rglob("*"):
        if path.is_symlink():
            raise ValueError("Downloaded checkpoint must not contain symlinks")
        if path.is_file():
            local_files[path.relative_to(local_checkpoint).as_posix()] = path

    for manifest_artifact in (specific_manifest, legacy_manifest or "training/artifact_manifest.json"):
        if not _manifest_exists(client, run_id, manifest_artifact):
            continue
        manifest_dir = output_path / "manifest"
        manifest_dir.mkdir(exist_ok=True)
        manifest_file = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=manifest_artifact, dst_path=str(manifest_dir)
        )
        manifest = read_json_object(Path(manifest_file))
        if manifest_artifact == specific_manifest and (
            manifest.get("schema_version") != 1 or manifest.get("run_id") != run_id
            or manifest.get("artifact_path") != artifact_path
        ):
            raise ValueError("Checkpoint manifest does not match the requested run and artifact")

        prefix = PurePosixPath(artifact_path).name + "/"
        entries = {}
        for entry in manifest["files"]:
            if not entry["path"].startswith(prefix):
                if manifest_artifact == specific_manifest:
                    raise ValueError("Unexpected file in checkpoint manifest")
                continue
            relative = entry["path"][len(prefix):]
            path = PurePosixPath(relative)
            if path.is_absolute() or not path.name or '..' in path.parts or path.as_posix() != relative:
                raise ValueError("Invalid checkpoint path in artifact manifest")
            if relative in entries:
                raise ValueError(f"Duplicate checkpoint file in artifact manifest: {relative}")
            entries[relative] = entry

        if not entries:
            raise ValueError("Checkpoint is absent from the selected run's artifact manifest")
        if manifest_artifact == specific_manifest and manifest.get("file_count") != len(entries):
            raise ValueError("Checkpoint manifest file count does not match its entries")
        for name in sorted(local_files.keys() - entries.keys()):
            raise ValueError(f"Checkpoint file absent from manifest: {name}")
        for name, entry in entries.items():
            path = local_files.get(name)
            if path is None or path.stat().st_size != entry["size"]:
                raise ValueError(f"Checkpoint file missing or wrong size: {name}")
            if sha256_file(path) != entry["sha256"]:
                raise ValueError(f"Checkpoint SHA256 mismatch: {name}")
        print(f"Checkpoint bytes match {manifest_artifact} (SHA256).")
        return

    remote_sizes = _checkpoint_entries(client, run_id, artifact_path)
    local_sizes = {name: path.stat().st_size for name, path in local_files.items()}
    if remote_sizes != local_sizes:
        raise ValueError(f"Checkpoint files or sizes differ from the MLflow listing: {artifact_path}")
    print(f"WARNING: No SHA256 manifest for {artifact_path}; verified file names and sizes only "
          "against the MLflow listing (same-size corruption cannot be detected).", file=sys.stderr)


def _copy_resume_file(source, target):
    # Large immutable weights/states can share disk; relocated JSON must not.
    if Path(source).name != "trainer_state.json":
        try:
            os.link(source, target)
            return target
        except OSError:
            pass  # Different filesystem: use an ordinary copy.
    return shutil.copy2(source, target)


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


def _validate_checkpoint(checkpoint_dir: Path, require_full_checkpoint: bool,
                         *, world_size: int = 2, fp16: bool = False) -> None:
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    missing = [name for name in ("trainer_state.json",) if not (checkpoint_dir / name).is_file()]
    if not any((checkpoint_dir / name).is_file() for name in MODEL_FILE_CANDIDATES):
        missing.append("adapter_model.safetensors/model.safetensors")
    if any((checkpoint_dir / name).is_file() for name in MODEL_FILE_CANDIDATES[:2]):
        if not (checkpoint_dir / "adapter_config.json").is_file():
            missing.append("adapter_config.json")

    if require_full_checkpoint:
        if type(world_size) is not int or world_size < 1:
            raise ValueError("world_size must be a positive integer")
        missing.extend(name for name in FULL_CHECKPOINT_FILES if not (checkpoint_dir / name).is_file())
        rng_files = (["rng_state.pth"] if world_size == 1 else
                     [f"rng_state_{rank}.pth" for rank in range(world_size)])
        required = rng_files + (["scaler.pt"] if fp16 else [])
        missing.extend(name for name in required if not (checkpoint_dir / name).is_file())

    if missing:
        raise RuntimeError(
            "Checkpoint is not resume-ready. Missing: "
            + ", ".join(dict.fromkeys(missing))
            + f"\nCheckpoint: {checkpoint_dir}"
        )
    if any(path.stat().st_size == 0 for path in checkpoint_dir.iterdir() if path.is_file()):
        raise RuntimeError("Checkpoint contains an empty file; download must be checked")


def validate_resume_target(checkpoint, target):
    _validate_checkpoint(checkpoint, True, world_size=2, fp16=True)
    state = read_json_object(checkpoint / "trainer_state.json")
    epoch, step = state.get("epoch"), state.get("global_step")
    if (type(epoch) not in (int, float) or not math.isfinite(epoch) or epoch < 0
            or type(step) is not int or step < 1
            or type(target) not in (int, float) or not math.isfinite(target) or target <= epoch):
        raise ValueError("Resume needs valid saved step/epoch and a total target greater than saved epoch")
    return state


def prepare_resume_checkpoint(
    project_dir: str | Path,
    output_dir: str | Path,
    enabled: bool = False,
    run_id: str | None = None,
    run_uuid: str | None = None,
    experiment_name: str | None = None,
    artifact_path: str | None = None,
    require_full_checkpoint: bool = True,
    world_size: int = 2,
    fp16: bool = False,
    manifest_artifact: str | None = None,
    training_output_dir: str | Path | None = None,
    expected_run_tags: dict[str, str] | None = None,
) -> str | None:
    """Download a DagsHub MLflow checkpoint artifact for LLaMA-Factory resume."""
    print(f"DagsHub Resume: {'ON' if enabled else 'OFF'}")
    os.environ.pop("RESUME_CHECKPOINT", None)
    os.environ.pop("RESUME_BEST_CHECKPOINT", None)
    os.environ.pop("RESUME_PREPARED_OUTPUT", None)
    if not enabled:
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
    # A step number is not unique across runs; never reuse another run's cache.
    if not run_id or len(run_id) != 32 or any(c not in "0123456789abcdefABCDEF" for c in run_id):
        raise ValueError("run_id must be the full 32-character MLflow run ID")
    artifact_path = artifact_path.rstrip("/")
    source = PurePosixPath(artifact_path)
    if (not source.name.startswith("checkpoint-") or ".." in source.parts
            or source.is_absolute() or source.as_posix() != artifact_path):
        raise ValueError("artifact_path must select a checkpoint directory")
    if expected_run_tags:
        tags = mlflow.MlflowClient().get_run(run_id).data.tags
        for name, expected in expected_run_tags.items():
            if tags.get(name) != expected:
                raise ValueError(f"Resume source tag mismatch: {name}")
    output_path = _resolve_path(project_dir, output_dir) / run_id
    output_path.mkdir(parents=True, exist_ok=True)

    if training_output_dir is not None:
        from scripts.checkpoint_storage import verify_prepared
        training_output = _resolve_path(project_dir, training_output_dir).resolve()
        if training_output.exists():
            raise FileExistsError("Resume needs a new training output; existing output is preserved")
        receipt = output_path / "resume_relocation.json"
        if receipt.exists():
            plan = json.loads(receipt.read_text(encoding="utf-8"))
            if (plan.get("schema_version") != 2 or plan.get("activated")
                    or plan.get("source_run_id") != run_id or plan.get("source_artifact") != artifact_path
                    or plan.get("training_output_dir") != str(training_output)):
                raise ValueError("Existing resume cache belongs to another plan; preserve it and use a new run")
            verify_prepared(plan)
            prepared = Path(plan["prepared_output_dir"])
            os.environ["RESUME_PREPARED_OUTPUT"] = str(prepared)
            if plan["best_checkpoint_name"]:
                os.environ["RESUME_BEST_CHECKPOINT"] = str(prepared / plan["best_checkpoint_name"])
            checkpoint = str(prepared / plan["checkpoint_name"])
            os.environ["RESUME_CHECKPOINT"] = checkpoint
            return checkpoint

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

    _validate_checkpoint(local_checkpoint, require_full_checkpoint=require_full_checkpoint,
                         world_size=world_size, fp16=fp16)
    _verify_checkpoint_download(mlflow, run_id, artifact_path, local_checkpoint, output_path, manifest_artifact)
    if training_output_dir is not None:
        from scripts.checkpoint_storage import stage_resume
        state = json.loads((local_checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
        old_best = state.get("best_model_checkpoint")
        best_checkpoint = None
        if old_best:
            best_name = Path(old_best).name
            if not best_name.startswith("checkpoint-") or not best_name[11:].isdigit():
                raise ValueError("Invalid best checkpoint name in saved Trainer state")
            best_checkpoint = local_checkpoint
            if best_name != checkpoint_name:
                best_checkpoint = Path(prepare_resume_checkpoint(
                    project_dir=project_dir, output_dir=output_dir, enabled=True,
                    run_id=run_id, artifact_path=str(Path(artifact_path).parent / best_name),
                    require_full_checkpoint=True, world_size=world_size, fp16=fp16,
                    manifest_artifact=manifest_artifact,
                ))
        plan = stage_resume(local_checkpoint, best_checkpoint, output_path, training_output,
                            run_id=run_id, artifact_path=artifact_path)
        prepared = Path(plan["prepared_output_dir"])
        local_checkpoint = prepared / checkpoint_name
        os.environ["RESUME_PREPARED_OUTPUT"] = str(prepared)
        if plan["best_checkpoint_name"]:
            os.environ["RESUME_BEST_CHECKPOINT"] = str(prepared / plan["best_checkpoint_name"])
        print("Resume best/latest staged once; wrapper will move them into Trainer retention.")
    os.environ["RESUME_CHECKPOINT"] = str(local_checkpoint)

    print(f"Resume run_id: {run_id}")
    print(f"Resume artifact: {artifact_path}")
    print(f"Local checkpoint: {local_checkpoint}")
    return str(local_checkpoint)


def prepare_resume_checkpoint_via_python(
    python: str | Path,
    project_dir: str | Path,
    output_dir: str | Path,
    enabled: bool = False,
    run_id: str = "",
    run_uuid: str = "",
    experiment_name: str = "",
    artifact_path: str = "",
    state_path: str | Path = "/kaggle/working/resume_checkpoint.json",
    require_full_checkpoint: bool = True,
    world_size: int = 2,
    fp16: bool = False,
    manifest_artifact: str | None = None,
) -> str | None:
    """Run checkpoint download with the training Python environment."""
    print(f"DagsHub Resume: {'ON' if enabled else 'OFF'}")
    os.environ.pop("RESUME_CHECKPOINT", None)
    os.environ.pop("RESUME_BEST_CHECKPOINT", None)
    os.environ.pop("RESUME_PREPARED_OUTPUT", None)
    if not enabled:
        return None

    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.unlink(missing_ok=True)

    command = [
        str(python),
        "scripts/dagshub_checkpoint.py",
        "--project-dir",
        str(project_dir),
        "--output-dir",
        str(output_dir),
        "--run-id",
        run_id,
        "--run-uuid",
        run_uuid,
        "--experiment",
        experiment_name,
        "--artifact-path",
        artifact_path,
        "--state-path",
        str(state_path),
        "--world-size",
        str(world_size),
    ]
    if fp16:
        command.append("--fp16")
    if manifest_artifact:
        command.extend(["--manifest-artifact", manifest_artifact])
    if not require_full_checkpoint:
        command.append("--allow-model-only")

    subprocess.run(command, cwd=str(project_dir), check=True)
    checkpoint = json.loads(state_path.read_text(encoding="utf-8"))["checkpoint"]
    os.environ["RESUME_CHECKPOINT"] = checkpoint
    return checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a DagsHub checkpoint for resume training.")
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--run-uuid", default="")
    parser.add_argument("--experiment", default="")
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--state-path", required=True)
    parser.add_argument("--allow-model-only", action="store_true")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--manifest-artifact", help="Legacy run manifest path (default: training/artifact_manifest.json)")
    parser.add_argument("--training-output-dir", help="Relocate saved best checkpoint for load_best_model_at_end")
    parser.add_argument("--expected-run-tags", type=json.loads, help="Required source tags for full resume")
    parser.add_argument("--adapter-sha256", help="Download only the pinned adapter from the selected artifact")
    parser.add_argument("--adapter-base-revision", help="Pinned base revision in adapter-only mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.adapter_sha256 is not None:
        if (args.training_output_dir or args.run_uuid
                or args.expected_run_tags is not None or args.manifest_artifact):
            raise ValueError("Adapter-only mode requires a run ID and uses the selected artifact's manifest")
        adapter = prepare_training_adapter(args.project_dir, args.output_dir, args.run_id,
                                           args.adapter_sha256, args.adapter_base_revision, args.artifact_path)
        state_path = Path(args.state_path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({"adapter": adapter, "sha256": args.adapter_sha256}), encoding="utf-8")
        return
    if args.adapter_base_revision is not None:
        raise ValueError("--adapter-base-revision requires --adapter-sha256")
    if args.expected_run_tags is not None and (
            not isinstance(args.expected_run_tags, dict)
            or not all(isinstance(value, str) and value for value in args.expected_run_tags.values())):
        raise ValueError("Expected run tags must map names to nonempty values")
    checkpoint = prepare_resume_checkpoint(
        project_dir=args.project_dir,
        output_dir=args.output_dir,
        enabled=True,
        run_id=args.run_id or None,
        run_uuid=args.run_uuid or None,
        experiment_name=args.experiment or None,
        artifact_path=args.artifact_path,
        require_full_checkpoint=not args.allow_model_only,
        world_size=args.world_size,
        fp16=args.fp16,
        manifest_artifact=args.manifest_artifact,
        training_output_dir=args.training_output_dir,
        expected_run_tags=args.expected_run_tags,
    )

    state_path = Path(args.state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "checkpoint": checkpoint,
                "run_id": args.run_id,
                "run_uuid": args.run_uuid,
                "artifact_path": args.artifact_path,
                "best_checkpoint": os.environ.get("RESUME_BEST_CHECKPOINT", ""),
                "prepared_output": os.environ.get("RESUME_PREPARED_OUTPUT", ""),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

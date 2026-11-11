from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
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
) -> str | None:
    """Download a DagsHub MLflow checkpoint artifact for LLaMA-Factory resume."""
    print(f"DagsHub Resume: {'ON' if enabled else 'OFF'}")
    os.environ.pop("RESUME_CHECKPOINT", None)
    os.environ.pop("RESUME_BEST_CHECKPOINT", None)
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
    if not Path(artifact_path).name.startswith("checkpoint-") or ".." in Path(artifact_path).parts:
        raise ValueError("artifact_path must select a checkpoint directory")
    output_path = _resolve_path(project_dir, output_dir) / run_id
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

    _validate_checkpoint(local_checkpoint, require_full_checkpoint=require_full_checkpoint,
                         world_size=world_size, fp16=fp16)
    if manifest_artifact:
        manifest_file = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=manifest_artifact,
            dst_path=str(output_path / "manifest"),
        )
        manifest = json.loads(Path(manifest_file).read_text(encoding="utf-8"))
        entries = {item["path"]: item for item in manifest["files"]
                   if item["path"].startswith(f"{checkpoint_name}/")}
        if not entries:
            raise ValueError("Checkpoint is absent from the selected run's artifact manifest")
        for name, entry in entries.items():
            relative = Path(name).relative_to(checkpoint_name)
            if ".." in relative.parts:
                raise ValueError("Invalid checkpoint path in artifact manifest")
            path = local_checkpoint / relative
            if not path.is_file() or path.stat().st_size != entry["size"]:
                raise ValueError(f"Checkpoint file missing or wrong size: {relative}")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != entry["sha256"]:
                raise ValueError(f"Checkpoint SHA256 mismatch: {relative}")
        # Every state used for resume must be covered, not just the adapter.
        for path in local_checkpoint.iterdir():
            if path.is_file() and f"{checkpoint_name}/{path.name}" not in entries:
                raise ValueError(f"Checkpoint file absent from manifest: {path.name}")
        print("Checkpoint bytes match the selected run's artifact manifest (SHA256).")
    if training_output_dir is not None:
        training_output = _resolve_path(project_dir, training_output_dir).resolve()
        if training_output.exists():
            raise FileExistsError("Resume needs a new training output; existing output is preserved")
        state = json.loads((local_checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
        old_best = state.get("best_model_checkpoint")
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
            # Relocate only a working copy; never change the verified source checkpoint.
            relocated_state = dict(state, best_model_checkpoint=str(training_output / best_name))
            working = output_path / "relocated" / checkpoint_name
            if working.exists():
                if json.loads((working / "trainer_state.json").read_text()) != relocated_state:
                    raise ValueError("Existing working checkpoint differs; preserve it and use a new run")
            else:
                shutil.copytree(local_checkpoint, working, copy_function=_copy_resume_file)
                (working / "trainer_state.json").write_text(
                    json.dumps(relocated_state, indent=2, ensure_ascii=False), encoding="utf-8")
            (output_path / "resume_relocation.json").write_text(json.dumps({
                "source_checkpoint": str(local_checkpoint), "working_checkpoint": str(working),
                "old_best_model_checkpoint": old_best,
                "new_best_model_checkpoint": relocated_state["best_model_checkpoint"],
                "best_checkpoint_source": str(best_checkpoint),
                "global_step": state.get("global_step"), "epoch": state.get("epoch"),
            }, indent=2), encoding="utf-8")
            local_checkpoint = working
            os.environ["RESUME_BEST_CHECKPOINT"] = str(best_checkpoint)
            print(f"Best checkpoint {best_name} will be preserved in the new training output.")
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
    parser.add_argument("--manifest-artifact", help="Optional run artifact manifest for SHA256 verification")
    parser.add_argument("--training-output-dir", help="Relocate saved best checkpoint for load_best_model_at_end")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

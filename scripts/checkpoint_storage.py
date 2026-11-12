"""Keep resumed best/latest checkpoints under Trainer retention, without extra cache copies."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import yaml

from scripts.checkpoint_contract import sha256_file
from scripts.runtime_yaml import _write_runtime_yaml

RESERVE_BYTES = 1024 ** 3


def _write_json(path, value):
    _write_runtime_yaml(Path(path), json.dumps(value, indent=2, ensure_ascii=False) + "\n", overwrite=True)


def _files(directory):
    result = {}
    for path in sorted(Path(directory).rglob("*")):
        if path.is_symlink():
            raise ValueError("Prepared checkpoints must not contain symlinks")
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = {
                "size": path.stat().st_size, "sha256": sha256_file(path),
            }
    return result


def verify_prepared(plan):
    prepared = Path(plan["prepared_output_dir"])
    if prepared.is_symlink() or not prepared.is_dir() or _files(prepared) != plan["files"]:
        raise ValueError("Prepared resume files differ; preserve them and use a new run")


def stage_resume(source, best, cache_root, output_dir, *, run_id, artifact_path):
    """Move only this run's verified downloads; retain original Trainer JSON as evidence."""
    source, cache_root, output_dir = (Path(p).resolve() for p in (source, cache_root, output_dir))
    prepared = cache_root / "prepared-output"
    if output_dir.exists() or prepared.exists():
        raise FileExistsError("Training output or prepared resume already exists")
    paths = {source} | ({Path(best)} if best else set())
    for path in paths:
        if path.parent != cache_root or path.is_symlink() or not path.is_dir():
            raise ValueError("Only verified downloads directly inside this run cache can be staged")
    state_bytes = (source / "trainer_state.json").read_bytes()
    state = json.loads(state_bytes)
    if source.name != f"checkpoint-{state['global_step']}":
        raise ValueError("Checkpoint name differs from saved global step")
    backup = cache_root / "source_trainer_state.json"
    if backup.exists():
        raise FileExistsError("Source state evidence already exists; preserve it")
    backup.write_bytes(state_bytes)
    prepared.mkdir()
    for path in paths:
        path.rename(prepared / path.name)
    current = prepared / source.name
    if best:
        state["best_model_checkpoint"] = str(output_dir / Path(best).name)
        _write_json(current / "trainer_state.json", state)
    plan = {
        "schema_version": 2, "source_run_id": run_id, "source_artifact": artifact_path,
        "training_output_dir": str(output_dir), "prepared_output_dir": str(prepared),
        "checkpoint_name": source.name, "best_checkpoint_name": Path(best).name if best else None,
        "initial_global_step": state["global_step"], "initial_epoch": state.get("epoch"),
        "checkpoint_bytes": max(sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
                                for path in prepared.iterdir()),
        "files": _files(prepared), "activated": False,
    }
    _write_json(cache_root / "resume_relocation.json", plan)
    return plan


def require_space(output_dir, checkpoint_bytes, *, future_peak=False, keep=2):
    output_dir = Path(output_dir)
    existing = sum(p.is_dir() and p.name.startswith("checkpoint-") for p in output_dir.iterdir())
    copies = max(1, keep + 1 - existing) if future_peak else 1
    required = copies * checkpoint_bytes + RESERVE_BYTES
    free = shutil.disk_usage(output_dir).free
    if free < required:
        raise RuntimeError(
            f"Insufficient checkpoint disk space: {free / 1024**3:.2f} GiB free, "
            f"need {required / 1024**3:.2f} GiB including save-before-rotation and 1 GiB reserve. "
            "Best/latest checkpoints are preserved; no partial checkpoint was started."
        )
    print(f"[checkpoint-storage] {free / 1024**3:.2f} GiB free; "
          f"reserved {required / 1024**3:.2f} GiB for checkpoint saving.", flush=True)


def activate_resume(prepared, output_dir, config_path, project_dir):
    """Atomically publish the staged best/latest and rewrite only runtime checkpoint paths."""
    prepared, output_dir, config_path = map(lambda p: Path(p).resolve(),
                                         (prepared, output_dir, config_path))
    receipt = prepared.parent / "resume_relocation.json"
    plan = json.loads(receipt.read_text(encoding="utf-8"))
    if (plan.get("schema_version") != 2 or plan.get("activated")
            or plan["prepared_output_dir"] != str(prepared)
            or plan["training_output_dir"] != str(output_dir)):
        raise ValueError("Resume storage plan does not match this run")
    if output_dir.exists():
        raise FileExistsError("Existing training output is never replaced")
    verify_prepared(plan)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checkpoint = prepared / plan["checkpoint_name"]
    if (not isinstance(config, dict) or config_path.parent != output_dir.parent
            or (Path(project_dir) / config["output_dir"]).resolve() != output_dir
            or config.get("resume_from_checkpoint") != str(checkpoint)
            or config.get("adapter_name_or_path") != str(checkpoint)
            or config.get("create_new_adapter") is not False):
        raise ValueError("Runtime YAML must select the prepared full resume and this new output")
    backup = config_path.with_name(config_path.stem + ".before-storage.yaml")
    if backup.exists():
        raise FileExistsError("Runtime backup already exists; refusing to overwrite it")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if prepared.stat().st_dev != output_dir.parent.stat().st_dev:
        raise ValueError("Resume staging and output must share a filesystem; no implicit duplicate copy")
    keep = config.get("save_total_limit") or 2
    require_space(prepared, plan["checkpoint_bytes"], future_peak=True, keep=keep)
    backup.write_bytes(config_path.read_bytes())
    prepared.rename(output_dir)
    config.update(resume_from_checkpoint=str(output_dir / plan["checkpoint_name"]),
                  adapter_name_or_path=str(output_dir / plan["checkpoint_name"]))
    _write_runtime_yaml(config_path, yaml.safe_dump(config, sort_keys=False, allow_unicode=True), overwrite=True)
    plan["activated"] = True
    _write_json(receipt, plan)
    plan_path = output_dir / "checkpoint_storage.json"
    _write_json(plan_path, plan)
    shutil.copy2(receipt.parent / "source_trainer_state.json", output_dir / "resume_source_trainer_state.json")
    print("[checkpoint-storage] Best/latest are now inside Trainer output; no source weight cache remains.", flush=True)
    return str(plan_path)


def make_storage_callback(plan_path):
    from transformers import TrainerCallback

    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))

    class CheckpointStorageCallback(TrainerCallback):
        def _check(self, args, state, *, future_peak):
            if not state.is_world_process_zero:
                return
            if Path(args.output_dir).resolve() != Path(plan["training_output_dir"]):
                raise ValueError("Stale checkpoint storage plan belongs to another output")
            require_space(args.output_dir, plan["checkpoint_bytes"], future_peak=future_peak,
                          keep=args.save_total_limit or 2)

        def on_train_begin(self, args, state, control, **kwargs):
            self._check(args, state, future_peak=True)

        def on_step_end(self, args, state, control, **kwargs):
            if control.should_save:
                self._check(args, state, future_peak=False)

    return CheckpointStorageCallback()

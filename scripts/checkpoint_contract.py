"""CPU-only integrity and mode checks for the project's single LoRA adapter.

A pinned manifest is an externally reviewed input, not proof of authenticity.
This module never loads model tensors or deserializes pickle training state.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re


BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")
MODES = ("fresh", "warm-start", "full-resume")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hex(value: str, length: int, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{%d}" % length, value):
        raise ValueError(f"Invalid {label}")
    return value


def require_regular_path(path: Path) -> Path:
    path = path.absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Checkpoint paths must not contain symlinks")
    return path.resolve()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _read_bytes(path: Path, limit: int) -> bytes:
    require_regular_path(path)
    if not path.is_file():
        raise ValueError("Missing or oversized checkpoint JSON")
    with path.open("rb") as stream:
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise ValueError("Missing or oversized checkpoint JSON")
    return content


def _parse_object(content: bytes) -> dict:
    try:
        result = json.loads(content.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError):
        raise ValueError("Invalid checkpoint JSON; contents omitted") from None
    if not isinstance(result, dict):
        raise ValueError("Checkpoint JSON must be an object")
    return result


def read_json_object(path: Path, limit: int = 16 * 1024 * 1024) -> dict:
    return _parse_object(_read_bytes(path, limit))


def load_manifest(path: str | Path, *, expected_sha256: str, run_id: str,
                  artifact_path: str, base_model: str = BASE_MODEL) -> dict:
    """Require a separately pinned hash and explicit source, before any download."""
    require_hex(expected_sha256, 64, "manifest SHA256")
    require_hex(run_id, 32, "MLflow run ID")
    path = require_regular_path(Path(path))
    content = _read_bytes(path, 1024 * 1024)
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("Checkpoint manifest hash mismatch")
    manifest = _parse_object(content)
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("Unsupported checkpoint manifest schema")
    step = manifest.get("global_step")
    if type(step) is not int or step < 0:
        raise ValueError("Invalid manifest global_step")
    if not isinstance(artifact_path, str):
        raise ValueError("Artifact path must be a string")
    parts = PurePosixPath(artifact_path)
    if (parts.is_absolute()
            or str(parts) != artifact_path or ".." in parts.parts
            or parts.name != f"checkpoint-{step}"):
        raise ValueError("Artifact must be a canonical relative checkpoint path")
    for key, expected in (("run_id", run_id), ("artifact_path", artifact_path),
                          ("base_model", base_model)):
        if manifest.get(key) != expected:
            raise ValueError(f"Checkpoint manifest {key} mismatch")
    revision = manifest.get("base_revision")
    if revision is not None:
        require_hex(revision, 40, "base model revision")
    contract = manifest.get("training_contract_sha256")
    if contract is not None:
        require_hex(contract, 64, "training contract SHA256")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Manifest needs a nonempty files mapping")
    for name, entry in files.items():
        if (not isinstance(name, str) or name in (".", "..")
                or not re.fullmatch(r"[A-Za-z0-9_.-]+", name)):
            raise ValueError("Manifest only supports flat, safe filenames")
        if not isinstance(entry, dict) or type(entry.get("size")) is not int or entry["size"] <= 0:
            raise ValueError(f"Invalid manifest size: {name}")
        require_hex(entry.get("sha256"), 64, "file SHA256")
    if sum(name in files for name in ADAPTER_WEIGHTS) != 1:
        raise ValueError("Manifest must identify exactly one unsharded adapter weight file")
    if "adapter_config.json" not in files:
        raise ValueError("Manifest missing adapter_config.json")
    return manifest


def required_files(manifest: dict, *, mode: str, world_size: int = 2,
                   fp16: bool = True) -> list[str]:
    if mode not in ("warm-start", "full-resume"):
        raise ValueError("Checkpoint validation requires warm-start or full-resume")
    if type(world_size) is not int or world_size < 1 or type(fp16) is not bool:
        raise ValueError("Invalid world size or precision flag")
    names = ["adapter_config.json", *[name for name in ADAPTER_WEIGHTS if name in manifest["files"]]]
    if mode == "full-resume":
        if type(manifest.get("world_size")) is not int or manifest["world_size"] != world_size:
            raise ValueError("Full resume requires the original world size")
        names += ["trainer_state.json", "optimizer.pt", "scheduler.pt"]
        names += (["rng_state.pth"] if world_size == 1 else
                  [f"rng_state_{rank}.pth" for rank in range(world_size)])
        if fp16:
            names.append("scaler.pt")
    missing = sorted(set(names) - manifest["files"].keys())
    if missing:
        raise ValueError("Manifest missing required files: " + ", ".join(missing))
    return names


def validate_checkpoint(checkpoint_dir: str | Path, manifest: dict, *, mode: str,
                        world_size: int = 2, fp16: bool = True, lora_rank: int = 64,
                        lora_alpha: int = 16, base_revision: str | None = None,
                        training_contract_sha256: str | None = None) -> dict:
    directory = require_regular_path(Path(checkpoint_dir))
    if not directory.is_dir():
        raise ValueError("Checkpoint directory is missing")
    if base_revision is not None:
        require_hex(base_revision, 40, "expected base revision")
        if manifest.get("base_revision") != base_revision:
            raise ValueError("Base model revision mismatch or unrecorded")
    if mode == "full-resume":
        require_hex(training_contract_sha256, 64, "expected training contract SHA256")
        if manifest.get("training_contract_sha256") != training_contract_sha256:
            raise ValueError("Full resume training contract mismatch or unrecorded")
    names = required_files(manifest, mode=mode, world_size=world_size, fp16=fp16)
    for item in directory.iterdir():
        if item.is_symlink() or not item.is_file() or item.name not in manifest["files"]:
            raise ValueError("Checkpoint has unexpected files, directories or symlinks")
    for name in names:
        path = require_regular_path(directory / name)
        entry = manifest["files"][name]
        if not path.is_file() or path.stat().st_size != entry["size"]:
            raise ValueError(f"Missing file or size mismatch: {name}")
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Checkpoint file hash mismatch: {name}")
    config = read_json_object(directory / "adapter_config.json", 1024 * 1024)
    if config.get("peft_type") != "LORA" or config.get("base_model_name_or_path") != manifest["base_model"]:
        raise ValueError("Adapter type or base model mismatch")
    if (type(config.get("r")) is not int or config["r"] != lora_rank
            or config.get("lora_alpha") != lora_alpha):
        raise ValueError("Adapter rank or alpha mismatch")
    config_revision = config.get("revision")
    if config_revision is not None and config_revision != manifest.get("base_revision"):
        raise ValueError("Adapter revision conflicts with manifest")
    if mode == "full-resume":
        state = read_json_object(directory / "trainer_state.json")
        if type(state.get("global_step")) is not int or state["global_step"] != manifest["global_step"]:
            raise ValueError("Trainer step differs from manifest")
    return {
        "checkpoint": str(directory), "mode": mode, "run_id": manifest["run_id"],
        "artifact_path": manifest["artifact_path"], "global_step": manifest["global_step"],
        "base_model": manifest["base_model"], "base_revision": manifest.get("base_revision"),
        "verified_files": names, "integrity": "matches_supplied_manifest",
        "tensor_load_tested": False,
    }


def build_training_overrides(mode: str, *, max_steps: int, checkpoint: dict | None = None,
                             continuing_run_id: str | None = None) -> dict:
    """Compile mode settings after validate_checkpoint; does not authorize a job.

The caller must bind the returned continuing_run_id to the actual MLflow run,
apply overrides AND clear_keys, and validate the full resolved training config.
"""
    if mode not in MODES or type(max_steps) is not int or max_steps <= 0:
        raise ValueError("Choose an explicit mode and a positive total max_steps")
    overrides = {"max_steps": max_steps, "create_new_adapter": mode == "fresh"}
    clear_keys = ["resume_from_checkpoint", "adapter_name_or_path"]
    initial_step = 0
    if mode == "fresh":
        if checkpoint is not None or continuing_run_id is not None:
            raise ValueError("Fresh training must not receive a checkpoint or old run")
    else:
        if (not checkpoint or checkpoint.get("mode") != mode
                or checkpoint.get("integrity") != "matches_supplied_manifest"):
            raise ValueError("Mode requires checkpoint validation evidence")
        overrides["adapter_name_or_path"] = checkpoint["checkpoint"]
        clear_keys.remove("adapter_name_or_path")
        if mode == "warm-start":
            if continuing_run_id is not None:
                raise ValueError("Warm start requires a new run, not continuing_run_id")
        else:
            if continuing_run_id != checkpoint["run_id"]:
                raise ValueError("Full resume must continue exactly the source MLflow run")
            initial_step = checkpoint["global_step"]
            if max_steps <= initial_step:
                raise ValueError("Full resume max_steps must exceed the restored total step")
            overrides["resume_from_checkpoint"] = checkpoint["checkpoint"]
            clear_keys.remove("resume_from_checkpoint")
    return {"mode": mode, "initial_step": initial_step, "overrides": overrides,
            "clear_keys": clear_keys, "continuing_run_id": continuing_run_id}

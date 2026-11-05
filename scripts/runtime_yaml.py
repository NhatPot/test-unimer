"""
Runtime YAML configuration helper for Kaggle notebooks.

Giúp tách logic xử lý YAML ra khỏi notebook, giữ notebook ngắn gọn và dễ đọc.
"""

import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterable, Optional, Tuple

import yaml


# Explicit additions needed by bounded smoke runs and separate validation splits.
EXTRA_OVERRIDE_KEYS = frozenset({
    "max_steps", "seed", "data_seed", "eval_dataset", "adapter_name_or_path", "create_new_adapter",
})


def _write_runtime_yaml(path: Path, content: str, overwrite: bool) -> None:
    if path.is_symlink():
        raise ValueError("Runtime YAML must not be a symlink")
    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            return
        if not overwrite:
            raise FileExistsError("Runtime YAML exists with different content; use a new path")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                prefix=".runtime-", suffix=".tmp", delete=False) as file:
            temporary_path = Path(file.name)
            file.write(content)
        if overwrite:
            os.replace(temporary_path, path)
        else:
            # Atomic create without replacing a file another process just wrote.
            try:
                os.link(temporary_path, path)
            except FileExistsError:
                if path.is_symlink() or path.read_text(encoding="utf-8") != content:
                    raise FileExistsError("Runtime YAML appeared with different content") from None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def prepare_runtime_yaml(
    project_dir: str,
    base_yaml_config: str,
    runtime_yaml_config: str,
    use_override: bool = True,
    overrides: Optional[Dict[str, Any]] = None,
    strict_keys: bool = True,
    *,
    clear_keys: Optional[Iterable[str]] = None,
    profile: Optional[str] = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Chuẩn bị YAML config cho training, với tùy chọn override runtime.

    Args:
        project_dir: Đường dẫn thư mục project
        base_yaml_config: Đường dẫn tương đối đến YAML gốc (vd: "train/Uni-MuMER-train.yaml")
        runtime_yaml_config: Đường dẫn tương đối đến YAML runtime (vd: "train/runtime_Uni-MuMER-train.yaml")
        use_override: True = tạo runtime YAML với overrides, False = dùng base YAML
        overrides: Dict các key cần override. Key có value None sẽ bị bỏ qua.
        strict_keys: Chỉ chấp nhận khóa nền và EXTRA_OVERRIDE_KEYS khi True.
        clear_keys: Đặt rõ các khóa thành null; None trong overrides vẫn là giữ nguyên.
        profile: "t4" yêu cầu bf16=False và fp16=True; không tự đổi cấu hình.
        overwrite: Cho phép thay runtime khác nội dung; mặc định giữ file cũ.
        verbose: In trạng thái và tên khóa đã đổi, không in giá trị cấu hình.

    Returns:
        (yaml_config, output_dir, yaml_data):
            - yaml_config: đường dẫn đã truyền vào cho YAML được chọn
            - output_dir: giá trị output_dir từ YAML
            - yaml_data: dict chứa toàn bộ config YAML
    """
    base_yaml_path = Path(project_dir) / base_yaml_config
    runtime_yaml_path = Path(project_dir) / runtime_yaml_config

    if not base_yaml_path.is_file():
        raise FileNotFoundError(f"Base YAML not found: {base_yaml_path}")

    with base_yaml_path.open(encoding="utf-8") as file:
        yaml_data = yaml.safe_load(file)

    if not isinstance(yaml_data, dict):
        raise TypeError(f"YAML root must be a mapping, got {type(yaml_data)}: {base_yaml_path}")
    if any(not isinstance(key, str) for key in yaml_data):
        raise TypeError("Base YAML keys must be strings")
    if overrides is not None and not isinstance(overrides, dict):
        raise TypeError("Overrides must be a mapping")
    if isinstance(clear_keys, str):
        raise TypeError("clear_keys must be an iterable of keys, not a string")
    clear_keys = () if clear_keys is None else tuple(clear_keys)
    if profile not in (None, "t4"):
        raise ValueError("Unknown hardware profile")
    if not use_override and clear_keys:
        raise ValueError("Cannot clear keys when overrides are disabled")

    changes = {}

    if use_override:
        if (base_yaml_path.resolve() == runtime_yaml_path.resolve()
                or (runtime_yaml_path.exists() and base_yaml_path.samefile(runtime_yaml_path))):
            raise ValueError("Runtime YAML must not overwrite the base YAML")
        overrides = overrides or {}
        allowed_keys = set(yaml_data) | EXTRA_OVERRIDE_KEYS
        for key in (*overrides, *clear_keys):
            if not isinstance(key, str):
                raise TypeError("Override keys must be strings")
            if strict_keys and key not in allowed_keys:
                raise KeyError(f"Unknown override key: {key}")
        for key in clear_keys:
            if overrides.get(key) is not None:
                raise ValueError(f"Cannot both override and clear key: {key}")
        for key, new_value in overrides.items():
            if new_value is None:
                continue
            if key not in yaml_data or yaml_data[key] != new_value:
                changes[key] = True
            yaml_data[key] = new_value
        for key in clear_keys:
            if key not in yaml_data or yaml_data[key] is not None:
                changes[key] = True
            yaml_data[key] = None

    output_dir = yaml_data.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError("YAML config must define a non-empty string output_dir")
    if profile == "t4" and (yaml_data.get("bf16") is not False or yaml_data.get("fp16") is not True):
        raise ValueError("T4 profile requires bf16: false and fp16: true")

    yaml_config = runtime_yaml_config if use_override else base_yaml_config
    if use_override:
        content = yaml.safe_dump(yaml_data, sort_keys=False, allow_unicode=True)
        _write_runtime_yaml(runtime_yaml_path, content, overwrite)

    if verbose:
        print(f"Runtime YAML Override: {'ON' if use_override else 'OFF'}")
        print(f"Base YAML: {base_yaml_path}")
        print(f"Training YAML: {Path(project_dir) / yaml_config}")
        if use_override:
            print("Changed keys: " + (", ".join(changes) or "none"))

    return yaml_config, output_dir, yaml_data


def update_mlflow_tags(
    yaml_config: str,
    yaml_data: Dict[str, Any],
    use_override: bool,
    existing_tags: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Cập nhật MLflow tags với thông tin từ YAML config.

    Args:
        yaml_config: Đường dẫn YAML đang dùng
        yaml_data: Dict chứa config YAML
        use_override: True/False override mode
        existing_tags: Dict tags hiện tại (optional)

    Returns:
        Dict tags đã cập nhật
    """
    tags = existing_tags.copy() if existing_tags else {}

    tags.update({
        "dataset": str(yaml_data.get("dataset", "")),
        "yaml_config": yaml_config,
        "yaml_override": str(use_override).lower(),
    })

    return tags


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve runtime YAML without importing ML libraries.")
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--base-yaml", required=True)
    parser.add_argument("--runtime-yaml", required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--overrides-json")
    source.add_argument("--overrides-file")
    source.add_argument("--no-override", action="store_true")
    parser.add_argument("--clear", action="append", default=[], metavar="KEY")
    parser.add_argument("--profile", choices=["t4"])
    parser.add_argument("--overwrite-runtime", action="store_true")
    parser.add_argument("--summary", action="store_true", help="Include selected resolved training values")
    args = parser.parse_args(argv)
    try:
        project_dir = Path(args.project_dir).resolve()
        if args.overrides_file:
            source_text = (project_dir / args.overrides_file).read_text(encoding="utf-8")
        else:
            source_text = args.overrides_json if args.overrides_json is not None else "{}"
        overrides = json.loads(source_text)
        if not isinstance(overrides, dict):
            raise TypeError("Overrides JSON must be an object")
        with redirect_stdout(sys.stderr):
            yaml_config, output_dir, resolved = prepare_runtime_yaml(
                str(project_dir), args.base_yaml, args.runtime_yaml,
                use_override=not args.no_override, overrides=overrides,
                clear_keys=args.clear, profile=args.profile, overwrite=args.overwrite_runtime,
            )
        payload = {
            "yaml_config": str((project_dir / yaml_config).resolve()),
            "output_dir": str((project_dir / output_dir).resolve()),
            "profile": args.profile,
            "python": sys.executable,
        }
        if args.summary:
            keys = ("model_name_or_path", "dataset", "max_samples", "num_train_epochs",
                    "per_device_train_batch_size", "gradient_accumulation_steps", "learning_rate",
                    "lora_rank", "lora_alpha", "quantization_bit", "quantization_type",
                    "bf16", "fp16", "save_steps", "eval_steps", "output_dir",
                    "resume_from_checkpoint", "adapter_name_or_path")
            payload["summary"] = {key: resolved.get(key) for key in keys}
        print(json.dumps(payload))
    except (json.JSONDecodeError, yaml.YAMLError):
        print("Invalid YAML or overrides JSON; input contents omitted", file=sys.stderr)
        return 2
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"Runtime YAML error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

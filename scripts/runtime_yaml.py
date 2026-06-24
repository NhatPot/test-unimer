"""
Runtime YAML configuration helper for Kaggle notebooks.

Giúp tách logic xử lý YAML ra khỏi notebook, giữ notebook ngắn gọn và dễ đọc.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml


def prepare_runtime_yaml(
    project_dir: str,
    base_yaml_config: str,
    runtime_yaml_config: str,
    use_override: bool = True,
    overrides: Optional[Dict[str, Any]] = None,
    strict_keys: bool = True,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Chuẩn bị YAML config cho training, với tùy chọn override runtime.

    Args:
        project_dir: Đường dẫn thư mục project
        base_yaml_config: Đường dẫn tương đối đến YAML gốc (vd: "train/Uni-MuMER-train.yaml")
        runtime_yaml_config: Đường dẫn tương đối đến YAML runtime (vd: "train/runtime_Uni-MuMER-train.yaml")
        use_override: True = tạo runtime YAML với overrides, False = dùng base YAML
        overrides: Dict các key cần override. Key có value None sẽ bị bỏ qua.
        strict_keys: True = báo lỗi nếu override key không tồn tại trong base YAML

    Returns:
        (yaml_config, output_dir, yaml_data):
            - yaml_config: đường dẫn tương đối đến YAML dùng để train
            - output_dir: giá trị output_dir từ YAML
            - yaml_data: dict chứa toàn bộ config YAML
    """
    base_yaml_path = Path(project_dir) / base_yaml_config
    runtime_yaml_path = Path(project_dir) / runtime_yaml_config

    # Đọc YAML gốc
    if not base_yaml_path.exists():
        raise FileNotFoundError(f"Base YAML not found: {base_yaml_path}")

    with base_yaml_path.open(encoding="utf-8") as file:
        yaml_data = yaml.safe_load(file)

    if not isinstance(yaml_data, dict):
        raise TypeError(f"YAML root must be a mapping, got {type(yaml_data)}: {base_yaml_path}")

    changes = {}

    if use_override:
        # Áp dụng overrides
        if overrides:
            for key, new_value in overrides.items():
                # Bỏ qua key có value None
                if new_value is None:
                    continue

                # Kiểm tra key tồn tại nếu strict mode
                if strict_keys and key not in yaml_data:
                    raise KeyError(
                        f"Override key '{key}' not found in base YAML: {base_yaml_path}\n"
                        f"Available keys: {list(yaml_data.keys())}"
                    )

                old_value = yaml_data.get(key)
                yaml_data[key] = new_value

                # Ghi nhận thay đổi
                if old_value != new_value:
                    changes[key] = (old_value, new_value)

        # Ghi runtime YAML
        runtime_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with runtime_yaml_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(yaml_data, file, sort_keys=False, allow_unicode=True)

        yaml_config = runtime_yaml_config
        mode = "ON"
    else:
        # Không override: xóa runtime YAML nếu tồn tại
        runtime_yaml_path.unlink(missing_ok=True)
        yaml_config = base_yaml_config
        mode = "OFF"

    # Lấy output_dir
    if not yaml_data.get("output_dir"):
        raise KeyError(f"YAML config must define 'output_dir': {base_yaml_path}")
    output_dir = str(yaml_data["output_dir"])

    # In thông tin
    print(f"Runtime YAML Override: {mode}")
    print(f"YAML gốc: {base_yaml_path}")
    print(f"YAML dùng để train: {Path(project_dir) / yaml_config}")

    if use_override:
        if changes:
            print("Các key đã đổi:")
            for key, (old_value, new_value) in changes.items():
                print(f"  {key}: {old_value} -> {new_value}")
        else:
            print("Các key đã đổi: không có (giá trị override trùng YAML gốc)")

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

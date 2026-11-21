"""Build K3's CROHME-only train and validation files from the K1/B4 split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import tempfile

from scripts.checkpoint_contract import require_regular_path, sha256_file
from scripts.prepare_baseline_data import _json
from scripts.prepare_k2_data import (
    B4_SPLIT_MANIFEST, B4_SPLIT_SHA256, DATA_FILES, K1_SPLIT_SHA256, RECOGNITION,
    _counts, _filter_b4, _read_k1, read_b4_manifest, replay_quotas,
)


CROHME_SOURCES = (RECOGNITION, *(f"{RECOGNITION}_{task}" for task in ("can", "tree", "error_find", "error_fix")))
VAL_H_SOURCE = "parquet_hme100k_train"
VARIANT = "K3_crohme_ft"


def _reject_validation_overlap(train, k1_val, b4_manifest, b4_sha256):
    # VAL-C and VAL-H were extracted from these two sources in the pinned B4 val
    # split. Compare family-scoped IDs as well as RGB hashes and split groups.
    b4 = read_b4_manifest(b4_manifest, b4_sha256)
    protected = [record for record in b4["records"] if record["split"] == "val"
                 and record["source"] in (RECOGNITION, VAL_H_SOURCE)]
    hashes = {record["image_sha256"] for record in protected}
    groups = {record["group"] for record in protected}
    crohme_ids = {str(record["source_id"]) for record in protected
                  if record["source"] == RECOGNITION and record["source_id"] is not None}
    hashes.update(record["image_sha256"] for _, record, _ in k1_val)
    groups.update(record["group"] for _, record, _ in k1_val)
    crohme_ids.update(str(record["source_id"]) for _, record, _ in k1_val
                     if record["source"] in CROHME_SOURCES and record["source_id"] is not None)
    for row_id, record, _ in train:
        if (record["image_sha256"] in hashes or record["group"] in groups
                or (record["source_id"] is not None and str(record["source_id"]) in crohme_ids)):
            raise ValueError(f"K3 train row {row_id} overlaps validation (K1/B4, VAL-C or VAL-H)")


def prepare_k3_data(k1_dir, output_dir, *, expected_k1_sha256=K1_SPLIT_SHA256,
                    b4_manifest=B4_SPLIT_MANIFEST, expected_b4_sha256=B4_SPLIT_SHA256, seed=42):
    k1_dir = require_regular_path(Path(k1_dir))
    output_dir = require_regular_path(Path(output_dir))
    if output_dir.is_relative_to(k1_dir) or k1_dir.is_relative_to(output_dir):
        raise ValueError("K3 output must be separate from the K1 dataset")
    schema, k1_by_source, k1_val = _read_k1(k1_dir, expected_k1_sha256)
    by_source, b4_val, _ = _filter_b4(b4_manifest, expected_b4_sha256, k1_by_source, k1_val)
    if any(not by_source.get(source) for source in CROHME_SOURCES):
        raise ValueError("K3 requires all five CROHME training sources")
    counts = {source: len(by_source[source]) for source in CROHME_SOURCES}
    train_pool = [row for source in CROHME_SOURCES for row in by_source[source]]
    _reject_validation_overlap(train_pool, k1_val, b4_manifest, expected_b4_sha256)
    quotas = replay_quotas(counts, (len(train_pool) + 1) // 2)
    rng = random.Random(seed)
    train = [row for source in sorted(quotas) for row in rng.sample(by_source[source], quotas[source])]
    val = [row for row in b4_val if row[1]["source"] in CROHME_SOURCES]
    if not val:
        raise ValueError("K3 requires CROHME validation rows")
    images_col = schema["columns"]["images"]
    for name in {payload[images_col][0] for _, _, payload in train + val}:
        path = Path(name)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Missing K1 image used by K3: {path}")

    registry = {"k3_train": dict(schema, file_name="train.json"), "k3_val": dict(schema, file_name="val.json")}
    manifest = {
        "schema_version": 1, "variant": VARIANT, "seed": seed,
        "k1_split_sha256": expected_k1_sha256, "b4_split_sha256": expected_b4_sha256,
        "train_pool_by_source": counts, "counts": {"train": _counts(train), "val": _counts(val)},
        "train_row_ids": [row[0] for row in train], "val_row_ids": [row[0] for row in val],
    }
    manifest_name = "k3_manifest.json"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".k3-", dir=output_dir.parent) as temp:
        staging = Path(temp) / "dataset"
        staging.mkdir()
        _json(staging / "train.json", [row[2] for row in train])
        _json(staging / "val.json", [row[2] for row in val])
        _json(staging / "dataset_info.json", registry)
        manifest["files"] = {name: sha256_file(staging / name) for name in DATA_FILES}
        _json(staging / manifest_name, manifest)
        digest = sha256_file(staging / manifest_name)
        if not output_dir.exists():
            try:
                staging.rename(output_dir)
            except OSError:
                if not output_dir.is_dir():
                    raise
        if staging.exists():
            for name, expected in {**manifest["files"], manifest_name: digest}.items():
                path = output_dir / name
                if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
                    raise ValueError(f"Existing K3 dataset differs: {path}; use a new output directory")
    return {
        "dataset_dir": str(output_dir), "train_dataset": "k3_train", "eval_dataset": "k3_val",
        "manifest_path": str(output_dir / manifest_name), "manifest_sha256": digest,
        "counts": manifest["counts"], "initial_global_step": 0, "initial_epoch": 0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k1-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--k1-manifest-sha256", default=K1_SPLIT_SHA256)
    parser.add_argument("--b4-manifest", type=Path, default=B4_SPLIT_MANIFEST)
    parser.add_argument("--b4-manifest-sha256", default=B4_SPLIT_SHA256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    result = prepare_k3_data(args.k1_dir, args.output_dir, expected_k1_sha256=args.k1_manifest_sha256,
                             b4_manifest=args.b4_manifest, expected_b4_sha256=args.b4_manifest_sha256,
                             seed=args.seed)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

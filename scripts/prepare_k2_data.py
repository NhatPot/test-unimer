"""Build a reproducible K2 ShareGPT dataset from the pinned K1 split."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import shutil
import tempfile

from scripts.checkpoint_contract import read_json_object, require_hex, require_regular_path, sha256_file
from scripts.prepare_baseline_data import _image, _json


K1_SPLIT_SHA256 = "50291282d654b23db88b1bc1373ed0f8cce171081ece7d8b35d08e8d8309ef20"
RECOGNITION = "parquet_crohme_train"
ONPOLICY = "k2_onpolicy"
EDL_SOURCES = {f"{RECOGNITION}_{task}": task for task in ("error_find", "error_fix")}
EDL_COUNTS = {"error_find": 3904, "error_fix": 1318}
TASKS_BY_SOURCE = {**EDL_SOURCES, f"{RECOGNITION}_can": "counting", f"{RECOGNITION}_tree": "tree"}
DATA_FILES = ("train.json", "val.json", "dataset_info.json")
IMAGE_TAGS = {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"}


def replay_quotas(counts, size):
    """Use the largest remainders, breaking ties by source name."""
    total = sum(counts.values())
    if not 0 < size <= total:
        raise ValueError(f"Cannot sample {size} replay rows from {total} K1 train rows")
    shares = {source: divmod(size * count, total) for source, count in counts.items()}
    quotas = {source: share[0] for source, share in shares.items()}
    ranked = sorted(counts, key=lambda source: (-shares[source][1], source))
    for source in ranked[:size - sum(quotas.values())]:
        quotas[source] += 1
    return quotas


def _read_k1(directory, expected_sha256):
    manifest_path = directory / "split_manifest.json"
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("K1 split manifest SHA256 mismatch")
    manifest = json.loads(raw)
    if (manifest.get("schema_version") != 1
            or manifest.get("grouping") != "source_family_id_or_decoded_rgb_sha256"):
        raise ValueError("K2 requires the K1 image-grouped split")
    for name in DATA_FILES:
        if sha256_file(directory / name) != manifest["files"][name]:
            raise ValueError(f"K1 dataset file SHA256 mismatch: {name}")
    registry = read_json_object(directory / "dataset_info.json")
    schema = registry["baseline_train"]
    if (registry["baseline_val"] != dict(schema, file_name="val.json")
            or schema.get("file_name") != "train.json"
            or schema.get("formatting") != "sharegpt"
            or any(schema["tags"].get(tag) != value for tag, value in IMAGE_TAGS.items())):
        raise ValueError("K1 train and validation must use the same ShareGPT from/value schema")
    images_col = schema["columns"]["images"]
    image_dir = require_regular_path(directory / "images")
    rows = {
        split: json.loads((directory / f"{split}.json").read_bytes())
        for split in ("train", "val")
    }
    # K1 emits both data files in the order of manifest records filtered by split.
    positions = Counter()
    by_source = defaultdict(list)
    val_records = []
    row_ids = set()
    for record in manifest["records"]:
        split, source = record["split"], record["source"]
        row_id = f"{source}:{record['row']}"
        if row_id in row_ids or split not in rows or positions[split] >= len(rows[split]):
            raise ValueError(f"K1 split does not align with dataset rows: {row_id}")
        row_ids.add(row_id)
        payload = rows[split][positions[split]]
        positions[split] += 1
        digest = require_hex(record["image_sha256"], 64, "K1 image SHA256")
        path = image_dir / f"{digest}.png"
        if payload.get(images_col) != [str(path)]:
            raise ValueError(f"K1 image path differs from the split record: {row_id}")
        if split == "train":
            by_source[source].append((row_id, record, payload))
        else:
            val_records.append((row_id, record, payload))
    if any(positions[split] != len(data) for split, data in rows.items()):
        raise ValueError("K1 split counts do not match the dataset files")
    return schema, by_source, val_records


def _onpolicy_rows(package, expected_sha256, counts, directory, schema, recognition_rows, val_records):
    raw = gzip.decompress(package.read_bytes())
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("On-policy package SHA256 mismatch")
    data = json.loads(raw)
    if set(data) != set(counts) or any(len(data[task]) != count for task, count in counts.items()):
        raise ValueError("On-policy package has unexpected task counts")
    val_images = {record["image_sha256"] for _, record, _ in val_records}
    val_groups = {record["group"] for _, record, _ in val_records}
    train_groups = {
        (str(record["source_id"]), record["image_sha256"]): record["group"]
        for _, record, _ in recognition_rows
    }
    seen_ids, checked_images, prepared = set(), set(), {}
    messages_col, images_col = (schema["columns"][name] for name in ("messages", "images"))
    for task, rows in data.items():
        prepared[task] = []
        for row in rows:
            row_id, img_id, messages = (row[key] for key in ("id", "img_id", "messages"))
            digest = require_hex(row["image_sha256"], 64, "on-policy image SHA256")
            if (not isinstance(row_id, str) or not row_id or row_id in seen_ids
                    or type(img_id) not in (str, int) or not str(img_id)):
                raise ValueError(f"Invalid or duplicate on-policy row ID: {row_id!r}")
            seen_ids.add(row_id)
            group = train_groups.get((str(img_id), digest))
            if digest in val_images or (group is not None and group in val_groups):
                raise ValueError(f"On-policy row {row_id} overlaps K1 validation")
            if group is None:
                raise ValueError(f"On-policy row {row_id} has no matching K1 crohme_train image: {digest}")
            path = directory / "images" / f"{digest}.png"
            if digest not in checked_images:
                if not path.is_file() or path.is_symlink():
                    raise FileNotFoundError(f"Missing K1 image for on-policy row {row_id}: {path}")
                if _image(path)[0] != digest:
                    raise ValueError(f"K1 image RGB SHA256 mismatch for on-policy row {row_id}: {path}")
                checked_images.add(digest)
            if (not isinstance(messages, list) or len(messages) != 2
                    or [message.get("from") for message in messages] != ["human", "gpt"]
                    or any(not isinstance(message.get("value"), str) or not message["value"]
                           for message in messages)
                    or messages[0]["value"].count("<image>") != 1 or "<image>" in messages[1]["value"]):
                raise ValueError(f"Invalid ShareGPT messages for on-policy row {row_id}")
            prepared[task].append((
                row_id, {"source": ONPOLICY, "image_sha256": digest, "task": task},
                {messages_col: messages, images_col: [str(path)]},
            ))
    return prepared


def _counts(rows):
    sources = Counter(record["source"] for _, record, _ in rows)
    tasks = Counter(record.get("task", TASKS_BY_SOURCE.get(record["source"], "recognition"))
                    for _, record, _ in rows)
    return {"rows": len(rows), "by_source": dict(sources), "by_task": dict(tasks)}


def _signature(row, messages_col):
    _, record, payload = row
    return record["image_sha256"], json.dumps(payload[messages_col], sort_keys=True, ensure_ascii=False)


def _select_control(by_source, onpolicy, quotas, replay_ids, rng, messages_col):
    selected, from_replay = [], {}
    onpolicy_signatures = {_signature(row, messages_col) for rows in onpolicy.values() for row in rows}
    for source, task in EDL_SOURCES.items():
        outside, inside = [], []
        for row in by_source[source]:
            if _signature(row, messages_col) not in onpolicy_signatures:
                (inside if row[0] in replay_ids else outside).append(row)
        count = quotas[task]
        if len(outside) + len(inside) < count:
            raise ValueError(f"Not enough distinct-from-on-policy K1 {task} rows for the control")
        primary = min(count, len(outside))
        from_replay[task] = count - primary
        selected.extend(rng.sample(outside, primary))
        selected.extend(rng.sample(inside, from_replay[task]))
    return selected, from_replay


def prepare_k2_data(k1_dir, package, output_dir, variant, *, expected_onpolicy_sha256,
                    expected_k1_sha256=K1_SPLIT_SHA256, replay_size=30_000,
                    edl_counts=EDL_COUNTS, seed=42):
    if variant not in ("onpolicy", "control"):
        raise ValueError("Choose K2 variant onpolicy or control")
    k1_dir = require_regular_path(Path(k1_dir))
    output_dir = require_regular_path(Path(output_dir))
    if output_dir.is_relative_to(k1_dir) or k1_dir.is_relative_to(output_dir):
        raise ValueError("K2 output must be separate from the K1 dataset")
    schema, by_source, val = _read_k1(k1_dir, expected_k1_sha256)
    if expected_k1_sha256 == K1_SPLIT_SHA256 and (sum(map(len, by_source.values())), len(val)) != (64_843, 7_157):
        raise ValueError("Unexpected K1 train or validation count")
    onpolicy = _onpolicy_rows(Path(package), expected_onpolicy_sha256, edl_counts, k1_dir,
                             schema, by_source[RECOGNITION], val)
    quotas = replay_quotas({source: len(rows) for source, rows in by_source.items()}, replay_size)
    rng = random.Random(seed)
    replay = [row for source in sorted(quotas) for row in rng.sample(by_source[source], quotas[source])]
    if variant == "control":
        edl, from_replay = _select_control(by_source, onpolicy, edl_counts, {row[0] for row in replay},
                                          rng, schema["columns"]["messages"])
    else:
        edl = [row for task in edl_counts for row in onpolicy[task]]
        from_replay = dict.fromkeys(edl_counts, 0)
    train = replay + edl
    images_col = schema["columns"]["images"]
    for name in {payload[images_col][0] for _, _, payload in train + val}:
        path = Path(name)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Missing K1 image used by K2: {path}")

    registry = {"k2_train": dict(schema, file_name="train.json"), "k2_val": dict(schema, file_name="val.json")}
    counts = {"train": _counts(train), "val": _counts(val), "replay": _counts(replay), "edl": _counts(edl)}
    manifest = {
        "schema_version": 1, "variant": f"K2_{variant}_sft", "seed": seed,
        "k1_split_sha256": expected_k1_sha256, "onpolicy_package_sha256": expected_onpolicy_sha256,
        "counts": counts, "control_edl_rows_reused_from_replay": from_replay,
        "replay_row_ids": [row[0] for row in replay],
        "edl_row_ids": [row[0] for row in edl],
        "val_row_ids": [row[0] for row in val],
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".k2-", dir=output_dir.parent) as temp:
        staging = Path(temp) / "dataset"
        staging.mkdir()
        _json(staging / "train.json", [row[2] for row in train])
        shutil.copyfile(k1_dir / "val.json", staging / "val.json")
        _json(staging / "dataset_info.json", registry)
        manifest["files"] = {name: sha256_file(staging / name) for name in DATA_FILES}
        manifest_name = "k2_manifest.json"
        _json(staging / manifest_name, manifest)
        digest = sha256_file(staging / manifest_name)
        if output_dir.exists():
            for name, expected in {**manifest["files"], manifest_name: digest}.items():
                path = output_dir / name
                if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
                    raise ValueError(f"Existing K2 dataset differs: {path}; use a new output directory")
        else:
            staging.rename(output_dir)
    return {
        "dataset_dir": str(output_dir), "train_dataset": "k2_train", "eval_dataset": "k2_val",
        "manifest_path": str(output_dir / manifest_name), "manifest_sha256": digest, "counts": counts,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k1-dir", required=True)
    parser.add_argument("--onpolicy-package", type=Path, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", required=True, choices=("onpolicy", "control"))
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--resume-data-sha256")
    parser.add_argument("--total-epochs", type=int, default=1)
    args = parser.parse_args(argv)
    state = {"global_step": 0, "epoch": 0}
    if args.resume_checkpoint is not None:
        from scripts.dagshub_checkpoint import validate_resume_target

        require_hex(args.resume_data_sha256, 64, "resume K2 data SHA256")
        state = validate_resume_target(args.resume_checkpoint, args.total_epochs)
    elif args.resume_data_sha256 is not None:
        parser.error("--resume-data-sha256 requires --resume-checkpoint")
    checksum_path = args.onpolicy_package.with_suffix("").with_suffix(".sha256")
    checksum = require_hex(checksum_path.read_text().strip(), 64, "on-policy package checksum")
    result = prepare_k2_data(args.k1_dir, args.onpolicy_package, args.output_dir, args.variant,
                             expected_onpolicy_sha256=checksum)
    if args.resume_checkpoint is not None and result["manifest_sha256"] != args.resume_data_sha256:
        raise ValueError("Rebuilt K2 data SHA256 differs from the resumed run")
    result.update(initial_global_step=state["global_step"], initial_epoch=state["epoch"])
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

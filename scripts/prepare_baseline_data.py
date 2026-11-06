"""Prepare a small, reusable image-grouped train/validation dataset on Kaggle.

Network access is only used by the CLI on a cache miss. Tests supply tiny rows.
The source repository/registry is never rewritten; generated files live outside it.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, closing
import hashlib
import io
from itertools import islice
import json
import math
from pathlib import Path
import random
import tempfile

from scripts.checkpoint_contract import require_regular_path, sha256_file


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _image(value):
    from PIL import Image
    if isinstance(value, dict):
        value = io.BytesIO(value["bytes"]) if value.get("bytes") else value.get("path")
    elif isinstance(value, bytes):
        value = io.BytesIO(value)
    if isinstance(value, Image.Image):
        image = value.convert("RGB")
    else:
        with Image.open(value) as opened:
            image = opened.convert("RGB")
    image.load()
    key = hashlib.sha256(f"RGB:{image.width}:{image.height}:".encode() + image.tobytes()).hexdigest()
    return key, image


def _settings(names, max_samples, val_ratio, seed):
    if type(max_samples) is not int or max_samples < 2:
        raise ValueError("max_samples must be at least 2 per source")
    if type(seed) is not int or not isinstance(names, list) or not names or len(set(names)) != len(names):
        raise ValueError("Provide distinct dataset names and an integer seed")
    if type(val_ratio) not in (int, float) or not math.isfinite(val_ratio) or not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    return dict(datasets=names, max_samples=max_samples, val_ratio=val_ratio, seed=seed)


def _result(directory, manifest):
    return dict(dataset_dir=str(directory), train_dataset="baseline_train", eval_dataset="baseline_val",
                manifest_path=str(directory / "split_manifest.json"),
                manifest_sha256=sha256_file(directory / "split_manifest.json"), counts=manifest["counts"])


def reuse_prepared(directory, settings):
    directory = require_regular_path(Path(directory))
    if not directory.exists():
        return None
    manifest = json.loads((directory / "split_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("settings") != settings:
        raise ValueError("Prepared subset differs; choose a new data directory instead of overwriting")
    for name, digest in manifest["files"].items():
        path = require_regular_path(directory / name)
        if not path.is_relative_to(directory) or not path.is_file() or sha256_file(path) != digest:
            raise ValueError("Prepared dataset file changed or is missing")
    return _result(directory, manifest)


def prepare_from_rows(directory, rows_by_source, entries, *, max_samples=1000,
                      val_ratio=0.1, seed=42, revisions=None):
    """Read at most max_samples rows per source, then split whole image groups.

    Same decoded RGB pixels are grouped even if encoded bytes differ. Matching
    source IDs within the same dataset family also join groups. This is not a
    perceptual detector for cropped/altered duplicates; source lineage matters.
    """
    names = list(rows_by_source)
    settings = _settings(names, max_samples, val_ratio, seed)
    settings["source_entries"] = entries
    directory = require_regular_path(Path(directory))
    cached = reuse_prepared(directory, settings)
    if cached:
        return cached
    directory.parent.mkdir(parents=True, exist_ok=True)
    lock = directory.with_name(directory.name + ".building")
    lock.mkdir()  # A stale build requires inspection, never replacement.
    try:
        staging = Path(tempfile.mkdtemp(prefix=directory.name + ".partial-", dir=directory.parent))
        (staging / "images").mkdir()
        records, parents, source_ids = [], {}, {}

        def find(key):
            parents.setdefault(key, key)
            if parents[key] != key:
                parents[key] = find(parents[key])
            return parents[key]

        def join(a, b):
            a, b = find(a), find(b)
            parents[max(a, b)] = min(a, b)

        template = None
        for name, rows in rows_by_source.items():
            entry = entries[name]
            schema = {k: entry[k] for k in ("formatting", "columns", "tags") if k in entry}
            if template is None:
                template = schema
            if schema != template or schema.get("formatting") != "sharegpt":
                raise ValueError("Baseline sources must share the ShareGPT image/message schema")
            image_column = entry["columns"]["images"]
            messages_column = entry["columns"]["messages"]
            selected = 0
            for index, row in enumerate(islice(rows, max_samples)):
                messages = row.get(messages_column)
                if not isinstance(messages, list) or not messages:
                    raise ValueError(f"Missing conversations in source {name}, row {index}")
                key, image = _image(row.get(image_column))
                find(key)
                image_file = staging / "images" / f"{key}.png"
                if not image_file.exists():
                    image.save(image_file, format="PNG")
                source_id = row.get("id")
                family = entry.get("subset", name)
                family = "crohme_train" if family.startswith("crohme_train") else family
                if source_id is not None:
                    identity = (entry.get("hf_hub_url", "local"), family, str(source_id))
                    if identity in source_ids:
                        join(key, source_ids[identity])
                    source_ids[identity] = key
                payload = {messages_column: messages, image_column: [str(directory / "images" / f"{key}.png")]}
                records.append(dict(source=name, row=index, source_id=source_id,
                                    image_sha256=key, payload=payload))
                selected += 1
            if not selected:
                raise ValueError(f"Empty source: {name}")
        groups = sorted({find(item["image_sha256"]) for item in records})
        if len(groups) < 2:
            raise ValueError("Need at least two independent image groups")
        random.Random(seed).shuffle(groups)
        n_val = min(len(groups) - 1, max(1, round(len(groups) * val_ratio)))
        val_groups = set(groups[:n_val])
        train, val, membership = [], [], []
        counts = {name: Counter() for name in names}
        for item in records:
            group = find(item["image_sha256"])
            split = "val" if group in val_groups else "train"
            (val if split == "val" else train).append(item.pop("payload"))
            counts[item["source"]][split] += 1
            membership.append(dict(item, group=group, split=split))
        registry = {"baseline_train": dict(template, file_name="train.json"),
                    "baseline_val": dict(template, file_name="val.json")}
        _json(staging / "train.json", train)
        _json(staging / "val.json", val)
        _json(staging / "dataset_info.json", registry)
        manifest = dict(schema_version=1, settings=settings, source_revisions=revisions or {},
                        grouping="source_family_id_or_decoded_rgb_sha256",
                        counts=dict(train_rows=len(train), val_rows=len(val),
                                    unique_images=len({r["image_sha256"] for r in membership}),
                                    train_groups=len(groups)-n_val, val_groups=n_val,
                                    image_group_overlap=0, by_source={k:dict(v) for k,v in counts.items()}),
                        records=membership,
                        files={p.relative_to(staging).as_posix(): sha256_file(p)
                               for p in sorted(staging.rglob("*")) if p.is_file()})
        _json(staging / "split_manifest.json", manifest)
        if directory.exists():
            raise FileExistsError("Data destination appeared during preparation")
        staging.rename(directory)
        return _result(directory, manifest)
    finally:
        lock.rmdir()  # Failed partial builds are preserved for diagnosis.


def prepare_baseline(project_dir, datasets, output_dir, *, max_samples=1000, val_ratio=0.1, seed=42):
    names = [name.strip() for name in datasets.split(",") if name.strip()]
    settings = _settings(names, max_samples, val_ratio, seed)
    registry = json.loads((Path(project_dir) / "train/dataset_info.json").read_text(encoding="utf-8"))
    entries = {name: registry[name] for name in names}
    settings["source_entries"] = entries
    output_dir = require_regular_path(Path(output_dir))
    cached = reuse_prepared(output_dir, settings)
    if cached:
        return cached
    from datasets import load_dataset_builder
    from huggingface_hub import HfApi
    revisions = {repo: HfApi().dataset_info(repo).sha for repo in {e["hf_hub_url"] for e in entries.values()}}
    with ExitStack() as readers:
        rows = {}
        for name, entry in entries.items():
            builder = load_dataset_builder(entry["hf_hub_url"], entry["subset"],
                                           revision=revisions[entry["hf_hub_url"]])
            if builder.info.builder_name != "parquet":
                raise ValueError("Baseline preparation requires Parquet sources")
            files = builder.config.data_files[entry.get("split", "train")]
            columns = [entry["columns"]["messages"], entry["columns"]["images"]]
            rows[name] = readers.enter_context(closing(_parquet_rows(files, columns, max_samples)))
        return prepare_from_rows(output_dir, rows, entries, max_samples=max_samples,
                                 val_ratio=val_ratio, seed=seed, revisions=revisions)


def _parquet_rows(files, columns, limit):
    """Avoid Arrow Dataset fragment readers surviving an early streaming stop."""
    import fsspec
    import pyarrow.parquet as pq

    remaining = limit
    for name in files:
        with fsspec.open(str(name), "rb") as stream, closing(pq.ParquetFile(stream)) as parquet:
            missing = set(columns) - set(parquet.schema_arrow.names)
            if missing:
                raise ValueError("Parquet source is missing required image/message columns")
            selected = columns + (["id"] if "id" in parquet.schema_arrow.names else [])
            for batch in parquet.iter_batches(batch_size=min(32, limit), columns=selected, use_threads=False):
                for row in batch.to_pylist():
                    yield row
                    remaining -= 1
                    if not remaining:
                        return


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--datasets")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-yaml")
    parser.add_argument("--runtime-yaml")
    parser.add_argument("--overrides-json")
    args = parser.parse_args(argv)
    overrides = json.loads(args.overrides_json) if args.overrides_json else None
    if overrides is not None:
        if not isinstance(overrides, dict) or not args.base_yaml or not args.runtime_yaml:
            raise ValueError("Runtime preparation requires overrides, base YAML and runtime YAML")
        output = Path(args.project_dir) / overrides["output_dir"]
        if output.exists():
            raise ValueError("Training output exists; use a new run to preserve it")
    names = overrides["dataset"] if overrides is not None else args.datasets
    if not names:
        raise ValueError("Dataset names are required")
    result = prepare_baseline(args.project_dir, names, args.output_dir,
                              max_samples=overrides.get("max_samples", 1000) if overrides else args.max_samples,
                              val_ratio=overrides.get("val_size", .1) if overrides else args.val_ratio,
                              seed=overrides.get("seed", 42) if overrides else args.seed)
    if overrides is not None:
        from scripts.runtime_yaml import prepare_runtime_yaml
        # max_samples/val_size are input subset controls, not a second loader split/cap.
        overrides.update(dataset_dir=result["dataset_dir"], dataset=result["train_dataset"],
                         eval_dataset=result["eval_dataset"], val_size=0, max_samples=None,
                         resume_from_checkpoint=None, adapter_name_or_path=None, max_steps=None,
                         create_new_adapter=True)
        import contextlib
        import sys
        with contextlib.redirect_stdout(sys.stderr):
            config, output, resolved = prepare_runtime_yaml(
                args.project_dir, args.base_yaml, args.runtime_yaml, overrides=overrides,
                clear_keys=["max_samples", "max_steps", "resume_from_checkpoint", "adapter_name_or_path"],
                profile="t4", overwrite=True)
        summary_keys = ("model_name_or_path", "dataset", "eval_dataset", "dataset_dir", "val_size",
                        "num_train_epochs", "max_samples", "max_steps", "seed",
                        "per_device_train_batch_size", "gradient_accumulation_steps", "learning_rate",
                        "lora_rank", "lora_alpha", "quantization_bit", "bf16", "fp16",
                        "save_steps", "eval_steps", "load_best_model_at_end", "resume_from_checkpoint")
        result.update(yaml_config=str((Path(args.project_dir) / config).resolve()),
                      output_dir=str((Path(args.project_dir) / output).resolve()),
                      summary={k:resolved.get(k) for k in summary_keys})
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

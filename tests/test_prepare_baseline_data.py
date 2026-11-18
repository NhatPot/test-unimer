import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest import mock
from collections import Counter
from contextlib import closing

from PIL import Image

from scripts.prepare_baseline_data import prepare_baseline, prepare_from_rows, reuse_prepared, _settings, _parquet_rows


def image_bytes(index, format="PNG"):
    stream = io.BytesIO()
    Image.new("RGB", (4, 4), (index, 20, 30)).save(stream, format=format)
    return stream.getvalue()


def rows(n=20, offset=0):
    return [{"id": str(i), "image": {"bytes": image_bytes(i+offset)},
             "conversations": [{"from": "human", "value": "<image>Read"},
                               {"from": "gpt", "value": str(i)}]} for i in range(n)]


def entry(subset):
    return {"hf_hub_url": "fixture/repo", "subset": subset, "formatting": "sharegpt",
            "columns": {"messages": "conversations", "images": "image"},
            "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"}}


class BaselineDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "data"
        self.sources = {"hmer": rows(), "count": rows()}
        self.entries = {"hmer": entry("crohme_train"), "count": entry("crohme_train_can")}

    def prepare(self, **kwargs):
        return prepare_from_rows(self.output, self.sources, self.entries,
                                 revisions={"fixture/repo": "fixture-only"}, **kwargs)

    def manifest(self):
        return json.loads((self.output / "split_manifest.json").read_text())

    def test_exact_same_images_across_tasks_stay_together(self):
        result = self.prepare()
        manifest = self.manifest()
        membership = {}
        for item in manifest["records"]:
            membership.setdefault(item["image_sha256"], set()).add(item["split"])
        self.assertTrue(all(len(splits) == 1 for splits in membership.values()))
        self.assertEqual(result["counts"]["val_groups"], 2)
        self.assertEqual(result["counts"]["train_groups"], 18)
        self.assertEqual(result["counts"]["train_rows"], 36)
        self.assertEqual(result["counts"]["val_rows"], 4)

    def test_cap_applies_per_source_before_split_only(self):
        self.prepare(max_samples=10)
        self.assertEqual(len(self.manifest()["records"]), 20)
        for counts in self.manifest()["counts"]["by_source"].values():
            self.assertEqual(sum(counts.values()), 10)

    def test_scalar_cap_manifest_matches_pre_k1_bytes(self):
        self.sources = {"hmer": rows(6), "count": rows(6)}
        self.prepare(max_samples=4)
        raw = (self.output / "split_manifest.json").read_bytes()
        files = json.loads(raw)["files"]
        # Captured at f9af2bf with this fixture. Only train/val hashes depend
        # on the temporary absolute path, so verify those payloads separately.
        for split, indices, placeholder in (("train", (0, 1, 3), b"0"), ("val", (2,), b"1")):
            expected = []
            for source_rows in self.sources.values():
                for index in indices:
                    row = source_rows[index]
                    pixels = bytes((index, 20, 30)) * 16
                    key = hashlib.sha256(b"RGB:4:4:" + pixels).hexdigest()
                    expected.append({"conversations": row["conversations"],
                                     "image": [str(self.output / "images" / f"{key}.png")]})
            payload = (json.dumps(expected, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
            self.assertEqual((self.output / f"{split}.json").read_bytes(), payload)
            digest = hashlib.sha256(payload).hexdigest()
            self.assertEqual(files[f"{split}.json"], digest)
            raw = raw.replace(digest.encode(), placeholder * 64)
        self.assertEqual(hashlib.sha256(raw).hexdigest(),
                         "fb7a34f32a38e579796124c4e495e34f0945dfb9efc553b46a19a19c15373eab")

    def test_per_source_caps_override_fallback_without_reading_past_either(self):
        def stream(n, offset):
            yield from rows(n, offset=offset)
            raise AssertionError("read past source cap")

        self.sources = {"hmer": stream(5, 0), "count": stream(3, 0)}
        self.prepare(max_samples=3, max_samples_by_source={"hmer": 5})
        self.assertEqual(Counter(r["source"] for r in self.manifest()["records"]),
                         {"hmer": 5, "count": 3})
        self.assertEqual(self.manifest()["settings"]["max_samples_by_source"], {"hmer": 5})

    def test_per_source_caps_reject_unknown_names_and_invalid_values(self):
        for caps in ({"unknown": 5}, {"hmer": True}, {"hmer": 1}, {"hmer": 2.0}, [("hmer", 5)]):
            with self.subTest(caps=caps), self.assertRaisesRegex(ValueError, "Per-source caps"):
                self.prepare(max_samples_by_source=caps)
        self.assertFalse(self.output.exists())

    def test_expansion_preserves_every_base_record_and_splits_only_new_groups(self):
        original = self.prepare(max_samples=20)
        old = self.manifest()
        archive = self.root / "base.json.gz"
        archive.write_bytes(gzip.compress(Path(original["manifest_path"]).read_bytes(), mtime=0))
        base_records = {(r["source"], r["row"]): r for r in old["records"]}
        val_record = next(r for r in old["records"] if r["split"] == "val")
        train_record = next(r for r in old["records"] if r["split"] == "train")
        self.sources["hmer"] = rows(40)
        self.sources["new"] = rows(30, offset=128)
        for row, original_record in zip(self.sources["new"], (val_record, train_record)):
            row["image"] = self.sources[original_record["source"]][original_record["row"]]["image"]
        self.entries["new"] = entry("mathwriting_train")
        args = dict(max_samples=20, max_samples_by_source={"hmer": 40, "new": 30},
                    base_manifest=archive, base_manifest_sha256=original["manifest_sha256"])
        self.output = self.root / "expanded"
        self.prepare(**args)
        expanded = self.manifest()
        self.assertEqual(expanded["settings"]["base_manifest_sha256"], original["manifest_sha256"])
        splits_by_group = {}
        for r in expanded["records"]:
            splits_by_group.setdefault(r["group"], set()).add(r["split"])
        current_records = {(r["source"], r["row"]): r for r in expanded["records"]}
        for key, record in base_records.items():
            self.assertEqual(current_records[key]["split"], record["split"])
        self.assertEqual(current_records["new", 0]["split"], "val")
        self.assertEqual(current_records["new", 1]["split"], "train")
        self.assertTrue(all(len(splits) == 1 for splits in splits_by_group.values()))
        inherited = {current_records[key]["group"] for key in base_records}
        new_groups = set(splits_by_group) - inherited
        self.assertEqual(len(new_groups), 48)
        self.assertEqual(sum(splits_by_group[g] == {"val"} for g in new_groups), round(.1 * len(new_groups)))
        self.output = self.root / "repeat"
        self.prepare(**args)
        self.assertEqual(self.manifest()["records"], expanded["records"])

    def test_base_manifest_rejects_cross_split_bridge_from_new_row(self):
        result = self.prepare(max_samples=20)
        train = next(r for r in self.manifest()["records"] if r["split"] == "train")
        val = next(r for r in self.manifest()["records"] if r["split"] == "val")
        bridge = dict(self.sources[train["source"]][train["row"]], id=val["source_id"])
        self.sources["hmer"].append(bridge)
        self.output = self.root / "expanded"
        with self.assertRaisesRegex(ValueError, "joins both base train and validation"):
            self.prepare(max_samples=21, base_manifest=result["manifest_path"])
        self.assertFalse(self.output.exists())

    def test_base_manifest_rejects_missing_changed_or_unpinned_records(self):
        result = self.prepare(max_samples=20)
        self.output = self.root / "expanded"
        with self.assertRaisesRegex(ValueError, "Base split manifest SHA256 differs"):
            self.prepare(base_manifest=result["manifest_path"], base_manifest_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "Base split record missing"):
            self.prepare(max_samples=19, base_manifest=result["manifest_path"])
        self.sources["hmer"][0]["image"] = {"bytes": image_bytes(128)}
        with self.assertRaisesRegex(ValueError, "Base split record changed"):
            self.prepare(base_manifest=result["manifest_path"])
        self.assertFalse(self.output.exists())

    def test_stream_is_not_consumed_beyond_cap(self):
        def stream():
            yield from rows(3)
            raise AssertionError("read past cap")
        self.sources = {"hmer": stream()}
        self.prepare(max_samples=3)

    def test_reencoding_same_pixels_is_one_group(self):
        self.sources["count"][0]["image"] = {"bytes": image_bytes(0, "BMP")}
        self.sources["count"][0]["id"] = "different-record-id"
        self.prepare()
        selected = [r for r in self.manifest()["records"] if r["row"] == 0]
        self.assertEqual(len({r["group"] for r in selected}), 1)

    def test_source_id_also_groups_altered_encoding_and_pixels(self):
        self.sources["count"][0]["image"] = {"bytes": image_bytes(200)}
        self.prepare()
        selected = [r for r in self.manifest()["records"] if r["row"] == 0]
        self.assertEqual(len({r["group"] for r in selected}), 1)

    def test_same_numeric_id_in_different_family_does_not_force_group(self):
        self.sources["other"] = rows(offset=100)
        self.entries["other"] = entry("hme100k_train")
        self.prepare()
        selected = [r for r in self.manifest()["records"] if r["row"] == 0]
        self.assertEqual(len({r["group"] for r in selected}), 2)

    def test_cache_reuse_does_not_consume_source_or_rewrite_files(self):
        first = self.prepare()
        timestamp = (self.output / "split_manifest.json").stat().st_mtime_ns
        self.sources = {name: iter(()) for name in self.sources}
        self.assertEqual(self.prepare(), first)
        self.assertEqual((self.output / "split_manifest.json").stat().st_mtime_ns, timestamp)

    def test_modified_cached_payload_stops(self):
        self.prepare()
        path = self.output / "train.json"
        path.write_text("[]")
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual(path.read_text(), "[]")

    def test_changed_settings_do_not_overwrite(self):
        self.prepare()
        original = (self.output / "split_manifest.json").read_bytes()
        with self.assertRaises(ValueError):
            self.prepare(seed=43)
        self.assertEqual((self.output / "split_manifest.json").read_bytes(), original)

    def test_changed_source_registry_does_not_reuse_old_subset(self):
        self.prepare()
        self.entries["count"]["subset"] = "different_source"
        with self.assertRaises(ValueError):
            self.prepare()

    def test_exported_registry_and_media_paths_are_loadable(self):
        self.prepare()
        registry = json.loads((self.output / "dataset_info.json").read_text())
        for name in ("baseline_train", "baseline_val"):
            records = json.loads((self.output / registry[name]["file_name"]).read_text())
            self.assertTrue(records)
            for record in records:
                with Image.open(record["image"][0]) as image:
                    self.assertEqual(image.mode, "RGB")

    def test_identical_seed_same_membership_in_different_directories(self):
        self.prepare()
        first = self.manifest()["records"]
        self.output = self.root / "second"
        self.prepare()
        self.assertEqual(self.manifest()["records"], first)

    def test_empty_or_invalid_source_not_published(self):
        self.sources["count"] = []
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.output.exists())
        self.assertTrue(list(self.root.glob("data.partial-*")))

    def test_build_lock_is_not_removed_if_owned_by_another_call(self):
        lock = self.output.with_name("data.building")
        lock.mkdir()
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertTrue(lock.exists())

    def test_invalid_settings_rejected(self):
        for changes in (dict(max_samples=1), dict(max_samples=True), dict(val_ratio=0),
                        dict(val_ratio=1), dict(val_ratio=float("nan")), dict(seed=True)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.prepare(**changes)
        self.assertFalse(self.output.exists())

    def test_bad_image_is_error_not_silently_dropped(self):
        self.sources["hmer"][0]["image"] = {"bytes": b"broken"}
        with self.assertRaises(Exception):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_parquet_prefix_preserves_rows_and_stops_before_next_file(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        path = self.root / "source.parquet"
        pq.write_table(pa.Table.from_pylist(rows()), path, row_group_size=10)
        with closing(_parquet_rows([path, self.root / "must-not-open.parquet"],
                                   ["conversations", "image"], 2)) as stream:
            selected = list(stream)
        self.assertEqual([r["id"] for r in selected], ["0", "1"])
        self.assertEqual(selected[0]["image"]["bytes"], image_bytes(0))

    def test_parquet_early_stop_exits_cleanly_in_fresh_processes(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        path = self.root / "source.parquet"
        pq.write_table(pa.Table.from_pylist(rows()), path, row_group_size=10)
        code = ("from contextlib import closing; from itertools import islice; "
                "from scripts.prepare_baseline_data import _parquet_rows; "
                f"stream=_parquet_rows([{str(path)!r}], ['conversations','image'], 2); "
                "\nwith closing(stream): assert len(list(islice(stream, 2))) == 2\n")
        for _ in range(3):
            result = subprocess.run([sys.executable, "-B", "-c", code],
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("Fatal Python error", result.stderr)

    def test_parquet_missing_required_columns_is_not_silently_dropped(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        path = self.root / "bad.parquet"
        pq.write_table(pa.table({"id": ["0"]}), path)
        with closing(_parquet_rows([path], ["conversations", "image"], 2)) as stream:
            with self.assertRaises(ValueError):
                next(stream)

    def test_requested_dataset_revision_is_used_for_all_remote_loads(self):
        project = self.root / "project"
        (project / "train").mkdir(parents=True)
        (project / "train/dataset_info.json").write_text(json.dumps(self.entries))
        revision = "a" * 40
        builder = mock.Mock()
        builder.info.builder_name = "parquet"
        builder.config.data_files = {"train": ["fixture.parquet"]}
        with mock.patch("huggingface_hub.HfApi") as api, \
                mock.patch("datasets.load_dataset_builder", return_value=builder) as load, \
                mock.patch("scripts.prepare_baseline_data._parquet_rows",
                           side_effect=lambda *a: (row for row in rows())) as read:
            api.return_value.dataset_info.return_value.sha = revision
            result = prepare_baseline(project, "hmer,count", self.output, dataset_revision=revision,
                                      max_samples=4, max_samples_by_source={"count": 7})
        api.return_value.dataset_info.assert_called_once_with("fixture/repo", revision=revision)
        self.assertEqual(load.call_count, 2)
        self.assertTrue(all(call.kwargs["revision"] == revision for call in load.call_args_list))
        self.assertEqual([call.args[-1] for call in read.call_args_list], [4, 7])
        self.assertEqual(result["counts"]["train_rows"] + result["counts"]["val_rows"], 11)
        self.assertEqual(json.loads(Path(result["manifest_path"]).read_text())["source_revisions"],
                         {"fixture/repo": revision})


if __name__ == "__main__":
    unittest.main()

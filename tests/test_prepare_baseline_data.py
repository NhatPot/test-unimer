import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from scripts.prepare_baseline_data import prepare_from_rows, reuse_prepared, _settings


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


if __name__ == "__main__":
    unittest.main()

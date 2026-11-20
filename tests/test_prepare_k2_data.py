"""CPU-only fixtures for K2 input selection and manifests."""

from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts import prepare_k2_data as k2
from scripts.checkpoint_contract import sha256_file
from scripts.prepare_baseline_data import prepare_from_rows
from test_prepare_baseline_data import entry, rows


def write_package(path, data):
    raw = json.dumps(data).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(raw, mtime=0))
    digest = hashlib.sha256(raw).hexdigest()
    path.with_suffix("").with_suffix(".sha256").write_text(digest + "\n")
    return digest


def onpolicy_row(record, task):
    return {
        "id": f"k2_onpolicy:{task}:{record['source_id']}",
        "img_id": record["source_id"],
        "image_sha256": record["image_sha256"],
        "messages": [{"from": "human", "value": f"<image>Fixture {task}?"},
                     {"from": "gpt", "value": "Fixture answer"}],
    }


class K2DataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.k1 = self.root / "k1"
        self.package = self.root / "onpolicy.json.gz"
        self.tasks = {"error_find": 3, "error_fix": 2}
        sources = {
            k2.RECOGNITION: rows(12),
            f"{k2.RECOGNITION}_error_find": rows(12),
            f"{k2.RECOGNITION}_error_fix": rows(10),
            "parquet_hme100k_train": rows(6, offset=100),
        }
        result = prepare_from_rows(self.k1, sources,
                                   {source: entry(source.removeprefix("parquet_")) for source in sources})
        self.k1_sha = result["manifest_sha256"]
        self.manifest = json.loads(Path(result["manifest_path"]).read_text())
        train = self.recognition("train")
        self.data = {task: [onpolicy_row(record, task) for record in train[:count]]
                     for task, count in self.tasks.items()}
        self.package_sha = write_package(self.package, self.data)

    def recognition(self, split):
        return [r for r in self.manifest["records"] if r["source"] == k2.RECOGNITION and r["split"] == split]

    def prepare(self, variant="onpolicy", **kwargs):
        options = dict(expected_onpolicy_sha256=self.package_sha, expected_k1_sha256=self.k1_sha,
                       replay_size=18, edl_counts=self.tasks)
        options.update(kwargs)
        return k2.prepare_k2_data(self.k1, self.package, self.root / variant, variant, **options)

    def test_largest_remainders_exact_proportional_total_and_tie(self):
        self.assertEqual(k2.replay_quotas({"z": 1, "a": 1, "b": 1}, 2), {"z": 0, "a": 1, "b": 1})
        self.assertEqual(k2.replay_quotas({"a": 8, "b": 2, "c": 1}, 5), {"a": 4, "b": 1, "c": 0})
        with self.assertRaisesRegex(ValueError, "Cannot sample"):
            k2.replay_quotas({"a": 8}, 9)

    def test_pair_preserves_replay_validation_counts_images_and_checksums(self):
        results = {variant: self.prepare(variant) for variant in ("onpolicy", "control")}
        manifests = {}
        train = {}
        for variant, result in results.items():
            directory = Path(result["dataset_dir"])
            path = Path(result["manifest_path"])
            manifest = manifests[variant] = json.loads(path.read_text())
            train[variant] = json.loads((directory / "train.json").read_text())
            for name, digest in manifest["files"].items():
                self.assertEqual(sha256_file(directory / name), digest)
            self.assertEqual(sha256_file(path), result["manifest_sha256"])
            self.assertEqual((directory / "val.json").read_bytes(), (self.k1 / "val.json").read_bytes())
            self.assertEqual(manifest["k1_split_sha256"], self.k1_sha)
            self.assertEqual(manifest["onpolicy_package_sha256"], self.package_sha)
            self.assertEqual(manifest["counts"]["train"]["rows"], 23)
            self.assertEqual(manifest["counts"]["edl"]["by_task"], self.tasks)
            self.assertEqual(manifest["counts"]["replay"]["by_source"],
                             k2.replay_quotas(Counter(r["source"] for r in self.manifest["records"]
                                                      if r["split"] == "train"), 18))
            self.assertTrue(all(Path(row["image"][0]).is_file() for row in train[variant]))
            self.assertEqual(len(set(manifest["replay_row_ids"])), 18)
            self.assertEqual(self.prepare(variant), result)
        self.assertEqual(manifests["onpolicy"]["replay_row_ids"], manifests["control"]["replay_row_ids"])
        self.assertEqual(train["onpolicy"][:18], train["control"][:18])
        self.assertTrue(set(manifests["onpolicy"]["edl_row_ids"]).isdisjoint(manifests["control"]["edl_row_ids"]))
        self.assertFalse(any(row in train["control"][18:] for row in train["onpolicy"][18:]))
        self.assertEqual(manifests["control"]["control_edl_rows_reused_from_replay"], dict.fromkeys(self.tasks, 0))
        self.assertNotEqual(results["onpolicy"]["manifest_sha256"], results["control"]["manifest_sha256"])

    def test_control_prefers_outside_replay_and_records_fallback(self):
        result = self.prepare("control", replay_size=self.manifest["counts"]["train_rows"] - 1)
        manifest = json.loads(Path(result["manifest_path"]).read_text())
        ids = set(manifest["replay_row_ids"])
        available = Counter(r["source"] for r in self.manifest["records"]
                            if r["split"] == "train" and f"{r['source']}:{r['row']}" not in ids)
        expected = {task: self.tasks[task] - available[source] for source, task in k2.EDL_SOURCES.items()}
        self.assertEqual(manifest["control_edl_rows_reused_from_replay"], expected)
        self.assertEqual(len(ids.intersection(manifest["edl_row_ids"])), sum(expected.values()))

    def test_control_excludes_exact_onpolicy_payload_even_if_original_id_differs(self):
        record = self.recognition("train")[0]
        original = next(row for row in json.loads((self.k1 / "train.json").read_text())
                        if Path(row["image"][0]).stem == record["image_sha256"])
        self.data["error_find"][0]["messages"] = original["conversations"]
        digest = write_package(self.package, self.data)
        result = self.prepare("control", expected_onpolicy_sha256=digest)
        selected = json.loads((Path(result["dataset_dir"]) / "train.json").read_text())
        self.assertNotIn(original, selected[18:])

    def test_missing_and_misnamed_image_fail_clearly(self):
        digest = self.data["error_find"][0]["image_sha256"]
        path = self.k1 / "images" / f"{digest}.png"
        path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing K1 image for on-policy row"):
            self.prepare("onpolicy")
        other_digest = self.recognition("train")[-1]["image_sha256"]
        path.write_bytes((self.k1 / "images" / f"{other_digest}.png").read_bytes())
        with self.assertRaisesRegex(ValueError, "RGB SHA256 mismatch"):
            self.prepare("onpolicy")
        self.assertFalse((self.root / "onpolicy").exists())

    def test_image_or_group_in_val_is_rejected(self):
        self.data["error_find"][0] = onpolicy_row(self.recognition("val")[0], "error_find")
        digest = write_package(self.package, self.data)
        with self.assertRaisesRegex(ValueError, "overlaps K1 validation"):
            self.prepare("onpolicy", expected_onpolicy_sha256=digest)

    def test_pins_and_cache_corruption_fail_without_overwriting(self):
        with self.assertRaisesRegex(ValueError, "K1 split manifest SHA256"):
            self.prepare(expected_k1_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "On-policy package SHA256 mismatch"):
            self.prepare(expected_onpolicy_sha256="0" * 64)
        result = self.prepare()
        train = Path(result["dataset_dir"]) / "train.json"
        train.write_text("[]")
        with self.assertRaisesRegex(ValueError, "Existing K2 dataset differs"):
            self.prepare()
        self.assertEqual(train.read_text(), "[]")

    def test_cli_requires_the_pinned_k1_split(self):
        args = ["--k1-dir", str(self.k1), "--onpolicy-package", str(self.package),
                "--output-dir", str(self.root / "cli"), "--variant", "onpolicy"]
        with self.assertRaisesRegex(ValueError, "K1 split manifest SHA256 mismatch"):
            k2.main(args)

    def test_repo_package_matches_metadata_and_has_no_image_paths(self):
        package = Path(__file__).resolve().parents[1] / "train/k2_data/onpolicy_b4_500.json.gz"
        raw = gzip.decompress(package.read_bytes())
        checksum = package.with_suffix("").with_suffix(".sha256").read_text().strip()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), checksum)
        data = json.loads(raw)
        self.assertEqual({task: len(rows) for task, rows in data.items()}, k2.EDL_COUNTS)
        self.assertTrue(all(set(row) == {"messages", "image_sha256", "id", "img_id"}
                            for rows in data.values() for row in rows))

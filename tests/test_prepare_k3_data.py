"""CPU fixtures for the CROHME-only split and its validation guards."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.checkpoint_contract import sha256_file
from scripts.prepare_baseline_data import prepare_from_rows
from scripts.prepare_k3_data import CROHME_SOURCES, VAL_H_SOURCE, prepare_k3_data
from test_prepare_baseline_data import entry, rows


class K3DataTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.k1, self.output, self.b4_manifest = root / "k1", root / "k3", root / "b4.json"
        source_rows = {source: rows(20) for source in CROHME_SOURCES}
        recognition = CROHME_SOURCES[0]
        source_rows[recognition].append(source_rows[recognition][0])
        source_rows[VAL_H_SOURCE] = rows(10, offset=50)
        source_rows["parquet_mathwriting_train"] = rows(10, offset=100)
        source_entries = {source: entry(source.removeprefix("parquet_")) for source in source_rows}
        b4_sources = {name: data for name, data in source_rows.items() if name != "parquet_mathwriting_train"}
        b4 = prepare_from_rows(root / "b4", b4_sources, source_entries)
        self.b4_manifest.write_bytes(Path(b4["manifest_path"]).read_bytes())
        k1 = prepare_from_rows(self.k1, source_rows, source_entries, base_manifest=self.b4_manifest,
                               base_manifest_sha256=sha256_file(self.b4_manifest))
        self.k1_manifest = Path(k1["manifest_path"])

    def prepare(self, **kwargs):
        return prepare_k3_data(
            self.k1, self.output, expected_k1_sha256=sha256_file(self.k1_manifest),
            b4_manifest=self.b4_manifest, expected_b4_sha256=sha256_file(self.b4_manifest), **kwargs,
        )

    def test_selects_ceiling_half_of_crohme_and_keeps_all_crohme_val(self):
        result = self.prepare()
        manifest_path = Path(result["manifest_path"])
        manifest = json.loads(manifest_path.read_bytes())
        pool = dict.fromkeys(CROHME_SOURCES, 18)
        pool[CROHME_SOURCES[0]] = 19
        self.assertEqual(manifest["train_pool_by_source"], pool)
        self.assertEqual(result["counts"]["train"]["rows"], 46)
        self.assertEqual(result["counts"]["val"]["by_source"], dict.fromkeys(CROHME_SOURCES, 2))
        self.assertEqual(sum(result["counts"]["train"]["by_source"].values()), 46)
        self.assertEqual(set(result["counts"]["train"]["by_source"]), set(CROHME_SOURCES))
        self.assertEqual(len(set(manifest["train_row_ids"])), 46)
        expected_hashes = {path: sha256_file(self.output / path) for path in
                           ("train.json", "val.json", "dataset_info.json")}
        self.assertEqual(manifest["files"], expected_hashes)
        self.assertEqual(result["manifest_sha256"], hashlib.sha256(manifest_path.read_bytes()).hexdigest())
        self.assertEqual(self.prepare(), result)
        with self.assertRaisesRegex(ValueError, "Existing K3 dataset differs"):
            self.prepare(seed=43)

    def test_rejects_val_c_source_id_even_when_rgb_and_group_differ(self):
        manifest = json.loads(self.k1_manifest.read_bytes())
        recognition = "parquet_crohme_train"
        protected = next(r for r in manifest["records"] if r["split"] == "val" and r["source"] == recognition)
        train = next(r for r in manifest["records"] if r["split"] == "train" and r["source"] in CROHME_SOURCES)
        train["source_id"] = protected["source_id"]
        self.k1_manifest.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "overlaps validation"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_rejects_val_h_rgb_hash_from_the_b4_manifest(self):
        manifest = json.loads(self.b4_manifest.read_bytes())
        protected = next(r for r in manifest["records"] if r["split"] == "val" and r["source"] == VAL_H_SOURCE)
        train = next(r for r in manifest["records"] if r["split"] == "train" and r["source"] in CROHME_SOURCES)
        protected["image_sha256"] = train["image_sha256"]
        self.b4_manifest.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "overlaps validation"):
            self.prepare()
        self.assertFalse(self.output.exists())

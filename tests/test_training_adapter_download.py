"""Check adapter-only downloads using a local fake MLflow artifact store."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts import dagshub_checkpoint as downloader
from scripts.checkpoint_contract import BASE_MODEL


class TrainingAdapterDownloadTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.remote = self.root / "remote"
        self.remote.mkdir()
        self.run_id = "a" * 32
        self.revision = "b" * 40
        config = self.remote / "adapter_config.json"
        config.write_text(json.dumps({"peft_type": "LORA", "base_model_name_or_path": BASE_MODEL,
                                      "r": 64, "lora_alpha": 16, "revision": self.revision}))
        self.weights = self.remote / "adapter_model.safetensors"
        self.weights.write_bytes(b"fixture")
        self.weights_sha = hashlib.sha256(self.weights.read_bytes()).hexdigest()
        self.manifest = self.remote / "artifact_manifest.json"
        entries = [{"path": path.name, "size": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for path in (config, self.weights)]
        entries.append({"path": "checkpoint-500/optimizer.pt", "size": 1, "sha256": "d" * 64})
        self.manifest.write_text(json.dumps({"files": entries, "file_count": len(entries)}))
        self.mlflow = Mock()
        self.mlflow.artifacts.download_artifacts.side_effect = self.download
        configure = patch.object(downloader, "_configure_mlflow", return_value=self.mlflow)
        configure.start()
        self.addCleanup(configure.stop)

    def download(self, *, artifact_path, run_id, dst_path):
        self.assertEqual(run_id, self.run_id)
        self.assertEqual(Path(artifact_path).parent, Path("training"))
        original = self.remote / Path(artifact_path).name
        return shutil.copyfile(original, Path(dst_path) / original.name)

    def prepare(self, weights_sha=None):
        return Path(downloader.prepare_training_adapter(
            self.root, "cache", self.run_id, weights_sha or self.weights_sha, self.revision,
        ))

    def test_downloads_only_adapter_files_reverifies_cache_and_rejects_corruption(self):
        path = self.prepare()
        self.assertEqual({p.name for p in path.iterdir()}, set(downloader.TRAINING_ADAPTER_FILES))
        self.assertEqual(self.prepare(), path)
        self.assertEqual([call.kwargs["artifact_path"] for call in self.mlflow.artifacts.download_artifacts.call_args_list],
                         ["training/artifact_manifest.json", "training/adapter_config.json",
                          "training/adapter_model.safetensors"])
        (path / self.weights.name).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Checkpoint file hash mismatch"):
            self.prepare()

    def test_manifest_must_cover_root_files_and_match_pinned_weights(self):
        with self.assertRaisesRegex(ValueError, "pinned weights SHA256"):
            self.prepare("0" * 64)
        data = json.loads(self.manifest.read_text())
        data["files"][0]["path"] = "checkpoint-500/adapter_config.json"
        self.manifest.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "root adapter file"):
            self.prepare()
        self.assertFalse((self.root / "cache" / self.run_id).exists())

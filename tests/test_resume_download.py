"""Resume plumbing only: fake checkpoint bytes, no network/model/training."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from scripts import dagshub_checkpoint as downloader
from scripts.train_with_checkpoint_sync import _seed_resume_best


class ResumeDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.remote = self.root / "remote/checkpoint-338"
        self.remote.mkdir(parents=True)
        for name in ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt",
                     "scheduler.pt", "scaler.pt", "rng_state_0.pth", "rng_state_1.pth"):
            (self.remote / name).write_bytes(b"fixture bytes only: " + name.encode())
        (self.remote / "trainer_state.json").write_text(json.dumps({"epoch": 2.0, "global_step": 338}))
        self.manifest = self.remote.parent / "artifact_manifest.json"
        self.manifest.write_text(json.dumps({"files": [
            {"path": f"checkpoint-338/{p.name}", "size": p.stat().st_size,
             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in self.remote.iterdir()
        ]}))
        self.client = mock.Mock()
        self.client.artifacts.download_artifacts.side_effect = self.download
        configure = mock.patch.object(downloader, "_configure_mlflow", return_value=self.client)
        configure.start()
        self.addCleanup(configure.stop)
        environment = mock.patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)

    def download(self, *, run_id, artifact_path, dst_path):
        dest = Path(dst_path)
        dest.mkdir(parents=True, exist_ok=True)
        if artifact_path.endswith("artifact_manifest.json"):
            target = dest / "artifact_manifest.json"
            shutil.copyfile(self.manifest, target)
        else:
            name = Path(artifact_path).name
            target = dest / name
            shutil.copytree(self.remote.parent / name, target)
        return str(target)

    def prepare(self, **kwargs):
        options = dict(project_dir=self.root, output_dir="cache", enabled=True,
                       run_id="a" * 32, artifact_path="training/checkpoint-338",
                       world_size=2, fp16=True, manifest_artifact="training/artifact_manifest.json")
        options.update(kwargs)
        return Path(downloader.prepare_resume_checkpoint(**options))

    def test_full_download_and_sha256_check(self):
        path = self.prepare()
        self.assertIn("a" * 32, path.parts)
        self.assertEqual(path.name, "checkpoint-338")
        self.assertEqual(os.environ["RESUME_CHECKPOINT"], str(path))
        self.assertTrue((path / "optimizer.pt").is_file())
        self.assertEqual(self.client.artifacts.download_artifacts.call_count, 2)

    def test_disabled_resume_has_no_network_or_files(self):
        os.environ["RESUME_CHECKPOINT"] = "stale"
        self.assertIsNone(downloader.prepare_resume_checkpoint(self.root, "cache", enabled=False))
        self.client.artifacts.download_artifacts.assert_not_called()
        self.assertNotIn("RESUME_CHECKPOINT", os.environ)
        self.assertFalse((self.root / "cache").exists())

    def test_two_runs_with_same_step_do_not_share_cache(self):
        first = self.prepare()
        second = self.prepare(run_id="b" * 32)
        self.assertNotEqual(first, second)
        self.assertTrue(first.is_dir())

    def test_missing_optimizer_rng_or_scaler_blocks_full_resume(self):
        for name in ("optimizer.pt", "rng_state_0.pth", "rng_state_1.pth", "scaler.pt",
                     "adapter_config.json"):
            with self.subTest(name=name):
                payload = (self.remote / name).read_bytes()
                (self.remote / name).unlink()
                with self.assertRaisesRegex(RuntimeError, name):
                    self.prepare(output_dir=f"cache-{name}")
                (self.remote / name).write_bytes(payload)

    def test_same_size_cache_corruption_stops_without_overwriting(self):
        path = self.prepare()
        weights = path / "adapter_model.safetensors"
        weights.write_bytes(b"x" * weights.stat().st_size)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.prepare()
        self.assertNotIn("RESUME_CHECKPOINT", os.environ)
        self.assertEqual(weights.read_bytes(), b"x" * weights.stat().st_size)

    def test_download_failure_clears_stale_checkpoint(self):
        os.environ["RESUME_CHECKPOINT"] = "old checkpoint"
        self.client.artifacts.download_artifacts.side_effect = RuntimeError("download failed")
        with self.assertRaisesRegex(RuntimeError, "download failed"):
            self.prepare()
        self.assertNotIn("RESUME_CHECKPOINT", os.environ)

    def test_manifest_must_cover_optimizer_not_only_adapter(self):
        manifest = json.loads(self.manifest.read_text())
        manifest["files"] = [e for e in manifest["files"] if not e["path"].endswith("optimizer.pt")]
        self.manifest.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "absent from manifest: optimizer.pt"):
            self.prepare()

    def test_single_gpu_non_fp16_mode_remains_available(self):
        for name in ("rng_state_0.pth", "rng_state_1.pth", "scaler.pt"):
            (self.remote / name).unlink()
        (self.remote / "rng_state.pth").write_bytes(b"single GPU fixture")
        path = self.prepare(world_size=1, fp16=False, manifest_artifact=None)
        self.assertTrue((path / "rng_state.pth").exists())

    def test_v5_python_wrapper_still_accepts_original_arguments(self):
        state = self.root / "resume.json"
        def subprocess_run(command, **kwargs):
            self.assertEqual(command[0], "/own-env/bin/python")
            self.assertEqual(command[command.index("--world-size") + 1], "2")
            self.assertNotIn("--allow-model-only", command)
            state.write_text(json.dumps({"checkpoint": "/cache/checkpoint-338"}))
        with mock.patch.object(downloader.subprocess, "run", side_effect=subprocess_run):
            checkpoint = downloader.prepare_resume_checkpoint_via_python(
                python="/own-env/bin/python", project_dir=self.root, output_dir="cache",
                enabled=True, run_id="a" * 32, run_uuid="", experiment_name="test",
                artifact_path="checkpoints/checkpoint-338", state_path=state,
            )
        self.assertEqual(checkpoint, "/cache/checkpoint-338")

    def test_uuid_lookup_from_v5_is_preserved(self):
        with mock.patch.object(downloader, "_find_run_id_by_uuid", return_value="a" * 32) as lookup:
            self.prepare(run_id=None, run_uuid="old-notebook-uuid", experiment_name="test")
        lookup.assert_called_once_with("test", "old-notebook-uuid")

    def add_best(self):
        best = self.remote.parent / "checkpoint-300"
        shutil.copytree(self.remote, best)
        (best / "trainer_state.json").write_text(json.dumps({"global_step": 300, "epoch": 1.8}))
        state = {"epoch": 2.0, "global_step": 338, "best_metric": .09,
                 "best_model_checkpoint": "/old-kaggle/checkpoints/checkpoint-300"}
        (self.remote / "trainer_state.json").write_text(json.dumps(state))
        self.manifest.write_text(json.dumps({"files": [
            {"path": f"{folder.name}/{p.name}", "size": p.stat().st_size,
             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for folder in (self.remote, best) for p in folder.iterdir()
        ]}))
        return state

    def test_resume_preserves_best_and_does_not_modify_verified_source(self):
        original_state = self.add_best()
        output = self.root / "new-run/checkpoints"
        working = self.prepare(training_output_dir=output)
        state = json.loads((working / "trainer_state.json").read_text())
        self.assertEqual(state["global_step"], 338)
        self.assertEqual(state["epoch"], 2.0)
        self.assertEqual(state["best_metric"], .09)
        self.assertEqual(state["best_model_checkpoint"], str(output / "checkpoint-300"))
        self.assertFalse(output.exists())
        cached = self.root / ("cache/" + "a" * 32 + "/checkpoint-338")
        self.assertEqual(json.loads((cached / "trainer_state.json").read_text()), original_state)
        self.assertNotEqual((cached / "trainer_state.json").stat().st_ino,
                            (working / "trainer_state.json").stat().st_ino)
        for file in cached.iterdir():
            if file.name != "trainer_state.json":
                self.assertEqual(file.read_bytes(), (working / file.name).read_bytes())
        best = Path(os.environ["RESUME_BEST_CHECKPOINT"])
        _seed_resume_best(best, output)
        self.assertTrue((output / "checkpoint-300/optimizer.pt").is_file())
        self.assertTrue(best.is_dir())
        with self.assertRaises(FileExistsError):
            _seed_resume_best(best, output)

    def test_missing_parent_best_does_not_silently_reset_best_metric(self):
        self.add_best()
        shutil.rmtree(self.remote.parent / "checkpoint-300")
        with self.assertRaises(FileNotFoundError):
            self.prepare(training_output_dir=self.root / "new-output")
        self.assertNotIn("RESUME_CHECKPOINT", os.environ)
        self.assertFalse((self.root / "new-output").exists())

    def test_no_recorded_best_does_not_create_output_or_reset_state(self):
        path = self.prepare(training_output_dir=self.root / "new-output")
        self.assertEqual(json.loads((path / "trainer_state.json").read_text())["global_step"], 338)
        self.assertNotIn("RESUME_BEST_CHECKPOINT", os.environ)
        self.assertFalse((self.root / "new-output").exists())


if __name__ == "__main__":
    unittest.main()

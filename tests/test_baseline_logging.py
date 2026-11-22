"""CPU-only coverage for the legacy baseline logging/sync entry points."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from functools import partial
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import selectors
import signal
import subprocess
import sys
import tempfile
from textwrap import dedent
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
RUN_UUID = "a" * 32
RUN_ID = "b" * 32
SECRET_ERROR = "https://user:PRIVATE_TOKEN@example.org/?token=PRIVATE_TOKEN"


class FakeClient:
    """Relevant MlflowClient call signatures, backed only by in-memory bytes."""

    def __init__(self):
        self.experiment = SimpleNamespace(experiment_id="17", name="baseline")
        self.runs = [SimpleNamespace(info=SimpleNamespace(run_id=RUN_ID),
                                     data=SimpleNamespace(tags={"mlflow.runName": "old-short"}))]
        self.files = {}
        self.searches = []
        self.listings = []
        self.tags = {}
        self.omit = set()
        self.sizes = {}
        self.created = 0
        self.terminations = []

    def log_artifact(self, run_id, local_path, artifact_path=None):
        self.sdk.log_artifact(local_path, artifact_path, run_id)

    def log_artifacts(self, run_id, local_dir, artifact_path=None):
        self.sdk.log_artifacts(local_dir, artifact_path, run_id)

    def get_experiment_by_name(self, name):
        return self.experiment

    def search_runs(self, experiment_ids, filter_string="", run_view_type=None,
                    max_results=1000, order_by=None, page_token=None):
        self.searches.append((experiment_ids, filter_string, max_results, order_by))
        return self.runs[:max_results]

    def list_artifacts(self, run_id, path=None):
        self.listings.append((run_id, path))
        prefix = f"{path}/" if path else ""
        entries = {}
        for remote, content in self.files.items():
            if remote in self.omit or not remote.startswith(prefix):
                continue
            child = remote[len(prefix):].split("/", 1)
            name = prefix + child[0]
            entries[name] = SimpleNamespace(path=name, is_dir=len(child) > 1,
                                            file_size=self.sizes.get(remote, len(content)))
        return list(entries.values())

    def set_tag(self, run_id, key, value):
        self.tags[key] = value

    def create_run(self, experiment_id, start_time=None, tags=None, run_name=None):
        self.created += 1
        return self.runs[0]

    def log_metric(self, run_id, key, value, timestamp=None, step=None, synchronous=None):
        pass

    def set_terminated(self, run_id, status=None, end_time=None):
        self.terminations.append((run_id, status))

    def delete_run(self, run_id):
        pass


class FakeMlflow(ModuleType):
    def __init__(self):
        super().__init__("mlflow")
        self.client = FakeClient()
        self.client.sdk = self
        self.MlflowClient = lambda: self.client
        self.tags = {}
        self.metrics = {}
        self.uploads = []
        self.upload_error = None
        self.discard_uploads = False
        self.set_experiment_calls = []

    def set_tracking_uri(self, uri):
        pass

    def set_experiment(self, experiment_name=None, experiment_id=None):
        self.set_experiment_calls.append(experiment_name)
        self.client.experiment = SimpleNamespace(experiment_id="17", name=experiment_name)
        return self.client.experiment

    @contextmanager
    def start_run(self, run_id=None, experiment_id=None, run_name=None):
        yield SimpleNamespace(info=SimpleNamespace(run_id=run_id))

    def set_tags(self, tags, run_id=None, synchronous=None):
        self.tags.update(tags)

    def log_metrics(self, metrics, step=None, synchronous=None, run_id=None, timestamp=None):
        self.metrics.update(metrics)

    def log_artifact(self, local_path, artifact_path=None, run_id=None):
        if self.upload_error:
            raise self.upload_error
        source = Path(local_path)
        self.uploads.append((source, artifact_path))
        if not self.discard_uploads:
            self.client.files[f"{artifact_path}/{source.name}"] = source.read_bytes()

    def log_artifacts(self, local_dir, artifact_path=None, run_id=None):
        if self.upload_error:
            raise self.upload_error
        source = Path(local_dir)
        self.uploads.append((source, artifact_path))
        if not self.discard_uploads:
            for path in source.rglob("*"):
                if path.is_file():
                    self.client.files[f"{artifact_path}/{path.relative_to(source).as_posix()}"] = path.read_bytes()


def load_script(name, sdk):
    spec = importlib.util.spec_from_file_location(f"_baseline_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"mlflow": sdk}):
        spec.loader.exec_module(module)
    return module


class BaselineLoggingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "output"
        self.output.mkdir()
        self.config = self.root / "runtime.yaml"
        self.config.write_text("load_best_model_at_end: false\n", encoding="utf-8")
        self.notebook = self.root / "source.ipynb"
        self.notebook.write_text("{}", encoding="utf-8")
        self.env = {"MLFLOW_TRACKING_URI": f"https://user:PRIVATE_TOKEN@example.org/repo.mlflow?token=PRIVATE_TOKEN",
                    "MLFLOW_TRACKING_USERNAME": "user", "MLFLOW_TRACKING_PASSWORD": "PRIVATE_TOKEN"}
        self.addCleanup(mock.patch.stopall)
        mock.patch.dict(os.environ, self.env, clear=True).start()
        self.sdk = FakeMlflow()
        mock.patch.dict(sys.modules, {"mlflow": self.sdk}).start()
        self.logger = load_script("dagshub_logger", self.sdk)
        self.wrapper = load_script("train_with_checkpoint_sync", self.sdk)
        mock.patch.object(self.logger, "_command", return_value="test-value").start()

    def test_training_log_upload_redacts_tracking_password(self):
        log_path = self.root / "train.log"
        log_path.write_bytes(b"step 1\nAuthorization PRIVATE_TOKEN leaked\n")
        captured = {}

        class Client:
            def log_artifact(self, run_id, local_path, artifact_path=None):
                captured[artifact_path] = Path(local_path).read_bytes()

        failures = []
        self.wrapper._upload_training_logs(Client(), "run", log_path, None, failures, "test")
        self.assertEqual(failures, [])
        self.assertNotIn(b"PRIVATE_TOKEN", captured["logs"])
        self.assertIn(b"[REDACTED]", captured["logs"])
        # The on-disk log stays untouched for local diagnosis.
        self.assertIn(b"PRIVATE_TOKEN", log_path.read_bytes())

    def upload(self):
        self.logger.upload_run("baseline", RUN_UUID, self.config, self.output, self.root, self.notebook)

    def checkpoint(self, name="checkpoint-5"):
        checkpoint = self.output / name
        checkpoint.mkdir()
        for name in (*self.wrapper.FULL_CHECKPOINT_FILES, "adapter_model.safetensors"):
            (checkpoint / name).write_bytes(b"test")
        return checkpoint

    def final(self):
        return self.wrapper._final_sync("baseline", RUN_UUID, self.output, "checkpoints",
                                        self.root / "state.json", 0, True)

    def test_unique_lookup_uses_two_results_not_latest(self):
        for module in (self.logger, self.wrapper):
            with self.subTest(module=module.__name__):
                self.assertEqual(module._find_run(self.sdk.client, "17", RUN_UUID).info.run_id, RUN_ID)
                self.assertEqual(self.sdk.client.searches[-1],
                                 (["17"], f"tags.run_uuid = '{RUN_UUID}'", 2, None))
                self.sdk.client.runs *= 2
                with self.assertRaisesRegex(RuntimeError, "Ambiguous"):
                    module._find_run(self.sdk.client, "17", RUN_UUID)
                self.sdk.client.runs = self.sdk.client.runs[:1]

    def test_uuid_regex_rejects_filter_injection_before_search(self):
        for module in (self.logger, self.wrapper):
            for invalid in ("", "short", "a' OR tags.x = 'b", RUN_UUID + "\n"):
                with self.assertRaises(ValueError):
                    module._find_run(self.sdk.client, "17", invalid)
            module._find_run(self.sdk.client, "17", "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
        self.assertEqual(len(self.sdk.client.searches), 2)

    def test_lookup_missing_never_falls_back_to_latest(self):
        self.sdk.client.runs = []
        with self.assertRaises(RuntimeError):
            self.logger._find_run(self.sdk.client, "17", RUN_UUID)
        self.assertIsNone(self.wrapper._find_run(self.sdk.client, "17", RUN_UUID))
        with self.assertRaisesRegex(RuntimeError, "not found"):
            self.final()

    def test_upload_and_watcher_configuration_are_read_only(self):
        self.sdk.client.experiment = None
        for configure in (self.logger._configure, self.wrapper._configure_mlflow):
            with self.assertRaisesRegex(RuntimeError, "not found"):
                configure("missing")
        self.assertEqual(self.sdk.set_experiment_calls, [])
        with redirect_stdout(io.StringIO()) as output:
            self.logger.check_connection("explicit-connectivity-test")
        self.assertEqual(self.sdk.set_experiment_calls, ["explicit-connectivity-test"])
        self.assertEqual(self.sdk.client.created, 1)
        self.assertNotIn("PRIVATE_TOKEN", output.getvalue())

    def test_missing_credentials_fail_without_echoing_values(self):
        del os.environ["MLFLOW_TRACKING_PASSWORD"]
        for configure in (self.logger._configure, self.wrapper._configure_mlflow):
            with self.assertRaisesRegex(RuntimeError, "MLFLOW_TRACKING_PASSWORD") as caught:
                configure("baseline")
            self.assertNotIn("PRIVATE_TOKEN", str(caught.exception))

    def test_metrics_read_runtime_yaml_and_only_label_known_final(self):
        (self.output / "all_results.json").write_text('{"eval_loss": 1.25, "flag": true}')
        for value in ("false", "true", "null", '"false"'):
            self.config.write_text(f"load_best_model_at_end: {value}\n")
            role = "final_output" if value == "false" else "best_or_reloaded_output"
            self.assertEqual(self.logger._final_metrics(self.output, self.config),
                             {f"{role}_eval_loss": 1.25})
        self.assertEqual(self.logger._final_metrics(self.output),
                         {"best_or_reloaded_output_eval_loss": 1.25})

    def test_best_metric_is_separate_from_actual_final_metric(self):
        self.config.write_text("load_best_model_at_end: false\nmetric_for_best_model: eval_loss\n")
        (self.output / "eval_results.json").write_text('{"eval_loss": 1.25}')
        (self.output / "trainer_state.json").write_text('{"best_metric": 0.9}')
        self.assertEqual(self.logger._final_metrics(self.output, self.config),
                         {"final_output_eval_loss": 1.25, "best_validation_eval_loss": 0.9})

    def test_final_upload_mapping_config_artifacts_and_recursive_sizes(self):
        checkpoint = self.checkpoint()
        (self.output / "trainer_state.json").write_text(json.dumps({"best_model_checkpoint": str(checkpoint)}))
        (self.output / "eval_results.json").write_text('{"eval_loss": 1.25}')
        generated = self.root / "generated"
        generated.mkdir()
        (generated / "dataset_info.json").write_text('{"train": {}, "val": {}}')
        split = self.root / "split.json"
        split.write_text('{"seed": 42}')
        os.environ.update(DATASET_DIR=str(generated), SPLIT_MANIFEST=str(split), SPLIT_SEED="42",
                          MLFLOW_TAGS='{"baseline_label": "dev_tt"}', RUN_UUID="wrong-env-uuid")
        with redirect_stdout(io.StringIO()) as captured:
            self.upload()
        metadata = json.loads((self.output / "run_metadata.json").read_text())
        self.assertEqual(metadata["run_uuid"], RUN_UUID)
        self.assertEqual(metadata["mlflow_run_id"], RUN_ID)
        self.assertEqual(metadata["mlflow_experiment_id"], "17")
        self.assertIn(RUN_UUID, metadata["run_name"])
        self.assertEqual(metadata["output_model_role"], "final_output")
        self.assertEqual(metadata["latest_checkpoint"], "checkpoint-5")
        self.assertEqual(metadata["best_checkpoint"], str(checkpoint))
        self.assertEqual(metadata["output_artifact_path"], "training")
        self.assertNotIn("stopped_intentionally_at_global_step", metadata)
        self.assertEqual(metadata["artifact_verification_level"], "sizes_only")
        self.assertEqual(metadata["tags"]["experiment_variant"], "baseline")
        self.assertEqual(metadata["tags"]["baseline_label"], "dev_tt")
        self.assertEqual(metadata["tags"]["split_seed"], "42")
        self.assertEqual(metadata["tags"]["split_manifest_sha256"], self.logger._sha256(split))
        self.assertIn("configuration/generated_registry/dataset_info.json", self.sdk.client.files)
        self.assertIn("configuration/split_manifest/split.json", self.sdk.client.files)
        self.assertIn((RUN_ID, "training/checkpoint-5"), self.sdk.client.listings)
        self.assertEqual(self.sdk.metrics, {"final_output_eval_loss": 1.25})
        self.assertEqual(self.sdk.client.tags["artifact.verification_status"], "verified")
        self.assertNotIn("PRIVATE_TOKEN", captured.getvalue())
        self.assertIn("https://example.org/repo.mlflow/#/experiments/17/runs/", captured.getvalue())
        self.assertIn("sizes_only", captured.getvalue())

    def test_intentional_stop_upload_preserves_horizon_and_hashes_checkpoint(self):
        checkpoint = self.checkpoint("checkpoint-250")
        trainer_state = {"global_step": 250, "epoch": .984, "num_train_epochs": 2, "max_steps": 508}
        (self.output / "trainer_state.json").write_text(json.dumps(trainer_state))
        os.environ["STOP_AT_GLOBAL_STEP"] = "250"
        self.upload()
        metadata = json.loads(self.sdk.client.files["training/run_metadata.json"])
        self.assertEqual(metadata["stopped_intentionally_at_global_step"], 250)
        self.assertEqual(metadata["final_global_step"], 250)
        self.assertEqual(metadata["training_horizon_epochs"], 2)
        self.assertEqual(metadata["training_horizon_steps"], 508)
        self.assertEqual(metadata["latest_checkpoint"], "checkpoint-250")
        manifest = json.loads(self.sdk.client.files["training/artifact_manifest.json"])
        files = {item["path"]: item["sha256"] for item in manifest["files"]}
        self.assertEqual(files["checkpoint-250/optimizer.pt"], self.logger._sha256(checkpoint / "optimizer.pt"))

    def test_intentional_stop_upload_refuses_checkpoint_from_different_step(self):
        self.checkpoint("checkpoint-249")
        (self.output / "trainer_state.json").write_text('{"global_step": 250, "max_steps": 508}')
        os.environ["STOP_AT_GLOBAL_STEP"] = "250"
        with self.assertRaisesRegex(RuntimeError, "requested checkpoint"):
            self.upload()
        self.assertEqual(self.sdk.uploads, [])

    def test_nonempty_stale_artifacts_do_not_verify_upload(self):
        self.sdk.client.files["training/old.bin"] = b"old"
        self.sdk.discard_uploads = True
        with self.assertRaisesRegex(RuntimeError, "verification failed"):
            self.upload()
        self.assertNotIn("artifact.verification_status", self.sdk.client.tags)

    def test_missing_nested_file_or_wrong_size_fail_verification(self):
        self.checkpoint()
        target = "training/checkpoint-5/optimizer.pt"
        for issue in ("missing", "size", "unknown_size"):
            with self.subTest(issue=issue):
                self.sdk.client.omit = {target} if issue == "missing" else set()
                self.sdk.client.sizes = {target: 999 if issue == "size" else None}
                with self.assertRaisesRegex(RuntimeError, "verification failed"):
                    self.upload()

    def test_optional_historical_configs_are_explicitly_skipped(self):
        with redirect_stdout(io.StringIO()) as captured:
            self.upload()
        self.assertIn("optional historical configuration: requirements.txt", captured.getvalue())
        self.assertIn("optional historical configuration: dataset_info.json", captured.getvalue())

    def test_required_configuration_missing_fails_before_upload(self):
        for name, path in (("SPLIT_MANIFEST", self.root / "absent.json"),
                           ("DATASET_DIR", self.root / "absent-dir")):
            with mock.patch.dict(os.environ, {name: str(path)}):
                with self.assertRaises(FileNotFoundError):
                    self.upload()
        self.config.unlink()
        with self.assertRaises(FileNotFoundError):
            self.upload()
        self.assertEqual(self.sdk.uploads, [])

    def test_missing_notebook_and_configured_registry_are_required(self):
        self.notebook.unlink()
        with self.assertRaises(FileNotFoundError):
            self.upload()
        self.notebook.write_text("{}")
        self.config.write_text("dataset_dir: missing\nload_best_model_at_end: false\n")
        with self.assertRaises(FileNotFoundError):
            self.upload()
        self.assertEqual(self.sdk.uploads, [])

    def test_logger_cli_suppresses_remote_exception_credentials(self):
        self.sdk.upload_error = RuntimeError(SECRET_ERROR)
        argv = ["dagshub_logger.py", "upload", "--experiment", "baseline", "--run-uuid", RUN_UUID,
                "--config", str(self.config), "--output-dir", str(self.output),
                "--project-dir", str(self.root), "--notebook", str(self.notebook)]
        with mock.patch.object(sys, "argv", argv), \
             mock.patch("subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="test-value")), \
             redirect_stderr(io.StringIO()) as err:
            with self.assertRaises(SystemExit) as caught:
                runpy.run_path(str(ROOT / "scripts/dagshub_logger.py"), run_name="__main__")
        self.assertEqual(caught.exception.code, 1)
        self.assertNotIn("PRIVATE_TOKEN", err.getvalue())
        self.assertNotIn("https://", err.getvalue())

    def test_final_sync_raises_for_upload_failure_or_incomplete_checkpoint(self):
        checkpoint = self.checkpoint()
        self.sdk.upload_error = RuntimeError(SECRET_ERROR)
        with self.assertRaises(RuntimeError):
            self.final()
        self.sdk.upload_error = None
        (checkpoint / "optimizer.pt").unlink()
        with self.assertRaisesRegex(RuntimeError, "not ready"):
            self.final()

    def test_state_is_bound_to_real_run_and_does_not_claim_verification(self):
        self.checkpoint()
        state_path = self.root / "state.json"
        state_path.write_text('{"uploaded": {"checkpoint-5": {"old": true}}}')
        self.final()
        state = json.loads(state_path.read_text())
        self.assertEqual(state["identity"]["run_id"], RUN_ID)
        self.assertEqual(state["uploaded"]["checkpoint-5"]["verification"], "not_verified")
        self.assertEqual(len(self.sdk.uploads), 2)
        self.final()
        self.assertEqual(len(self.sdk.uploads), 2)
        self.sdk.client.runs[0].info.run_id = "c" * 32
        self.final()
        self.assertEqual(len(self.sdk.uploads), 4)

    def test_sync_publishes_manifest_after_files_without_changing_run_lifecycle(self):
        checkpoint = self.checkpoint()
        (checkpoint / "nested").mkdir()
        (checkpoint / "nested/metadata").write_bytes(b"metadata")
        with mock.patch.object(self.sdk, "start_run", side_effect=AssertionError("Trainer owns the run")):
            self.final()
        target = "checkpoints/checkpoint-5"
        manifest_path = "checkpoint_manifests/checkpoint-5.json"
        self.assertEqual([folder for _, folder in self.sdk.uploads], [target, "checkpoint_manifests"])
        manifest = json.loads(self.sdk.client.files[manifest_path])
        self.assertEqual(manifest["artifact_path"], target)
        self.assertEqual(manifest["run_id"], RUN_ID)
        self.assertEqual(manifest["file_count"], len(list(checkpoint.rglob("*"))) - 1)
        for entry in manifest["files"]:
            payload = self.sdk.client.files["checkpoints/" + entry["path"]]
            self.assertEqual(entry["size"], len(payload))
            self.assertEqual(entry["sha256"], hashlib.sha256(payload).hexdigest())
        state = json.loads((self.root / "state.json").read_text())
        self.assertEqual(state["uploaded"]["checkpoint-5"]["manifest_artifact"], manifest_path)

    def test_a_noweight_retains_and_syncs_full_best_and_off_interval_final(self):
        from transformers import Trainer, TrainerControl, TrainerState
        from transformers.trainer_callback import DefaultFlowCallback
        from transformers.trainer_utils import IntervalStrategy, SaveStrategy

        for final_step, best_step, previous_step in ((338, 250, 300), (676, 500, 650)):
            with self.subTest(final_step=final_step):
                output = self.output / str(final_step)
                best = output / f"checkpoint-{best_step}"
                latest = output / f"checkpoint-{final_step}"
                previous = output / f"checkpoint-{previous_step}"
                args = SimpleNamespace(save_total_limit=2, load_best_model_at_end=True,
                                       logging_strategy=IntervalStrategy.STEPS,
                                       eval_strategy=IntervalStrategy.STEPS,
                                       save_strategy=SaveStrategy.STEPS, eval_delay=0)
                state = TrainerState(global_step=final_step, max_steps=final_step, logging_steps=1,
                                     eval_steps=50, save_steps=50, best_model_checkpoint=str(best))
                control = DefaultFlowCallback().on_step_end(args, state, TrainerControl())
                self.assertTrue(control.should_save)
                self.assertFalse(control.should_evaluate)
                required_files = ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt",
                                  "scheduler.pt", "scaler.pt", "rng_state_0.pth", "rng_state_1.pth")
                for checkpoint in (best, previous, latest):
                    checkpoint.mkdir(parents=True)
                    for name in required_files:
                        (checkpoint / name).write_bytes(name.encode())
                    state.save_to_json(str(checkpoint / "trainer_state.json"))

                trainer = SimpleNamespace(args=args, state=state)
                trainer._sorted_checkpoints = partial(Trainer._sorted_checkpoints, trainer)
                Trainer._rotate_checkpoints(trainer, output_dir=output)
                self.assertEqual(set(output.iterdir()), {best, latest})
                self.wrapper._final_sync("baseline", RUN_UUID, output, "checkpoints", output / "sync.json",
                                         0, True, required_final_step=final_step)
                for checkpoint in (best, latest):
                    manifest = json.loads(self.sdk.client.files[f"checkpoint_manifests/{checkpoint.name}.json"])
                    files = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
                    for name in (*required_files, "trainer_state.json"):
                        path = f"{checkpoint.name}/{name}"
                        self.assertEqual(self.sdk.client.files[f"checkpoints/{path}"], (checkpoint / name).read_bytes())
                        self.assertEqual(files[path], hashlib.sha256((checkpoint / name).read_bytes()).hexdigest())

    def test_required_final_sync_fails_for_missing_latest_or_incomplete_amp_state(self):
        def final():
            self.wrapper._final_sync("baseline", RUN_UUID, self.output, "checkpoints",
                                     self.root / "state.json", 0, True, required_final_step=338)

        with self.assertRaisesRegex(FileNotFoundError, "checkpoint-338"):
            final()
        checkpoint = self.checkpoint("checkpoint-338")
        (checkpoint / "trainer_state.json").write_text('{"global_step": 338, "best_model_checkpoint": "checkpoint-300"}')
        for name in ("adapter_config.json", "rng_state_0.pth", "rng_state_1.pth"):
            (checkpoint / name).write_bytes(b"test")
        with self.assertRaisesRegex(RuntimeError, "scaler.pt"):
            final()
        self.assertFalse(self.sdk.client.files)
        (checkpoint / "scaler.pt").write_bytes(b"test")
        with self.assertRaisesRegex(FileNotFoundError, "checkpoint-300"):
            final()

    def test_failed_manifest_upload_is_not_acknowledged_and_is_retried(self):
        self.checkpoint()
        with mock.patch.object(self.sdk.client, "log_artifact", side_effect=RuntimeError("manifest unavailable")):
            with self.assertRaisesRegex(RuntimeError, "manifest unavailable"):
                self.final()
        self.assertFalse((self.root / "state.json").exists())
        self.assertNotIn("checkpoint_manifests/checkpoint-5.json", self.sdk.client.files)
        self.final()
        self.assertIn("checkpoint_manifests/checkpoint-5.json", self.sdk.client.files)

    def test_checkpoint_mutated_during_upload_is_not_published_with_stale_hashes(self):
        checkpoint = self.checkpoint()
        upload = self.sdk.client.log_artifacts

        def changing_upload(*args, **kwargs):
            upload(*args, **kwargs)
            (checkpoint / "optimizer.pt").write_bytes(b"rewritten")

        with mock.patch.object(self.sdk.client, "log_artifacts", side_effect=changing_upload):
            with self.assertRaisesRegex(RuntimeError, "changed during upload"):
                self.final()
        self.assertFalse((self.root / "state.json").exists())
        self.assertNotIn("checkpoint_manifests/checkpoint-5.json", self.sdk.client.files)

    def test_background_errors_are_collected_without_credentials(self):
        stop = threading.Event()
        failures = []

        def fail(**kwargs):
            stop.set()
            raise RuntimeError(SECRET_ERROR)

        with mock.patch.object(self.wrapper, "_sync_checkpoints_once", side_effect=fail), \
             redirect_stdout(io.StringIO()) as captured:
            self.wrapper._watch_checkpoints("baseline", RUN_UUID, self.output, "checkpoints",
                                            self.root / "state.json", 0, 0, True, stop, failures)
        self.assertEqual(failures, [{"stage": "background_upload", "error_type": "RuntimeError"}])
        self.assertNotIn("PRIVATE_TOKEN", captured.getvalue())

    def test_background_setup_failure_is_collected_without_credentials(self):
        failures = []
        with mock.patch.object(self.wrapper, "_configure_mlflow", side_effect=RuntimeError(SECRET_ERROR)), \
             redirect_stdout(io.StringIO()) as captured:
            self.wrapper._watch_checkpoints("baseline", RUN_UUID, self.output, "checkpoints",
                                            self.root / "state.json", 0, 0, True,
                                            threading.Event(), failures)
        self.assertEqual(failures, [{"stage": "background_setup", "error_type": "RuntimeError"}])
        self.assertNotIn("PRIVATE_TOKEN", captured.getvalue())

    def test_existing_cli_arguments_remain_sufficient(self):
        argv = ["dagshub_logger.py", "upload", "--experiment", "baseline", "--run-uuid", RUN_UUID,
                "--config", str(self.config), "--output-dir", str(self.output),
                "--project-dir", str(self.root), "--notebook", str(self.notebook)]
        with mock.patch.object(sys, "argv", argv):
            self.assertEqual(self.logger.parse_args().run_uuid, RUN_UUID)
        argv = ["train_with_checkpoint_sync.py", "--experiment", "baseline", "--run-uuid", RUN_UUID,
                "--yaml-config", str(self.config), "--output-dir", str(self.output),
                "--project-dir", str(self.root), "--run-name", "baseline", "seed=42"]
        with mock.patch.object(sys, "argv", argv):
            args = self.wrapper.parse_args()
        self.assertEqual(args.train_overrides, ["seed=42"])
        self.assertEqual(args.run_uuid, RUN_UUID)
        self.assertEqual(args.post_stop_timeout_seconds, 90 * 60)
        self.assertEqual(args.post_final_timeout_seconds, 45 * 60)
        self.assertEqual(args.sync_timeout_seconds, 45 * 60)

    def test_relative_artifact_paths_resolve_against_project_dir(self):
        self.logger.upload_run("baseline", RUN_UUID, Path("runtime.yaml"), Path("output"),
                               self.root, Path("source.ipynb"))
        self.assertIn("configuration/runtime.yaml", self.sdk.client.files)
        self.assertIn("training/run_metadata.json", self.sdk.client.files)

    def wrapper_args(self):
        return SimpleNamespace(experiment="baseline", run_uuid=RUN_UUID, yaml_config=str(self.config),
                               output_dir=str(self.output), project_dir=self.root, run_name="baseline",
                               artifact_path="checkpoints", interval_seconds=1, stable_seconds=0,
                               state_path=str(self.root / "state.json"), allow_model_only=False,
                               post_stop_timeout_seconds=60, post_final_timeout_seconds=60, sync_timeout_seconds=60,
                               train_overrides=[])

    def run_main(self, returncode=0, final_effect=None, spawn_error=None, watcher_error=False):
        events = []
        args = self.wrapper_args()
        process = mock.Mock()
        process.wait.return_value = returncode

        class FakeThread:
            def __init__(self, target, kwargs, daemon, args=()):
                self.target, self.args, self.kwargs = target, args, kwargs
                self.is_watcher = "stop_event" in kwargs

            def start(self):
                if self.is_watcher:
                    events.append("watcher_start")
                    if watcher_error:
                        self.kwargs["failures"].append({"stage": "background_upload", "error_type": "RuntimeError"})
                elif self.target is not self_outer.wrapper._tee_training_log:
                    self.target(*self.args, **self.kwargs)

            def join(self, timeout):
                if self.is_watcher:
                    self_outer.assertTrue(self.kwargs["stop_event"].is_set())
                    events.append("watcher_joined")

            def is_alive(self):
                return False

        self_outer = self
        effect = iter(final_effect or [None])

        def final(**kwargs):
            self.assertEqual(events[-1] if "final" not in events else events[-2], "watcher_joined")
            events.append("final")
            result = next(effect)
            if result is not None:
                raise result

        with mock.patch.object(self.wrapper, "parse_args", return_value=args), \
             mock.patch.object(self.wrapper.signal, "signal"), \
             mock.patch.object(self.wrapper.threading, "Thread", FakeThread), \
             mock.patch.object(self.wrapper.subprocess, "Popen", return_value=process,
                               side_effect=spawn_error) as popen, \
             mock.patch.object(self.wrapper, "_final_sync", side_effect=final), \
             redirect_stdout(io.StringIO()) as captured:
            with self.assertRaises(SystemExit) as caught:
                self.wrapper.main()
        self.assertNotIn("PRIVATE_TOKEN", captured.getvalue())
        return caught.exception.code, events, popen, captured.getvalue()

    def test_wrapper_fully_joins_before_final_writer_and_passes_full_uuid(self):
        code, events, popen, output = self.run_main(watcher_error=True)
        self.assertEqual(code, 0)
        self.assertEqual(events, ["watcher_start", "watcher_joined", "final"])
        command = popen.call_args.args[0]
        self.assertEqual(command[-1], f"run_name=baseline-{RUN_UUID}")
        tags = json.loads(popen.call_args.kwargs["env"]["MLFLOW_TAGS"])
        self.assertEqual(tags["run_uuid"], RUN_UUID)
        self.assertEqual(tags["experiment_variant"], "baseline")
        state = json.loads((self.root / "state.json").read_text())
        self.assertEqual(state["final_sync"]["verification"], "not_verified")
        self.assertEqual(state["sync_failures"][0]["stage"], "background_upload")

    def test_wrapper_retry_is_deterministic_and_records_recovered_failure(self):
        code, events, _, _ = self.run_main(final_effect=[RuntimeError(SECRET_ERROR), None])
        self.assertEqual(code, 0)
        self.assertEqual(events.count("final"), 2)
        state = json.loads((self.root / "state.json").read_text())
        self.assertEqual(state["final_sync"], {"status": "submitted", "verification": "not_verified", "attempts": 2})
        self.assertEqual(len(state["sync_failures"]), 1)

    def test_wrapper_final_failure_nonzero_and_preserves_training_failure(self):
        for child_code, expected in ((0, 1), (9, 9)):
            with self.subTest(child_code=child_code):
                code, events, _, _ = self.run_main(child_code, [RuntimeError(SECRET_ERROR)] * 2)
                self.assertEqual(code, expected)
                self.assertEqual(events.count("final"), 2)

    def test_spawn_failure_does_not_start_watcher_or_attempt_final_sync(self):
        code, events, _, _ = self.run_main(spawn_error=OSError(SECRET_ERROR))
        self.assertEqual(code, 127)
        self.assertEqual(events, [])
        state = json.loads((self.root / "state.json").read_text())
        self.assertEqual(state["final_sync"]["attempts"], 0)

    def test_blocked_background_or_final_sync_is_bounded_without_parallel_writer(self):
        for stage in ("background", "final"):
            with self.subTest(stage=stage):
                release = threading.Event()
                finished = threading.Event()
                sentinel = self.root / "state.json"
                sentinel.write_text('{"uploaded": {"prior": true}}')

                def block(**kwargs):
                    try:
                        release.wait(5)
                    finally:
                        finished.set()

                args = self.wrapper_args()
                args.sync_timeout_seconds = .1
                process = mock.Mock()
                process.wait.return_value = 0
                with mock.patch.object(self.wrapper, "parse_args", return_value=args), \
                     mock.patch.object(self.wrapper.signal, "signal"), \
                     mock.patch.object(self.wrapper.subprocess, "Popen", return_value=process), \
                     mock.patch.object(self.wrapper, "_watch_checkpoints", side_effect=block if stage == "background" else None), \
                     mock.patch.object(self.wrapper, "_final_sync", side_effect=block if stage == "final" else None) as final:
                    try:
                        with self.assertRaisesRegex(self.wrapper.SyncTimeout, stage):
                            self.wrapper.main()
                        self.assertEqual(final.call_count, int(stage == "final"))
                        self.assertEqual(sentinel.read_text(), '{"uploaded": {"prior": true}}')
                    finally:
                        release.set()
                        self.assertTrue(finished.wait(1))

    def test_wrapper_cli_exits_124_even_with_stalled_sdk_executor(self):
        launcher = self.root / "llamafactory-cli"
        launcher.write_text("#!/bin/sh\nexit 0\n")
        launcher.chmod(0o700)
        (self.root / "mlflow.py").write_text(
            "from concurrent.futures import ThreadPoolExecutor\n"
            "from time import sleep\n"
            "def set_tracking_uri(uri):\n"
            "    pass\n"
            "class MlflowClient:\n"
            "    def get_experiment_by_name(self, name):\n"
            "        with ThreadPoolExecutor() as pool:\n"
            "            return pool.submit(sleep, 60).result()\n"
        )
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "scripts/train_with_checkpoint_sync.py"),
             "--experiment", "cpu", "--run-uuid", RUN_UUID, "--yaml-config", str(self.config),
             "--output-dir", str(self.output), "--project-dir", str(self.root),
             "--run-name", "cpu", "--sync-timeout-seconds", "1"],
            cwd=self.root, env=dict(self.env, PATH=str(self.root), PYTHONPATH=str(self.root)),
            text=True, capture_output=True, timeout=5,
        )
        self.assertEqual(result.returncode, 124, result.stderr)
        self.assertIn("Timed out waiting for background checkpoint sync", result.stderr)

    def test_post_stop_timeout_terminates_a_real_stalled_cpu_process(self):
        marker = self.root / "stop-step-reached"
        final_marker = self.root / "final-step-reached"
        final_step_hang = threading.Event()
        marker.touch()
        with subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                              start_new_session=True) as process:
            try:
                self.assertEqual(self.wrapper._wait_for_training(
                    process, marker, .05, [], final_marker=final_marker, final_step_hang=final_step_hang
                ), 124)
                self.assertIsNotNone(process.poll())
                self.assertFalse(final_step_hang.is_set())
            finally:
                if process.poll() is None:
                    process.kill()

    def run_fake_launcher(self, source):
        launcher = self.root / "llamafactory-cli"
        launcher.write_text(f"#!{sys.executable}\n" + dedent(source), encoding="utf-8")
        launcher.chmod(0o700)
        args = self.wrapper_args()
        args.post_final_timeout_seconds = .25
        args.interval_seconds = .05
        with mock.patch.dict(os.environ, {"PATH": str(self.root)}), \
             mock.patch.object(self.wrapper, "TRAIN_POLL_SECONDS", .02), \
             mock.patch.object(self.wrapper, "STACK_DUMP_GRACE_SECONDS", .2), \
             mock.patch.object(self.wrapper, "parse_args", return_value=args), \
             mock.patch.object(self.wrapper.signal, "signal"), \
             redirect_stdout(io.StringIO()) as captured, self.assertRaises(SystemExit) as caught:
            self.wrapper.main()
        return caught.exception.code, captured.getvalue()

    @unittest.skipUnless(sys.platform == "linux", "requires Linux rank process discovery")
    def test_final_step_hang_dumps_two_ranks_and_syncs_checkpoint_and_logs(self):
        (self.root / "fake_rank.py").write_text(dedent("""
            import faulthandler
            import os
            from pathlib import Path
            import signal
            import sys
            import time

            path = Path(os.environ["TRAIN_STACK_DUMP_DIR"]) / f"rank{sys.argv[1]}"
            with path.with_suffix(".txt").open("w") as stream:
                faulthandler.register(signal.SIGUSR1, all_threads=True, file=stream)
                print(f"stdout from rank {sys.argv[1]}")
                print(f"stderr from rank {sys.argv[1]}", file=sys.stderr)
                path.with_suffix(".pid").write_text(str(os.getpid()))
                time.sleep(4)
        """), encoding="utf-8")
        self.checkpoint("checkpoint-138")
        start = time.monotonic()
        with mock.patch.object(self.wrapper.os, "kill", wraps=os.kill) as kill, \
             mock.patch.object(self.wrapper, "_sync_checkpoints_once",
                               wraps=self.wrapper._sync_checkpoints_once) as sync:
            code, output = self.run_fake_launcher("""
                import os
                from pathlib import Path
                import subprocess
                import sys
                import time

                ranks = [subprocess.Popen([sys.executable, "fake_rank.py", str(rank)], start_new_session=True)
                         for rank in range(2)]
                dump_dir = Path(os.environ["TRAIN_STACK_DUMP_DIR"])
                while len(list(dump_dir.glob("rank*.pid"))) < 2 and all(rank.poll() is None for rank in ranks):
                    time.sleep(.01)
                Path(os.environ["TRAIN_FINAL_MARKER"]).touch()
                for rank in ranks:
                    rank.wait()
            """)
        self.assertEqual(code, 124)
        self.assertLess(time.monotonic() - start, 4)
        self.assertIn("Trainer did not exit within 0.25s after the final step (exit 124).", output)
        self.assertEqual(self.sdk.client.terminations, [(RUN_ID, "FAILED")])
        self.assertEqual(self.sdk.client.tags["train.exit_reason"], "hang_after_final_step")
        self.assertTrue(any(call.kwargs.get("final") for call in sync.call_args_list))
        self.assertIn("checkpoints/checkpoint-138/optimizer.pt", self.sdk.client.files)

        log_dir = self.output.parent / "train_logs"
        log_bytes = (log_dir / f"{RUN_UUID}.log").read_bytes()
        self.assertEqual(self.sdk.client.files[f"logs/{RUN_UUID}.log"], log_bytes)
        self.assertEqual(len([path for _, path in self.sdk.uploads if path == "logs"]), 2)
        for rank in range(2):
            self.assertIn(f"stdout from rank {rank}", output)
            self.assertIn(f"stderr from rank {rank}", output)
            self.assertIn(f"stdout from rank {rank}".encode(), log_bytes)
            self.assertIn(f"stderr from rank {rank}".encode(), log_bytes)
            self.assertIn(b"fake_rank.py", self.sdk.client.files[f"logs/stacks/rank{rank}.txt"])
            pid = int((log_dir / f"{RUN_UUID}-stacks" / f"rank{rank}.pid").read_text())
            kill.assert_any_call(pid, signal.SIGUSR1)
        self.assertEqual(kill.call_count, 2)

    def test_exiting_after_final_marker_does_not_kill_or_fail_run(self):
        with mock.patch.object(self.wrapper, "_terminate_training") as terminate, \
             mock.patch.object(self.wrapper, "_dump_rank_stacks") as dump:
            code, output = self.run_fake_launcher("""
                import os
                from pathlib import Path
                import time

                Path(os.environ["TRAIN_FINAL_MARKER"]).touch()
                time.sleep(.1)
                print("normal final output")
            """)
        self.assertEqual(code, 0)
        self.assertIn("Final step reached", output)
        self.assertIn("normal final output", output)
        self.assertIn(b"normal final output", self.sdk.client.files[f"logs/{RUN_UUID}.log"])
        terminate.assert_not_called()
        dump.assert_not_called()
        self.assertEqual(self.sdk.client.terminations, [])
        self.assertNotIn("train.exit_reason", self.sdk.client.tags)

    def test_periodic_log_upload_is_rate_limited_and_retries_failures(self):
        log_path = self.root / "train.log"
        log_path.write_text("progress")
        stop = threading.Event()
        failures = []
        ticks = iter((0, 0, 599, 600, 601))

        def tick():
            now = next(ticks)
            if now == 601:
                stop.set()
            return now

        with mock.patch.object(self.wrapper.time, "monotonic", side_effect=tick), \
             mock.patch.object(self.sdk.client, "log_artifact", side_effect=[RuntimeError(SECRET_ERROR), None]) as call, \
             redirect_stdout(io.StringIO()) as captured:
            self.wrapper._watch_checkpoints("baseline", RUN_UUID, self.output, "checkpoints",
                                            self.root / "state.json", 0, 0, True, stop, failures, log_path)
        self.assertEqual(call.call_count, 2)
        self.assertEqual(failures, [{"stage": "background_logs", "error_type": "RuntimeError"}])
        self.assertNotIn("PRIVATE_TOKEN", captured.getvalue())

    def test_final_log_upload_failure_does_not_change_successful_exit(self):
        with mock.patch.object(self.sdk.client, "log_artifact", side_effect=RuntimeError(SECRET_ERROR)):
            code, _, _, output = self.run_main()
        self.assertEqual(code, 0)
        state = json.loads((self.root / "state.json").read_text())
        self.assertEqual(state["sync_failures"], [{"stage": "final_logs", "error_type": "RuntimeError"}])
        self.assertIn("Training log upload failed (RuntimeError)", output)

    @unittest.skipUnless(sys.platform == "linux", "requires Linux process group discovery")
    def test_timeout_signals_a_rank_in_its_own_process_group(self):
        import signal

        rank = "import os, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print(os.getpid(), flush=True); time.sleep(60)"
        launcher = ("import signal, subprocess, sys; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                    "subprocess.run([sys.executable, '-c', sys.argv[1]], start_new_session=True)")
        with subprocess.Popen([sys.executable, "-c", launcher, rank], start_new_session=True,
                              stdout=subprocess.PIPE, text=True) as process:
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    self.assertTrue(selector.select(3), "CPU rank did not start")
                rank_pid = int(process.stdout.readline())
                with mock.patch.object(self.wrapper.os, "killpg", wraps=os.killpg) as kill_group:
                    self.wrapper._terminate_training(process, grace_seconds=.05)
                kill_group.assert_any_call(rank_pid, signal.SIGTERM)
                kill_group.assert_any_call(rank_pid, signal.SIGKILL)
                self.assertEqual(process.poll(), -signal.SIGKILL)
            finally:
                if process.poll() is None:
                    self.wrapper._terminate_training(process, grace_seconds=.05)


if __name__ == "__main__":
    unittest.main()

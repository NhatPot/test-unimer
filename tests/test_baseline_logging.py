"""CPU-only coverage for the legacy baseline logging/sync entry points."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import threading
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
        pass

    def delete_run(self, run_id):
        pass


class FakeMlflow(ModuleType):
    def __init__(self):
        super().__init__("mlflow")
        self.client = FakeClient()
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
        self.assertEqual(len(self.sdk.uploads), 1)
        self.final()
        self.assertEqual(len(self.sdk.uploads), 1)
        self.sdk.client.runs[0].info.run_id = "c" * 32
        self.final()
        self.assertEqual(len(self.sdk.uploads), 2)

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
                               train_overrides=[])

    def run_main(self, returncode=0, final_effect=None, spawn_error=None, watcher_error=False):
        events = []
        args = self.wrapper_args()
        process = mock.Mock()
        process.wait.return_value = returncode

        class FakeThread:
            def __init__(self, target, kwargs, daemon):
                self.kwargs = kwargs

            def start(self):
                events.append("watcher_start")
                if watcher_error:
                    self.kwargs["failures"].append({"stage": "background_upload", "error_type": "RuntimeError"})

            def join(self):
                self_outer.assertTrue(self.kwargs["stop_event"].is_set())
                events.append("watcher_joined")

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


if __name__ == "__main__":
    unittest.main()

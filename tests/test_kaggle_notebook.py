"""Contracts for the simplified notebook. Never execute setup/install/train."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import yaml
from scripts.prepare_baseline_data import prepare_from_rows
from test_prepare_baseline_data import rows

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "uni-mumer-kaggle-dagshub-banmoi.ipynb"


class KaggleNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.sources = ["".join(c["source"]) if isinstance(c["source"], list)
                       else c["source"] for c in cls.notebook["cells"]]

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def namespace(self):
        with mock.patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()):
            namespace = {}
            exec(compile(self.sources[1], "cell-1", "exec"), namespace)
        return namespace

    def runtime_namespace(self):
        project = self.root / "project"
        (project / "scripts").mkdir(parents=True)
        (project / "train").mkdir()
        for name in ("scripts/runtime_yaml.py", "scripts/prepare_baseline_data.py",
                     "scripts/checkpoint_contract.py", "train/Uni-MuMER-train.yaml", "train/dataset_info.json"):
            shutil.copyfile(ROOT / name, project / name)
        namespace = self.namespace()
        namespace.update(PROJECT_DIR=str(project), PYTHON=sys.executable,
                         RUNTIME_YAML_CONFIG=str(self.root / "runtime.yaml"),
                         OUTPUT_DIR=str(self.root / "output"), BASELINE_DATA_ROOT=str(self.root / "data"))
        registry = json.loads((project / "train/dataset_info.json").read_text())
        tree = ast.parse(self.sources[6])
        settings = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "YAML_OVERRIDES" for t in n.targets))
        names = ast.literal_eval(settings.values[0]).split(", ")
        prepared = Path(namespace["BASELINE_DATA_ROOT"]) / "n8000-v0.1-s42"
        prepare_from_rows(prepared, {name:rows() for name in names}, {name:registry[name] for name in names}, max_samples=8000)
        return namespace

    def run_runtime(self, namespace):
        with mock.patch.dict(os.environ, MLFLOW_TAGS="{}"), contextlib.redirect_stdout(io.StringIO()):
            exec(compile(self.sources[6], "cell-6", "exec"), namespace)
            return dict(os.environ)

    def bash(self, index, **environment):
        env = dict(os.environ, RUN_TRAINING="0", PATH="/nonexistent")
        env.update(environment)
        return subprocess.run(["/bin/bash", "-c", self.sources[index].split("\n", 1)[1]],
                              env=env, text=True, capture_output=True, timeout=5)

    def test_notebook_structure_and_no_historical_outputs(self):
        self.assertEqual(len(self.notebook["cells"]), 12)
        self.assertEqual(self.notebook["nbformat"], 4)
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_python_bash_and_embedded_python_parse(self):
        for index, cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = self.sources[index]
            with self.subTest(index=index):
                if source.startswith("%%bash\n"):
                    result = subprocess.run(["/bin/bash", "-n"], input=source.split("\n", 1)[1],
                                            text=True, capture_output=True, timeout=5)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    if "<<'PY'\n" in source:
                        ast.parse(source.split("<<'PY'\n", 1)[1].split("\nPY", 1)[0])
                else:
                    ast.parse(source)

    def test_host_kernel_has_no_project_yaml_or_ml_import(self):
        allowed = {"json", "os", "uuid", "subprocess", "pathlib", "kaggle_secrets"}
        for index in (1, 6):
            for node in ast.walk(ast.parse(self.sources[index])):
                names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                         else [node.module] if isinstance(node, ast.ImportFrom) else [])
                for name in names:
                    self.assertIn(name.split(".")[0], allowed)

    def test_single_training_switch_defaults_off(self):
        namespace = self.namespace()
        self.assertIs(namespace["RUN_TRAINING"], False)
        for obsolete in ("ALLOW_SETUP", "ALLOW_TRAINING", "ALLOW_REMOTE_WRITES",
                         "PIPELINE_REVIEW_COMPLETE", "SOURCE_COMMIT",
                         "MINICONDA_INSTALLER_SHA256"):
            self.assertNotIn(obsolete, namespace)

    def test_session_rerun_retains_uuid_and_clears_stale_config_secret(self):
        with mock.patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()):
            namespace = {}
            exec(self.sources[1], namespace)
            identity, path = namespace["RUN_UUID"], os.environ["PATH"]
            os.environ.update(YAML_CONFIG="stale", MLFLOW_TRACKING_PASSWORD="secret")
            exec(self.sources[1], namespace)
            self.assertEqual(namespace["RUN_UUID"], identity)
            self.assertEqual(os.environ["PATH"], path)
            self.assertEqual(os.environ["YAML_CONFIG"], "")
            self.assertNotIn("MLFLOW_TRACKING_PASSWORD", os.environ)
            self.assertEqual(os.environ["PYTHONNOUSERSITE"], "1")

    def test_default_skips_remote_train_upload_before_commands(self):
        for index in (7, 8, 10):
            with self.subTest(index=index):
                result = self.bash(index)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("RUN_TRAINING=False", result.stdout)

    def test_setup_reuses_existing_conda_and_python_without_network(self):
        conda = self.root / "conda/bin"
        conda.mkdir(parents=True)
        (conda / "conda").write_text("#!/bin/bash\nexit 99\n")
        python = conda / "python"
        python.write_text("#!/bin/bash\nprintf 'Python 3.10 fake\\n'\n")
        python.chmod(0o700)
        (conda / "conda").chmod(0o700)
        result = self.bash(2, CONDA_DIR=str(conda.parent), PYTHON=str(python))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Python 3.10", result.stdout)

    def test_setup_creates_env_without_reinstalling_existing_conda(self):
        conda = self.root / "conda/bin"
        conda.mkdir(parents=True)
        python = self.root / "env/bin/python"
        python.parent.mkdir(parents=True)
        # Fake Conda: record calls; only create a tiny fake Python on create.
        command = conda / "conda"
        command.write_text(
            '#!/bin/bash\nprintf "%s\\n" "$*" >> "$CALLS"\n'
            'if [[ "$1" == create ]]; then\n'
            'printf \'#!/bin/bash\\necho Python-fake\\n\' > "$PYTHON"\n'
            '/bin/chmod +x "$PYTHON"\nfi\n'
        )
        command.chmod(0o700)
        calls = self.root / "calls"
        result = self.bash(2, CONDA_DIR=str(conda.parent), PYTHON=str(python), CALLS=str(calls))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls.read_text().splitlines()), 3)
        self.assertIn("create -n unimumer python=3.10 -y", calls.read_text())
        self.assertNotIn(" -f ", self.sources[2].split("bash /tmp/miniconda.sh", 1)[1].split("\n", 1)[0])

    def test_clone_cell_never_pulls_or_resets_existing_source(self):
        self.assertNotIn("git pull", self.sources[3])
        self.assertNotIn("git reset", self.sources[3])
        self.assertIn("--branch dev_tt", self.sources[3])
        self.assertIn("branch --show-current", self.sources[3])

    def test_project_and_framework_dependencies_resolve_together(self):
        self.assertIn('"$PYTHON" -m pip install -q -r requirements.txt -e train/LLaMA-Factory', self.sources[5])
        self.assertEqual(self.sources[5].count("-m pip install"), 1)
        self.assertIn('"$PYTHON" -m pip check', self.sources[5])

    def test_image_baseline_avoids_decord_and_prometheus_major_drift(self):
        from packaging.requirements import Requirement
        requirements = [Requirement(line) for line in (ROOT / "requirements.txt").read_text().splitlines()
                        if line.strip() and not line.lstrip().startswith("#")]
        packages = {r.name: r for r in requirements}
        self.assertEqual(packages["qwen-vl-utils"].extras, set())
        self.assertNotIn("decord", packages)
        instrumentator = packages["prometheus-fastapi-instrumentator"].specifier
        self.assertIn("7.1.0", instrumentator)
        self.assertNotIn("8.1.0", instrumentator)

    def test_runtime_real_cli_isolated_python_and_t4_values(self):
        namespace = self.runtime_namespace()
        env = self.run_runtime(namespace)
        config = yaml.safe_load(Path(env["YAML_CONFIG"]).read_text())
        self.assertIs(config["fp16"], True)
        self.assertIs(config["bf16"], False)
        self.assertIsNone(config["max_samples"])
        self.assertEqual(config["num_train_epochs"], 1)
        self.assertEqual(config["val_size"], 0)
        self.assertEqual(config["eval_dataset"], "baseline_val")
        self.assertEqual(config["dataset"], "baseline_train")
        self.assertIsNone(config["resume_from_checkpoint"])
        self.assertIsNone(config["adapter_name_or_path"])
        self.assertEqual(config["max_steps"], -1)  # Trainer compares this integer with zero.
        self.assertTrue(config["create_new_adapter"])
        self.assertFalse(config["load_best_model_at_end"])
        self.assertEqual(config["eval_steps"], 50)
        self.assertEqual(config["save_steps"], 50)
        self.assertEqual(config["per_device_train_batch_size"], 2)
        self.assertEqual(config["gradient_accumulation_steps"], 64)
        self.assertEqual(config["lora_rank"], 64)
        self.assertEqual(config["lora_alpha"], 16)
        self.assertEqual(config["quantization_bit"], 4)
        self.assertEqual(config["image_max_pixels"], 262144)
        self.assertEqual(config["cutoff_len"], 2048)
        self.assertFalse(config["overwrite_output_dir"])
        self.assertEqual(env["OUTPUT_DIR"], namespace["OUTPUT_DIR"])

    def test_epoch_baseline_resets_stale_step_limit_to_integer_default(self):
        namespace = self.runtime_namespace()
        path = Path(namespace["PROJECT_DIR"]) / namespace["BASE_YAML_CONFIG"]
        config = yaml.safe_load(path.read_text())
        config["max_steps"] = 700
        path.write_text(yaml.safe_dump(config))
        env = self.run_runtime(namespace)
        actual = yaml.safe_load(Path(env["YAML_CONFIG"]).read_text())
        self.assertIs(type(actual["max_steps"]), int)
        self.assertEqual(actual["max_steps"], -1)
        self.assertEqual(actual["num_train_epochs"], 1)

    def test_same_runtime_rerun_is_idempotent(self):
        namespace = self.runtime_namespace()
        first = self.run_runtime(namespace)
        path = Path(first["YAML_CONFIG"])
        old = path.stat().st_mtime_ns
        self.run_runtime(namespace)
        self.assertEqual(path.stat().st_mtime_ns, old)

    def test_runtime_failure_invalidates_old_yaml_environment(self):
        namespace = self.runtime_namespace()
        namespace["PYTHON"] = "/missing/python"
        with mock.patch.dict(os.environ, YAML_CONFIG="old"):
            with self.assertRaises(FileNotFoundError), contextlib.redirect_stdout(io.StringIO()):
                exec(self.sources[6], namespace)
            self.assertEqual(os.environ["YAML_CONFIG"], "")

    def test_editing_overrides_before_training_can_regenerate_runtime(self):
        namespace = self.runtime_namespace()
        self.run_runtime(namespace)
        runtime = Path(namespace["RUNTIME_YAML_CONFIG"])
        source = self.sources[6].replace('"learning_rate": 1.0e-4', '"learning_rate": 2.0e-4')
        with mock.patch.dict(os.environ, MLFLOW_TAGS="{}"), contextlib.redirect_stdout(io.StringIO()):
            exec(source, namespace)
        self.assertEqual(yaml.safe_load(runtime.read_text())["learning_rate"], 2.0e-4)

    def test_runtime_changes_blocked_after_training_output_exists(self):
        namespace = self.runtime_namespace()
        self.run_runtime(namespace)
        runtime = Path(namespace["RUNTIME_YAML_CONFIG"])
        original = runtime.read_bytes()
        Path(namespace["OUTPUT_DIR"]).mkdir()
        with self.assertRaises(RuntimeError):
            self.run_runtime(namespace)
        self.assertEqual(runtime.read_bytes(), original)

    def test_training_preserves_existing_output(self):
        output = self.root / "existing"
        output.mkdir()
        result = self.bash(8, RUN_TRAINING="1", PROJECT_DIR=str(self.root),
                           YAML_CONFIG="config.yaml", OUTPUT_DIR=str(output))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Output", result.stdout)
        self.assertTrue(output.is_dir())

    def test_training_without_runtime_stops_before_launch(self):
        result = self.bash(8, RUN_TRAINING="1", PROJECT_DIR=str(self.root),
                           YAML_CONFIG="", OUTPUT_DIR=str(self.root / "new"))
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("command not found", result.stderr)

    def test_train_command_uses_explicit_python_and_scoped_state(self):
        fake = self.root / "python"
        fake.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$CALLS"\n')
        fake.chmod(0o700)
        calls = self.root / "calls"
        result = self.bash(8, RUN_TRAINING="1", PROJECT_DIR=str(self.root),
                           YAML_CONFIG="runtime.yaml", OUTPUT_DIR=str(self.root / "new"),
                           PYTHON=str(fake), RUN_UUID="a"*32, SYNC_STATE_PATH="run/state.json",
                           MLFLOW_EXPERIMENT_NAME="experiment", CALLS=str(calls))
        self.assertEqual(result.returncode, 0, result.stderr)
        args = calls.read_text().splitlines()
        self.assertEqual(args[0], "scripts/train_with_checkpoint_sync.py")
        self.assertEqual(args[args.index("--state-path") + 1], "run/state.json")

    def test_no_shared_benchmark_upload_or_new_pipeline_claim(self):
        self.assertNotIn("mlflow.start_run", self.sources[10])
        self.assertNotIn("kaggle_test_results", self.sources[10])
        self.assertIn("không dùng checkpoint 700", self.sources[0])
        self.assertIn("wrapper cũ", self.sources[8])
        for index in (4, 9, 11):
            self.assertTrue(self.sources[index].startswith("# %%bash"))

    def test_notebook_source_is_shorter_and_setup_readable(self):
        active = sum(len(s.splitlines()) for s in self.sources)
        self.assertLess(active, 280)
        self.assertLess(len(self.sources[2].splitlines()), 22)

    def test_summary_cli_is_opt_in_and_does_not_include_arbitrary_secrets(self):
        base = self.root / "base.yaml"
        base.write_text("output_dir: saves\nfp16: true\nbf16: false\nsecret_key: sentinel-value\n")
        command = [sys.executable, "-B", str(ROOT / "scripts/runtime_yaml.py"),
                   "--project-dir", str(self.root), "--base-yaml", str(base),
                   "--runtime-yaml", "runtime.yaml", "--summary"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout)["summary"]
        self.assertTrue(summary["fp16"])
        self.assertNotIn("secret_key", summary)
        self.assertNotIn("sentinel-value", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()

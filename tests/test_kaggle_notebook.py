"""Contracts for the simplified notebook. Never execute setup/install/train."""
import ast
import contextlib
import gzip
import hashlib
import io
import json
import os
from functools import partial
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import yaml
from scripts import prepare_k2_data as k2
from scripts.prepare_baseline_data import prepare_from_rows
from scripts.prepare_k2_data import prepare_k2_data
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
from test_prepare_baseline_data import rows
from test_prepare_k2_data import onpolicy_row, write_package

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "uni-mumer-kaggle-dagshub-banmoi.ipynb"


class KaggleNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.sources = ["".join(c["source"]) if isinstance(c["source"], list)
                       else c["source"] for c in cls.notebook["cells"]]
        cls.k1_source = cls.sources.pop(7)
        cls.resume_source = cls.sources.pop(6)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def namespace(self, variant="onpolicy"):
        with mock.patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()):
            namespace = {}
            source = self.sources[1].replace('K2_VARIANT = "onpolicy"', f'K2_VARIANT = "{variant}"')
            exec(compile(source, "cell-1", "exec"), namespace)
            exec(compile(self.resume_source, "cell-resume", "exec"), namespace)
        return namespace

    def runtime_overrides(self):
        settings = next(node.value for node in ast.parse(self.sources[6]).body
                        if isinstance(node, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "YAML_OVERRIDES" for t in node.targets))
        values = {key.value: ast.literal_eval(value) for key, value in zip(settings.keys, settings.values)
                  if key.value not in ("output_dir", "model_revision")}
        values["model_revision"] = self.namespace()["MODEL_REVISION"]
        return values

    def runtime_namespace(self, variant="onpolicy"):
        project = self.root / "project"
        (project / "scripts").mkdir(parents=True, exist_ok=True)
        (project / "train").mkdir(exist_ok=True)
        for name in ("scripts/runtime_yaml.py", "scripts/prepare_baseline_data.py",
                     "scripts/dagshub_checkpoint.py",
                     "scripts/checkpoint_contract.py", "train/Uni-MuMER-train.yaml", "train/dataset_info.json"):
            shutil.copyfile(ROOT / name, project / name)
        namespace = self.namespace(variant)
        namespace.update(PROJECT_DIR=str(project), PYTHON=sys.executable, RUN_TRAINING=True,
                         RUNTIME_YAML_CONFIG=str(self.root / "runtime.yaml"), RUN_DIR=str(self.root / "run"),
                         OUTPUT_DIR=str(self.root / "output"), K1_DATASET_DIR=str(self.root / "K1_datascale_sft"),
                         DATASET_DIR=str(self.root / namespace["RUN_VARIANT"]))
        registry = json.loads((project / "train/dataset_info.json").read_text())
        settings = {node.targets[0].id: ast.literal_eval(node.value) for node in ast.parse(self.k1_source).body
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in ("K1_DATASETS", "K1_PER_SOURCE_CAPS")}
        names = settings["K1_DATASETS"].split(", ")
        source_rows = {name: rows() for name in names if name != "parquet_mathwriting_train"}
        revisions = {"phxember/Uni-MuMER-Data": namespace["DATASET_REVISION"]}
        base = prepare_from_rows(self.root / "base", source_rows, {name: registry[name] for name in source_rows},
                                 max_samples=8000, revisions=revisions)
        namespace.update(BASE_SPLIT_MANIFEST=base["manifest_path"], BASE_SPLIT_SHA256=base["manifest_sha256"])
        source_rows["parquet_hme100k_train"] = rows(25)
        source_rows["parquet_mathwriting_train"] = rows(22, offset=100)
        prepared = Path(namespace["K1_DATASET_DIR"])
        result = prepare_from_rows(
            prepared, source_rows, {name: registry[name] for name in names}, revisions=revisions,
            max_samples=8000, max_samples_by_source=settings["K1_PER_SOURCE_CAPS"],
            base_manifest=base["manifest_path"], base_manifest_sha256=base["manifest_sha256"],
        )
        namespace["K1_SPLIT_SHA256"] = result["manifest_sha256"]
        records = json.loads(Path(result["manifest_path"]).read_text())["records"]
        images = [r for r in records if r["source"] == k2.RECOGNITION and r["split"] == "train"]
        self.edl_counts = {"error_find": 3, "error_fix": 2}
        self.package = project / "train/k2_data/onpolicy_b4_500.json.gz"
        self.package_sha = write_package(self.package, {
            task: [onpolicy_row(row, task) for row in images[:count]]
            for task, count in self.edl_counts.items()
        })
        self.adapter_downloads = []
        return namespace

    def prepare_fixture_k2(self, *args, **kwargs):
        return prepare_k2_data(*args, **kwargs, replay_size=18, edl_counts=self.edl_counts)

    def run_runtime(self, namespace, runtime_source=None):
        run = subprocess.run

        def run_fixture(command, **kwargs):
            if command[1:3] == ["-m", "scripts.prepare_k2_data"]:
                argv = command[3:].copy()
                argv[argv.index("--onpolicy-package") + 1] = str(self.package)
                prepare = partial(self.prepare_fixture_k2, expected_k1_sha256=namespace["K1_SPLIT_SHA256"])
                try:
                    with mock.patch.object(k2, "prepare_k2_data", side_effect=prepare), \
                            contextlib.redirect_stdout(io.StringIO()) as stdout:
                        k2.main(argv)
                    return subprocess.CompletedProcess(command, 0, stdout.getvalue(), "")
                except (ValueError, RuntimeError) as error:
                    return subprocess.CompletedProcess(command, 1, "", str(error))
            if command[1] == "scripts/dagshub_checkpoint.py":
                self.adapter_downloads.append(command)
                state = Path(command[command.index("--state-path") + 1])
                state.parent.mkdir(parents=True, exist_ok=True)
                state.write_text(json.dumps({"adapter": str(self.root / "adapter")}))
                return subprocess.CompletedProcess(command, 0)
            return run(command, **kwargs)

        with mock.patch.dict(os.environ, MLFLOW_TAGS="{}"), contextlib.redirect_stdout(io.StringIO()), \
                mock.patch("subprocess.run", side_effect=run_fixture):
            exec(compile(self.k1_source, "cell-k1", "exec"), namespace)
            exec(compile(runtime_source or self.sources[6], "cell-k2", "exec"), namespace)
            return dict(os.environ)

    def set_resume_data_sha(self, namespace):
        result = self.prepare_fixture_k2(
            namespace["K1_DATASET_DIR"], self.package, namespace["DATASET_DIR"], namespace["K2_VARIANT"],
            expected_k1_sha256=namespace["K1_SPLIT_SHA256"], expected_onpolicy_sha256=self.package_sha,
        )
        namespace["RESUME_DATA_SHA256"] = result["manifest_sha256"]

    def bash(self, index, **environment):
        env = dict(os.environ, RUN_TRAINING="0", PATH="/nonexistent")
        env.update(environment)
        return subprocess.run(["/bin/bash", "-c", self.sources[index].split("\n", 1)[1]],
                              env=env, text=True, capture_output=True, timeout=5)

    def test_notebook_structure_and_no_historical_outputs(self):
        self.assertEqual(len(self.notebook["cells"]), 14)
        self.assertEqual(self.notebook["nbformat"], 4)
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_python_bash_and_embedded_python_parse(self):
        for index, source in enumerate(self.sources[1:] + [self.resume_source, self.k1_source], 1):
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
        for node in ast.walk(ast.parse(self.resume_source)):
            if isinstance(node, ast.Import):
                self.assertTrue(all(a.name in allowed for a in node.names))
            elif isinstance(node, ast.ImportFrom):
                self.assertIn(node.module, allowed)

    def test_single_training_switch_defaults_off(self):
        namespace = self.namespace()
        self.assertIs(namespace["RUN_TRAINING"], False)
        for obsolete in ("ALLOW_SETUP", "ALLOW_TRAINING", "ALLOW_REMOTE_WRITES",
                         "PIPELINE_REVIEW_COMPLETE", "SOURCE_COMMIT",
                         "MINICONDA_INSTALLER_SHA256"):
            self.assertNotIn(obsolete, namespace)

    def test_k2_clears_previous_stop_step_at_default(self):
        with mock.patch.dict(os.environ, STOP_AT_GLOBAL_STEP="250"), contextlib.redirect_stdout(io.StringIO()):
            exec(self.sources[1], {})
            self.assertEqual(os.environ["STOP_AT_GLOBAL_STEP"], "")

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
        self.assertIn("--branch dev-caitien", self.sources[3])
        self.assertIn("branch --show-current", self.sources[3])

    def test_pinned_split_mismatch_blocks_runtime_before_train(self):
        namespace = self.runtime_namespace()
        namespace["BASE_SPLIT_SHA256"] = "0" * 64
        with self.assertRaises(RuntimeError):
            self.run_runtime(namespace)
        self.assertFalse(Path(namespace["RUNTIME_YAML_CONFIG"]).exists())

    def test_cached_dataset_revision_mismatch_is_rejected(self):
        namespace = self.runtime_namespace()
        namespace["DATASET_REVISION"] = "0" * 40
        with self.assertRaises(RuntimeError):
            self.run_runtime(namespace)
        self.assertFalse(Path(namespace["RUNTIME_YAML_CONFIG"]).exists())

    def test_pinned_k1_manifest_mismatch_blocks_k2(self):
        namespace = self.runtime_namespace()
        namespace["K1_SPLIT_SHA256"] = "0" * 64
        with self.assertRaises(RuntimeError):
            self.run_runtime(namespace)
        self.assertFalse(Path(namespace["DATASET_DIR"]).exists())
        self.assertEqual(self.adapter_downloads, [])

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
        self.assertEqual(config["eval_dataset"], "k2_val")
        self.assertEqual(config["dataset"], "k2_train")
        self.assertIsNone(config["resume_from_checkpoint"])
        self.assertEqual(config["adapter_name_or_path"], str(self.root / "adapter"))
        self.assertEqual(config["max_steps"], -1)  # Trainer compares this integer with zero.
        self.assertFalse(config["create_new_adapter"])
        self.assertTrue(config["load_best_model_at_end"])
        self.assertEqual(config["eval_steps"], 69)
        self.assertEqual(config["save_steps"], 69)
        self.assertEqual(config["per_device_train_batch_size"], 4)
        self.assertEqual(config["gradient_accumulation_steps"], 32)
        self.assertEqual(config["per_device_train_batch_size"] * config["gradient_accumulation_steps"] * 2, 256)
        self.assertEqual(config["lora_rank"], 64)
        self.assertEqual(config["lora_alpha"], 16)
        self.assertEqual(config["lora_dropout"], .05)
        self.assertEqual(config["quantization_bit"], 4)
        self.assertEqual(config["quantization_type"], "nf4")
        self.assertEqual(config["image_max_pixels"], 262144)
        self.assertEqual(config["cutoff_len"], 2048)
        self.assertFalse(config["overwrite_output_dir"])
        self.assertFalse(config["use_hard_example_loss"])
        self.assertEqual(config["hard_example_loss_alpha"], 0.5)
        self.assertEqual(config["hard_example_loss_max_weight"], 1.5)
        self.assertEqual(config["model_revision"], "66285546d2b821cf421d4f5eb2576359d3770cd3")
        self.assertNotIn("max_samples_by_source", config)
        tags = json.loads(env["MLFLOW_TAGS"])
        self.assertEqual(tags["variant"], "K2_onpolicy_sft")
        self.assertEqual(tags["init_adapter_run_id"], namespace["INIT_ADAPTER_RUN_ID"])
        self.assertEqual(tags["init_adapter_sha256"], namespace["INIT_ADAPTER_SHA256"])
        self.assertEqual(tags["split_sha256"], namespace["K1_SPLIT_SHA256"])
        self.assertEqual(tags["k2_data_sha256"], hashlib.sha256(Path(env["SPLIT_MANIFEST"]).read_bytes()).hexdigest())
        self.assertEqual(tags["initial_global_step"], "0")
        [command] = self.adapter_downloads
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[command.index("--artifact-path") + 1], "training")
        self.assertEqual(command[command.index("--adapter-sha256") + 1], tags["init_adapter_sha256"])
        self.assertEqual(env["OUTPUT_DIR"], namespace["OUTPUT_DIR"])

    def test_control_uses_same_training_config_and_selects_control_manifest(self):
        onpolicy = self.runtime_namespace()
        first = self.run_runtime(onpolicy)
        onpolicy_config = yaml.safe_load(Path(first["YAML_CONFIG"]).read_text())
        control = self.runtime_namespace("control")
        second = self.run_runtime(control)
        control_config = yaml.safe_load(Path(second["YAML_CONFIG"]).read_text())
        self.assertNotEqual(onpolicy_config.pop("dataset_dir"), control_config.pop("dataset_dir"))
        self.assertEqual(onpolicy_config, control_config)
        first_manifest = json.loads(Path(first["SPLIT_MANIFEST"]).read_text())
        second_manifest = json.loads(Path(second["SPLIT_MANIFEST"]).read_text())
        self.assertEqual(first_manifest["replay_row_ids"], second_manifest["replay_row_ids"])
        self.assertEqual(first_manifest["files"]["val.json"], second_manifest["files"]["val.json"])
        self.assertEqual(second_manifest["variant"], "K2_control_sft")
        self.assertEqual(json.loads(second["MLFLOW_TAGS"])["variant"], "K2_control_sft")

    def test_run_training_false_builds_k2_without_downloading_adapter_or_publishing_yaml(self):
        namespace = self.runtime_namespace()
        namespace["RUN_TRAINING"] = False
        env = self.run_runtime(namespace)
        self.assertTrue((Path(namespace["DATASET_DIR"]) / "k2_manifest.json").is_file())
        self.assertEqual(env["YAML_CONFIG"], "")
        self.assertEqual(self.adapter_downloads, [])

    def test_current_notebook_configuration(self):
        namespace = self.namespace()
        self.assertFalse(namespace["USE_DAGSHUB_RESUME"])
        self.assertEqual(namespace["RESUME_RUN_ID"], "")
        self.assertEqual(namespace["RESUME_ARTIFACT_PATH"], "")
        self.assertEqual(namespace["RESUME_DATA_SHA256"], "")
        self.assertEqual(namespace["RESUME_TOTAL_EPOCHS"], 1)
        self.assertIn('"training/artifact_manifest.json"', self.resume_source)

        overrides = self.runtime_overrides()
        sources = [
            "parquet_crohme_train", "parquet_crohme_train_can", "parquet_crohme_train_tree",
            "parquet_crohme_train_error_find", "parquet_crohme_train_error_fix",
            "parquet_hme100k_train", "parquet_mathwriting_train",
        ]
        self.assertIn(f'K1_DATASETS = "{", ".join(sources)}"', self.k1_source)
        self.assertIn('"--max-samples", "8000"', self.k1_source)
        self.assertIn('{"parquet_hme100k_train": 16000, "parquet_mathwriting_train": 16000}', self.k1_source)
        self.assertEqual(overrides["num_train_epochs"], 1)
        self.assertEqual(overrides["per_device_train_batch_size"], 4)
        self.assertEqual(overrides["gradient_accumulation_steps"], 32)
        self.assertEqual(overrides["per_device_train_batch_size"] * overrides["gradient_accumulation_steps"] * 2, 256)
        self.assertFalse(overrides["use_hard_example_loss"])
        self.assertEqual(overrides["hard_example_loss_alpha"], 0.5)
        self.assertEqual(overrides["hard_example_loss_max_weight"], 1.5)
        self.assertTrue(overrides["load_best_model_at_end"])
        self.assertEqual(overrides["metric_for_best_model"], "eval_loss")
        self.assertFalse(overrides["greater_is_better"])
        self.assertEqual(overrides["save_total_limit"], 2)

    def test_packaged_base_manifest_matches_registry_and_pinned_revision(self):
        namespace = self.namespace()
        original = gzip.decompress((ROOT / namespace["BASE_SPLIT_MANIFEST"]).read_bytes())
        self.assertEqual(hashlib.sha256(original).hexdigest(),
                         "a3ffe3480ceaafcaebf96f31102e9f97dadd9effc56ca235dc573cb0ff00a11a")
        self.assertEqual(hashlib.sha256(original).hexdigest(), namespace["BASE_SPLIT_SHA256"])
        base = json.loads(original)
        self.assertEqual(len(base["records"]), 48000)
        self.assertEqual(base["source_revisions"], {"phxember/Uni-MuMER-Data": namespace["DATASET_REVISION"]})
        registry = json.loads((ROOT / "train/dataset_info.json").read_text())
        self.assertEqual(base["settings"]["source_entries"],
                         {name: registry[name] for name in base["settings"]["datasets"]})
        mathwriting = registry["parquet_mathwriting_train"]
        self.assertEqual(mathwriting["subset"], "mathwriting_train")
        self.assertEqual(mathwriting["columns"], {"messages": "conversations", "images": "image"})

    def test_resume_cell_when_run_training_false_stops_safely_without_network(self):
        source = self.resume_source.replace("USE_DAGSHUB_RESUME = False", "USE_DAGSHUB_RESUME = True")
        with mock.patch("subprocess.run") as run:
            with mock.patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()):
                namespace = {}
                exec(compile(self.sources[1], "cell-1", "exec"), namespace)
                self.assertFalse(namespace["RUN_TRAINING"])
                with self.assertRaisesRegex(RuntimeError, "Resume cần RUN_TRAINING=True để đọc DagsHub Secrets."):
                    exec(compile(source, "cell-resume", "exec"), namespace)
            run.assert_not_called()

    def test_resume_cell_defaults_to_fresh_without_network(self):
        with mock.patch("subprocess.run") as run:
            namespace = self.namespace()
        run.assert_not_called()
        self.assertFalse(namespace["USE_DAGSHUB_RESUME"])
        self.assertIsNone(namespace["RESUME_CHECKPOINT"])

    def test_resume_cell_uses_own_python_cache_and_two_gpu_fp16_states(self):
        namespace = self.namespace()
        namespace.update(RUN_TRAINING=True, RUN_DIR=str(self.root / "run"),
                         PROJECT_DIR=str(self.root), PYTHON="/own-env/bin/python")
        state = self.root / "run/resume_checkpoint.json"
        state.parent.mkdir()
        source = self.resume_source.replace("USE_DAGSHUB_RESUME = False", "USE_DAGSHUB_RESUME = True")
        for name, value in (("RESUME_RUN_ID", "a" * 32), ("RESUME_ARTIFACT_PATH", "checkpoints/checkpoint-250"),
                            ("RESUME_DATA_SHA256", "b" * 64)):
            source = source.replace(f'{name} = ""', f'{name} = "{value}"')

        def download(command, **kwargs):
            self.assertEqual(command[0], "/own-env/bin/python")
            self.assertEqual(command[command.index("--world-size") + 1], "2")
            self.assertIn("--fp16", command)
            self.assertIn("resume-source", command[command.index("--output-dir") + 1])
            self.assertIn("--manifest-artifact", command)
            self.assertEqual(command[command.index("--manifest-artifact") + 1], "training/artifact_manifest.json")
            tags = json.loads(command[command.index("--expected-run-tags") + 1])
            self.assertEqual(tags["variant"], "K2_onpolicy_sft")
            self.assertEqual(tags["k2_data_sha256"], "b" * 64)
            self.assertEqual(tags["init_adapter_run_id"], namespace["INIT_ADAPTER_RUN_ID"])
            self.assertEqual(tags["init_adapter_sha256"], namespace["INIT_ADAPTER_SHA256"])
            state.write_text(json.dumps({"checkpoint": "/cache/checkpoint-250",
                                         "prepared_output": "/cache/prepared-output"}))
        with mock.patch("subprocess.run", side_effect=download), mock.patch.dict(os.environ), \
                contextlib.redirect_stdout(io.StringIO()):
            exec(source, namespace)
        self.assertEqual(namespace["RESUME_CHECKPOINT"], "/cache/checkpoint-250")

    def test_full_resume_runtime_preserves_checkpoint_and_target_epoch(self):
        namespace = self.runtime_namespace()
        checkpoint = self.root / "cache/checkpoint-250"
        checkpoint.mkdir(parents=True)
        for name in ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt",
                     "scheduler.pt", "scaler.pt", "rng_state_0.pth", "rng_state_1.pth"):
            (checkpoint / name).write_bytes(b"fixture only; never unpickle")
        (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 250, "epoch": .984}))
        namespace.update(USE_DAGSHUB_RESUME=True, RESUME_CHECKPOINT=str(checkpoint),
                         RESUME_RUN_ID="a" * 32, RESUME_ARTIFACT_PATH="checkpoints/checkpoint-250")
        self.set_resume_data_sha(namespace)
        env = self.run_runtime(namespace)
        config = yaml.safe_load(Path(env["YAML_CONFIG"]).read_text())
        self.assertEqual(config["resume_from_checkpoint"], str(checkpoint))
        self.assertEqual(config["adapter_name_or_path"], str(checkpoint))
        self.assertFalse(config["create_new_adapter"])
        self.assertEqual(config["num_train_epochs"], 1)
        self.assertEqual(config["max_steps"], -1)
        self.assertEqual(config["lr_scheduler_type"], "cosine")
        self.assertEqual(config["warmup_ratio"], .1)
        self.assertFalse(config["use_hard_example_loss"])
        self.assertFalse(Path(namespace["OUTPUT_DIR"]).exists())
        tags = json.loads(env["MLFLOW_TAGS"])
        self.assertEqual(tags["parent_run_id"], namespace["RESUME_RUN_ID"])
        self.assertEqual(tags["initial_global_step"], "250")
        self.assertEqual(tags["initial_epoch"], "0.984")
        self.assertEqual(tags["variant"], "K2_onpolicy_sft")
        self.assertEqual(self.adapter_downloads, [])
        namespace["RESUME_TOTAL_EPOCHS"] = 0
        with self.assertRaises(RuntimeError):
            self.run_runtime(namespace)
        namespace.update(RESUME_TOTAL_EPOCHS=1, RESUME_DATA_SHA256="0" * 64)
        with self.assertRaises(RuntimeError):
            self.run_runtime(namespace)
        self.assertEqual(yaml.safe_load(Path(env["YAML_CONFIG"]).read_text()), config)

    def test_enabled_resume_without_download_does_not_fall_back_to_fresh(self):
        namespace = self.runtime_namespace()
        namespace.update(USE_DAGSHUB_RESUME=True, RESUME_CHECKPOINT=None)
        with self.assertRaisesRegex(RuntimeError, "Resume"):
            self.run_runtime(namespace)
        self.assertFalse(Path(namespace["RUNTIME_YAML_CONFIG"]).exists())

    def test_resume_epoch_four_to_six_preserves_full_state_paths_and_training_settings(self):
        run_id = "51c046b18b5140a080c387416d4828e8"
        artifact_path = "training/checkpoint-676"
        total_epochs = 6
        namespace = self.runtime_namespace()
        namespace.update(RESUME_RUN_ID=run_id, RESUME_ARTIFACT_PATH=artifact_path, RESUME_TOTAL_EPOCHS=total_epochs)
        self.assertEqual(namespace["RESUME_RUN_ID"], run_id)
        self.assertEqual(namespace["RESUME_ARTIFACT_PATH"], artifact_path)
        self.assertEqual(namespace["RESUME_TOTAL_EPOCHS"], total_epochs)
        checkpoint = self.root / "cache/checkpoint-676"
        checkpoint.mkdir(parents=True)
        for name in ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt",
                     "scheduler.pt", "scaler.pt", "rng_state_0.pth", "rng_state_1.pth"):
            (checkpoint / name).write_bytes(b"fixture only; never unpickle")
        (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 676, "epoch": 4.0}))
        namespace.update(USE_DAGSHUB_RESUME=True, RESUME_CHECKPOINT=str(checkpoint))
        self.set_resume_data_sha(namespace)
        env = self.run_runtime(namespace)
        config = yaml.safe_load(Path(env["YAML_CONFIG"]).read_text())
        self.assertEqual(config["resume_from_checkpoint"], str(checkpoint))
        self.assertEqual(config["adapter_name_or_path"], str(checkpoint))
        self.assertFalse(config["create_new_adapter"])
        self.assertEqual(config["num_train_epochs"], 6)
        self.assertEqual(config["max_steps"], -1)
        self.assertEqual(config["lr_scheduler_type"], "cosine")
        self.assertEqual(config["save_steps"], 69)
        self.assertEqual(config["eval_steps"], 69)
        self.assertEqual(config["save_total_limit"], 2)
        self.assertTrue(config["load_best_model_at_end"])
        tags = json.loads(env["MLFLOW_TAGS"])
        self.assertEqual(tags["initial_global_step"], "676")
        self.assertEqual(tags["initial_epoch"], "4.0")

    def test_training_uses_prepared_resume_not_the_old_persistent_best_cache(self):
        self.assertIn('--resume-prepared-output "$RESUME_PREPARED_OUTPUT"', self.sources[8])
        self.assertNotIn("--resume-best-checkpoint", self.sources[8])

    def test_fresh_mode_ignores_stale_resume_variable(self):
        namespace = self.runtime_namespace()
        namespace["RESUME_CHECKPOINT"] = "/stale/checkpoint-338"
        env = self.run_runtime(namespace)
        config = yaml.safe_load(Path(env["YAML_CONFIG"]).read_text())
        self.assertIsNone(config["resume_from_checkpoint"])
        self.assertFalse(config["create_new_adapter"])

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
                exec(self.k1_source, namespace)
            self.assertEqual(os.environ["YAML_CONFIG"], "")

    def test_editing_overrides_before_training_can_regenerate_runtime(self):
        namespace = self.runtime_namespace()
        self.run_runtime(namespace)
        runtime = Path(namespace["RUNTIME_YAML_CONFIG"])
        source = self.sources[6].replace('"learning_rate": 1.0e-4', '"learning_rate": 2.0e-4')
        self.run_runtime(namespace, runtime_source=source)
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
                           MLFLOW_EXPERIMENT_NAME="experiment", EXPERIMENT_VARIANT="K1_datascale_sft",
                           CALLS=str(calls))
        self.assertEqual(result.returncode, 0, result.stderr)
        args = calls.read_text().splitlines()
        self.assertEqual(args[0], "scripts/train_with_checkpoint_sync.py")
        self.assertEqual(args[args.index("--run-name") + 1], "K1_datascale_sft-" + "a" * 32)
        self.assertEqual(args[args.index("--state-path") + 1], "run/state.json")

    def test_no_shared_benchmark_upload_or_new_pipeline_claim(self):
        self.assertNotIn("mlflow.start_run", self.sources[10])
        self.assertNotIn("kaggle_test_results", self.sources[10])
        self.assertIn("full resume", self.sources[0])
        self.assertIn("wrapper cũ", self.sources[8])
        for index in (4, 9, 11):
            self.assertTrue(self.sources[index].startswith("# %%bash"))

    def test_upload_propagates_timeout_exit_status(self):
        timeout = self.root / "timeout"
        timeout.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\nexit 124\n')
        timeout.chmod(0o700)
        result = self.bash(10, RUN_TRAINING="1", PROJECT_DIR=str(self.root), PATH=str(self.root),
                           YAML_CONFIG="runtime.yaml", OUTPUT_DIR="output", PYTHON="/own-env/bin/python",
                           RUN_UUID="a" * 32, MLFLOW_EXPERIMENT_NAME="test", NOTEBOOK_PATH="source.ipynb")
        self.assertEqual(result.returncode, 124, result.stderr)
        self.assertEqual(result.stdout.splitlines()[:5], ["--kill-after=30s", "30m", "/own-env/bin/python",
                                                         "scripts/dagshub_logger.py", "upload"])

    def test_notebook_source_is_shorter_and_setup_readable(self):
        active = sum(len(s.splitlines()) for s in self.sources + [self.resume_source, self.k1_source])
        self.assertLess(active, 380)
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

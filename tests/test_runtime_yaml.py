import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import yaml

from scripts.runtime_yaml import prepare_runtime_yaml, update_mlflow_tags


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/runtime_yaml.py"


class RuntimeYamlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "base.yaml"
        self.runtime = self.root / "run/runtime.yaml"
        self.config = {
            "output_dir": "saves/new-run", "dataset": "train", "max_samples": 16000,
            "bf16": True, "fp16": False, "resume_from_checkpoint": "old/checkpoint-700",
        }
        self.base.write_text(yaml.safe_dump(self.config), encoding="utf-8")

    def resolve(self, **kwargs):
        options = dict(project_dir=str(self.root), base_yaml_config="base.yaml",
                       runtime_yaml_config="run/runtime.yaml", verbose=False)
        options.update(kwargs)
        return prepare_runtime_yaml(**options)

    def cli(self, *args):
        env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
                   CUDA_VISIBLE_DEVICES="")
        return subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "--project-dir", str(self.root),
             "--base-yaml", "base.yaml", "--runtime-yaml", "run/runtime.yaml", *args],
            cwd=self.root, env=env, capture_output=True, text=True, timeout=10,
        )

    def test_override_preserves_base_and_return_contract(self):
        original = self.base.read_bytes()
        path, output, config = self.resolve(overrides={"max_samples": 8000})
        self.assertEqual(path, "run/runtime.yaml")
        self.assertEqual(output, "saves/new-run")
        self.assertEqual(config["max_samples"], 8000)
        self.assertEqual(yaml.safe_load(self.runtime.read_text()), config)
        self.assertEqual(self.base.read_bytes(), original)

    def test_none_keeps_old_resume(self):
        _, _, config = self.resolve(overrides={"resume_from_checkpoint": None})
        self.assertEqual(config["resume_from_checkpoint"], "old/checkpoint-700")

    def test_explicit_clear_sets_null(self):
        _, _, config = self.resolve(clear_keys=["resume_from_checkpoint"])
        self.assertIsNone(config["resume_from_checkpoint"])
        self.assertIsNone(yaml.safe_load(self.runtime.read_text())["resume_from_checkpoint"])

    def test_clear_can_accompany_none_override(self):
        _, _, config = self.resolve(overrides={"resume_from_checkpoint": None},
                                    clear_keys=["resume_from_checkpoint"])
        self.assertIsNone(config["resume_from_checkpoint"])

    def test_none_clear_keys_leaves_config_unchanged(self):
        _, _, config = self.resolve(clear_keys=None)
        self.assertEqual(config, self.config)

    def test_clear_supported_extra_key_absent_from_base(self):
        _, _, config = self.resolve(clear_keys=["eval_dataset"])
        self.assertIn("eval_dataset", config)
        self.assertIsNone(config["eval_dataset"])

    def test_clear_and_non_null_override_conflict(self):
        with self.assertRaises(ValueError):
            self.resolve(overrides={"resume_from_checkpoint": "new"},
                         clear_keys=["resume_from_checkpoint"])
        self.assertFalse(self.runtime.exists())

    def test_unknown_key_including_none_is_rejected(self):
        for value in (1, None):
            with self.subTest(value=value), self.assertRaises(KeyError):
                self.resolve(overrides={"max_step_typo": value})
        self.assertFalse(self.runtime.parent.exists())

    def test_unknown_clear_key_is_rejected(self):
        with self.assertRaises(KeyError):
            self.resolve(clear_keys=["resuem_from_checkpoint"])

    def test_supported_new_keys(self):
        extra = {"max_steps": 10, "seed": 42, "data_seed": 42, "eval_dataset": "validation"}
        _, _, config = self.resolve(overrides=extra)
        for key, value in extra.items():
            self.assertEqual(config[key], value)

    def test_legacy_non_strict_api_is_explicit(self):
        _, _, config = self.resolve(strict_keys=False, overrides={"custom_key": 1})
        self.assertEqual(config["custom_key"], 1)

    def test_non_mapping_overrides_rejected(self):
        with self.assertRaises(TypeError):
            self.resolve(overrides=[1])

    def test_non_string_override_key_rejected(self):
        with self.assertRaises(TypeError):
            self.resolve(overrides={1: "bad"})

    def test_string_clear_keys_rejected(self):
        with self.assertRaises(TypeError):
            self.resolve(clear_keys="resume_from_checkpoint")

    def test_disabled_override_keeps_existing_runtime(self):
        self.runtime.parent.mkdir()
        self.runtime.write_text("historical runtime", encoding="utf-8")
        path, _, config = self.resolve(use_override=False, overrides={"max_samples": 3})
        self.assertEqual(path, "base.yaml")
        self.assertEqual(config, self.config)
        self.assertEqual(self.runtime.read_text(), "historical runtime")

    def test_disabled_override_does_not_create_directory(self):
        self.resolve(use_override=False)
        self.assertFalse(self.runtime.parent.exists())

    def test_disabled_override_cannot_clear(self):
        with self.assertRaises(ValueError):
            self.resolve(use_override=False, clear_keys=["resume_from_checkpoint"])

    def test_t4_rejects_bf16_before_writing(self):
        with self.assertRaisesRegex(ValueError, "T4 profile"):
            self.resolve(profile="t4")
        self.assertFalse(self.runtime.parent.exists())

    def test_t4_accepts_explicit_fp16_without_changing_other_values(self):
        _, _, config = self.resolve(profile="t4", overrides={"bf16": False, "fp16": True})
        expected = dict(self.config, bf16=False, fp16=True)
        self.assertEqual(config, expected)

    def test_t4_rejects_strings_and_integer_booleans(self):
        for bf16, fp16 in [("false", "true"), (0, 1), (False, False), (True, True)]:
            with self.subTest(bf16=bf16, fp16=fp16), self.assertRaises(ValueError):
                self.resolve(profile="t4", overrides={"bf16": bf16, "fp16": fp16})

    def test_t4_validates_base_when_override_disabled(self):
        with self.assertRaises(ValueError):
            self.resolve(profile="t4", use_override=False,
                         overrides={"bf16": False, "fp16": True})

    def test_unknown_profile_rejected(self):
        with self.assertRaises(ValueError):
            self.resolve(profile="typo")

    def test_no_profile_preserves_precision(self):
        _, _, config = self.resolve()
        self.assertIs(config["bf16"], True)

    def test_output_dir_validation_precedes_any_write(self):
        for value in (None, "", "  ", False, 1, ["path"]):
            self.base.write_text(yaml.safe_dump({"output_dir": value}), encoding="utf-8")
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.resolve()
            self.assertFalse(self.runtime.parent.exists())

    def test_missing_output_dir(self):
        self.base.write_text("dataset: train\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.resolve()

    def test_missing_base(self):
        with self.assertRaises(FileNotFoundError):
            self.resolve(base_yaml_config="missing.yaml")

    def test_non_mapping_base(self):
        for source in ("", "- item\n", "scalar\n"):
            self.base.write_text(source, encoding="utf-8")
            with self.subTest(source=source), self.assertRaises(TypeError):
                self.resolve()

    def test_non_string_base_key(self):
        self.base.write_text("1: value\noutput_dir: saves\n", encoding="utf-8")
        with self.assertRaises(TypeError):
            self.resolve()

    def test_absolute_paths(self):
        path, _, _ = self.resolve(base_yaml_config=str(self.base),
                                  runtime_yaml_config=str(self.runtime))
        self.assertEqual(path, str(self.runtime))

    def test_same_base_path_rejected(self):
        original = self.base.read_bytes()
        for path in (str(self.base), "base.yaml", "run/../base.yaml"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.resolve(runtime_yaml_config=path, overwrite=True)
        self.assertEqual(self.base.read_bytes(), original)

    def test_symlink_and_hardlink_to_base_rejected(self):
        for name, make_link in (("symlink.yaml", os.symlink), ("hardlink.yaml", os.link)):
            target = self.root / name
            make_link(self.base, target)
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.resolve(runtime_yaml_config=name, overwrite=True)

    def test_symlink_to_other_runtime_rejected(self):
        target = self.root / "other.yaml"
        target.write_text("keep me", encoding="utf-8")
        link = self.root / "linked.yaml"
        link.symlink_to(target)
        with self.assertRaises(ValueError):
            self.resolve(runtime_yaml_config=str(link), overwrite=True)
        self.assertEqual(target.read_text(), "keep me")

    def test_identical_rerun_leaves_existing_file_untouched(self):
        self.resolve()
        before = self.runtime.stat()
        self.resolve()
        after = self.runtime.stat()
        self.assertEqual((before.st_ino, before.st_mtime_ns), (after.st_ino, after.st_mtime_ns))

    def test_changed_rerun_requires_explicit_overwrite(self):
        self.resolve()
        original = self.runtime.read_bytes()
        with self.assertRaises(FileExistsError):
            self.resolve(overrides={"max_samples": 10})
        self.assertEqual(self.runtime.read_bytes(), original)
        _, _, config = self.resolve(overrides={"max_samples": 10}, overwrite=True)
        self.assertEqual(config["max_samples"], 10)

    def test_failed_validation_preserves_old_runtime_even_with_overwrite(self):
        self.resolve()
        original = self.runtime.read_bytes()
        with self.assertRaises(ValueError):
            self.resolve(profile="t4", overwrite=True)
        self.assertEqual(self.runtime.read_bytes(), original)

    def test_atomic_publication_failure_cleans_only_own_temporary_file(self):
        with mock.patch("scripts.runtime_yaml.os.link", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                self.resolve()
        self.assertFalse(self.runtime.exists())
        self.assertEqual(list(self.runtime.parent.glob(".runtime-*.tmp")), [])
        self.assertTrue(self.base.is_file())

    def test_concurrent_creation_is_not_overwritten(self):
        def competing_writer(source, destination):
            Path(destination).write_text("other run", encoding="utf-8")
            raise FileExistsError("already created")
        with mock.patch("scripts.runtime_yaml.os.link", side_effect=competing_writer):
            with self.assertRaises(FileExistsError):
                self.resolve()
        self.assertEqual(self.runtime.read_text(), "other run")
        self.assertEqual(list(self.runtime.parent.glob(".runtime-*.tmp")), [])

    def test_verbose_output_does_not_dump_config_values(self):
        sentinel = "PRIVATE_TEST_VALUE_NOT_A_REAL_TOKEN"
        self.config["hub_token"] = sentinel
        self.base.write_text(yaml.safe_dump(self.config), encoding="utf-8")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.resolve(verbose=True, overrides={"hub_token": sentinel + "_new"})
        self.assertNotIn(sentinel, output.getvalue())

    def test_cli_success_is_one_json_object_with_absolute_paths(self):
        result = self.cli("--profile", "t4", "--overrides-json",
                          '{"bf16": false, "fp16": true, "max_steps": 10}',
                          "--clear", "resume_from_checkpoint")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["yaml_config"], str(self.runtime))
        self.assertEqual(payload["output_dir"], str(self.root / "saves/new-run"))
        self.assertEqual(payload["python"], sys.executable)
        self.assertEqual(payload["profile"], "t4")
        self.assertNotIn("resume_from_checkpoint", result.stdout)

    def test_cli_accepts_overrides_file(self):
        (self.root / "overrides.json").write_text('{"max_samples": 2}', encoding="utf-8")
        result = self.cli("--overrides-file", "overrides.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(yaml.safe_load(self.runtime.read_text())["max_samples"], 2)

    def test_cli_no_override_rejects_explicit_override_sources(self):
        for args in [("--overrides-json", "{}"),
                     ("--overrides-json", '{"max_samples": 2}'),
                     ("--overrides-file", "missing.json")]:
            with self.subTest(args=args):
                result = self.cli("--no-override", *args)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")
                self.assertFalse(self.runtime.exists())

    def test_cli_overwrite_runtime_is_explicit(self):
        self.resolve()
        original = self.runtime.read_bytes()
        args = ("--overrides-json", '{"max_samples": 2}')
        result = self.cli(*args)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.runtime.read_bytes(), original)
        result = self.cli(*args, "--overwrite-runtime")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(yaml.safe_load(self.runtime.read_text())["max_samples"], 2)

    def test_cli_error_has_no_success_json_or_runtime_file(self):
        for args in [("--profile", "t4"), ("--overrides-json", "[]"),
                     ("--overrides-json", '{"typo": 1}'), ("--overrides-json", "not-json")]:
            with self.subTest(args=args):
                result = self.cli(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertTrue(result.stderr.strip())
                self.assertFalse(self.runtime.exists())

    def test_cli_parse_error_does_not_echo_yaml_contents(self):
        sentinel = "PRIVATE_TEST_VALUE_NOT_A_REAL_TOKEN"
        self.base.write_text("hub_token: [" + sentinel, encoding="utf-8")
        result = self.cli()
        self.assertEqual(result.returncode, 2)
        self.assertNotIn(sentinel, result.stderr + result.stdout)

    def test_cli_disabled_override_preserves_runtime(self):
        self.resolve()
        original = self.runtime.read_bytes()
        result = self.cli("--no-override")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["yaml_config"], str(self.base))
        self.assertEqual(self.runtime.read_bytes(), original)

    def test_actual_project_yaml_profile_without_mutating_original(self):
        actual = ROOT / "train/Uni-MuMER-train.yaml"
        original = actual.read_bytes()
        self.base.write_bytes(original)
        with self.assertRaises(ValueError):
            self.resolve(profile="t4")
        _, _, config = self.resolve(profile="t4", overrides={"bf16": False, "fp16": True})
        for key, value in {"lora_rank": 64, "lora_alpha": 16, "quantization_bit": 4,
                           "quantization_type": "nf4", "image_max_pixels": 262144,
                           "cutoff_len": 2048}.items():
            self.assertEqual(config[key], value)
        self.assertEqual(actual.read_bytes(), original)

    def test_tags_preserve_callers_mapping(self):
        original = {"run_uuid": "test-run"}
        tags = update_mlflow_tags("runtime.yaml", self.config, True, original)
        self.assertEqual(original, {"run_uuid": "test-run"})
        self.assertEqual(tags["yaml_override"], "true")
        self.assertEqual(tags["run_uuid"], "test-run")


if __name__ == "__main__":
    unittest.main()

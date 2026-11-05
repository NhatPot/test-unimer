import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from scripts import checkpoint_contract as contract
from scripts.runtime_yaml import prepare_runtime_yaml


RUN_A = "a" * 32
RUN_B = "b" * 32
ARTIFACT = "checkpoints/checkpoint-700"
TRAINING_HASH = "c" * 64
REVISION = "d" * 40


class CheckpointFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "source/checkpoint-700"
        self.checkpoint.mkdir(parents=True)
        self.adapter_config = {"peft_type": "LORA", "base_model_name_or_path": contract.BASE_MODEL,
                               "r": 64, "lora_alpha": 16, "revision": None}
        self.payloads = {
            "adapter_config.json": json.dumps(self.adapter_config).encode(),
            "adapter_model.safetensors": b"fake weight bytes, NOT a loadable tensor",
            "trainer_state.json": json.dumps({"global_step": 700}).encode(),
            "optimizer.pt": b"fake optimizer", "scheduler.pt": b"fake scheduler",
            "scaler.pt": b"fake scaler", "rng_state_0.pth": b"fake RNG 0",
            "rng_state_1.pth": b"fake RNG 1",
        }
        self.manifest = {
            "schema_version": 1, "run_id": RUN_A, "artifact_path": ARTIFACT,
            "global_step": 700, "base_model": contract.BASE_MODEL, "base_revision": REVISION,
            "world_size": 2, "training_contract_sha256": TRAINING_HASH, "files": {},
        }
        self.manifest_path = self.root / "manifest.json"
        for name, content in self.payloads.items():
            self.set_file(name, content)
        self.save_manifest()

    def set_file(self, name, content):
        self.payloads[name] = content
        (self.checkpoint / name).write_bytes(content)
        self.manifest["files"][name] = {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}

    def save_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.pin = contract.sha256_file(self.manifest_path)

    def load(self, **kwargs):
        options = dict(expected_sha256=self.pin, run_id=RUN_A, artifact_path=ARTIFACT)
        options.update(kwargs)
        return contract.load_manifest(self.manifest_path, **options)

    def validate(self, mode="warm-start", **kwargs):
        options = dict(mode=mode, world_size=2, fp16=True)
        if mode == "full-resume":
            options["training_contract_sha256"] = TRAINING_HASH
        options.update(kwargs)
        return contract.validate_checkpoint(self.checkpoint, self.load(), **options)


class ManifestTests(CheckpointFixture):
    def test_valid_manifest_and_warm_start_integrity_not_tensor_validation(self):
        evidence = self.validate()
        self.assertEqual(evidence["verified_files"], ["adapter_config.json", "adapter_model.safetensors"])
        self.assertFalse(evidence["tensor_load_tested"])

    def test_pin_mismatch(self):
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.load(expected_sha256="0" * 64)

    def test_explicit_source_mismatch(self):
        for change in ({"run_id": RUN_B}, {"artifact_path": "other/checkpoint-700"},
                       {"base_model": "different/model"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load(**change)

    def test_path_traversal_absolute_and_noncanonical_artifacts(self):
        for path in ("../checkpoint-700", "/checkpoint-700", "checkpoints//checkpoint-700",
                     "checkpoints/./checkpoint-700", "checkpoints/checkpoint-701"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.load(artifact_path=path)

    def test_unsafe_file_names(self):
        original = copy.deepcopy(self.manifest)
        for name in ("../evil", "/evil", "dir/evil", "..", ".", "a\\b"):
            self.manifest = copy.deepcopy(original)
            self.manifest["files"][name] = {"size": 1, "sha256": "0" * 64}
            self.save_manifest()
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.load()

    def test_wrong_file_schema(self):
        for entry in ({"size": 0, "sha256": "0" * 64}, {"size": True, "sha256": "0" * 64},
                      {"size": 1, "sha256": "not-a-hash"}, None):
            self.manifest["files"]["adapter_config.json"] = entry
            self.save_manifest()
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                self.load()

    def test_duplicate_json_keys(self):
        self.manifest_path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
        self.pin = contract.sha256_file(self.manifest_path)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.load()

    def test_symlink_manifest_or_ancestor_rejected(self):
        alias = self.root / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            contract.load_manifest(alias / "manifest.json", expected_sha256=self.pin,
                                   run_id=RUN_A, artifact_path=ARTIFACT)

    def test_ambiguous_adapter_weights_rejected(self):
        self.set_file("adapter_model.bin", b"another weight file")
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.load()

    def test_full_model_not_a_lora_adapter(self):
        entry = self.manifest["files"].pop("adapter_model.safetensors")
        self.manifest["files"]["model.safetensors"] = entry
        self.save_manifest()
        with self.assertRaises(ValueError):
            self.load()

    def test_oversized_json(self):
        self.manifest_path.write_bytes(b" " * (1024 * 1024 + 1))
        self.pin = contract.sha256_file(self.manifest_path)
        with self.assertRaisesRegex(ValueError, "oversized"):
            self.load()

    def test_boolean_schema_and_step_are_not_integers(self):
        for key in ("schema_version", "global_step"):
            old = self.manifest[key]
            self.manifest[key] = True
            self.save_manifest()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load()
            self.manifest[key] = old


class ValidationTests(CheckpointFixture):
    def test_warm_start_needs_no_optimizer_or_trainer_state(self):
        for name in self.payloads:
            if not name.startswith("adapter_"):
                (self.checkpoint / name).unlink()
        self.validate()
        with self.assertRaises(ValueError):
            self.validate("full-resume")

    def test_missing_config(self):
        (self.checkpoint / "adapter_config.json").unlink()
        with self.assertRaisesRegex(ValueError, "adapter_config.json"):
            self.validate()

    def test_modified_weight_with_same_size(self):
        target = self.checkpoint / "adapter_model.safetensors"
        target.write_bytes(b"x" * target.stat().st_size)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.validate()

    def test_wrong_adapter_settings(self):
        for change in ({"r": 32}, {"lora_alpha": 64}, {"peft_type": "OTHER"},
                       {"base_model_name_or_path": "other"}, {"revision": "e" * 40}):
            config = dict(self.adapter_config, **change)
            self.set_file("adapter_config.json", json.dumps(config).encode())
            self.save_manifest()
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate()

    def test_corrupt_adapter_json_even_with_matching_hash(self):
        self.set_file("adapter_config.json", b"not JSON")
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "JSON"):
            self.validate()

    def test_full_resume_with_states_passes_only_byte_integrity(self):
        evidence = self.validate("full-resume")
        self.assertIn("optimizer.pt", evidence["verified_files"])
        self.assertFalse(evidence["tensor_load_tested"])

    def test_full_resume_requires_each_state_file(self):
        for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "scaler.pt",
                     "rng_state_0.pth", "rng_state_1.pth"):
            content = (self.checkpoint / name).read_bytes()
            (self.checkpoint / name).unlink()
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                self.validate("full-resume")
            (self.checkpoint / name).write_bytes(content)

    def test_trainer_global_step_must_match(self):
        self.set_file("trainer_state.json", b'{"global_step": 701}')
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "step"):
            self.validate("full-resume")

    def test_topology_and_training_contract_must_match(self):
        for kwargs in ({"world_size": 1}, {"training_contract_sha256": "0" * 64},
                       {"training_contract_sha256": None}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.validate("full-resume", **kwargs)

    def test_single_gpu_rng_and_no_fp16_scaler(self):
        self.manifest["world_size"] = 1
        self.set_file("rng_state.pth", b"fake single GPU RNG")
        (self.checkpoint / "scaler.pt").unlink()
        self.save_manifest()
        self.validate("full-resume", world_size=1, fp16=False)
        with self.assertRaises(ValueError):
            self.validate("full-resume", world_size=1, fp16=True)

    def test_revision_check_and_unrecorded_revision(self):
        self.validate(base_revision=REVISION)
        self.manifest["base_revision"] = None
        self.save_manifest()
        self.assertIsNone(self.validate()["base_revision"])
        with self.assertRaisesRegex(ValueError, "revision"):
            self.validate(base_revision=REVISION)

    def test_extra_file_and_symlink_rejected(self):
        extra = self.checkpoint / "unrecorded.json"
        extra.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unexpected"):
            self.validate()
        extra.unlink()
        extra.symlink_to(self.manifest_path)
        with self.assertRaises(ValueError):
            self.validate()


class ModeTests(CheckpointFixture):
    def test_fresh_clears_old_adapter_and_resume(self):
        plan = contract.build_training_overrides("fresh", max_steps=10)
        self.assertEqual(plan["initial_step"], 0)
        self.assertTrue(plan["overrides"]["create_new_adapter"])
        self.assertEqual(set(plan["clear_keys"]), {"resume_from_checkpoint", "adapter_name_or_path"})

    def test_warm_start_700_begins_at_zero_and_trains_same_adapter(self):
        plan = contract.build_training_overrides("warm-start", max_steps=10, checkpoint=self.validate())
        self.assertEqual(plan["initial_step"], 0)
        self.assertFalse(plan["overrides"]["create_new_adapter"])
        self.assertEqual(plan["clear_keys"], ["resume_from_checkpoint"])
        self.assertEqual(plan["overrides"]["adapter_name_or_path"], str(self.checkpoint))

    def test_full_resume_total_step_guard(self):
        evidence = self.validate("full-resume")
        for step in (10, 700):
            with self.subTest(step=step), self.assertRaisesRegex(ValueError, "max_steps"):
                contract.build_training_overrides("full-resume", max_steps=step,
                                                  checkpoint=evidence, continuing_run_id=RUN_A)
        plan = contract.build_training_overrides("full-resume", max_steps=710,
                                                 checkpoint=evidence, continuing_run_id=RUN_A)
        self.assertEqual(plan["initial_step"], 700)
        self.assertEqual(plan["overrides"]["resume_from_checkpoint"], str(self.checkpoint))
        self.assertEqual(plan["clear_keys"], [])

    def test_mixed_or_missing_modes_rejected(self):
        for mode, kwargs in (
            ("fresh", {"checkpoint": self.validate()}),
            ("warm-start", {}), ("full-resume", {"checkpoint": self.validate()}),
            ("warm-start", {"checkpoint": self.validate(), "continuing_run_id": RUN_A}),
            ("full-resume", {"checkpoint": self.validate("full-resume"), "continuing_run_id": RUN_B}),
            ("typo", {}),
        ):
            with self.subTest(mode=mode, kwargs=kwargs), self.assertRaises(ValueError):
                contract.build_training_overrides(mode, max_steps=710, **kwargs)

    def test_mode_plan_resolves_through_real_yaml_helper(self):
        base = self.root / "base.yaml"
        base.write_text("output_dir: output\nresume_from_checkpoint: OLD\nadapter_name_or_path: OLD\n",
                        encoding="utf-8")
        for mode in contract.MODES:
            kwargs = {} if mode == "fresh" else {"checkpoint": self.validate(mode)}
            if mode == "full-resume":
                kwargs["continuing_run_id"] = RUN_A
            plan = contract.build_training_overrides(mode, max_steps=710, **kwargs)
            _, _, config = prepare_runtime_yaml(
                str(self.root), "base.yaml", f"{mode}.yaml", overrides=plan["overrides"],
                clear_keys=plan["clear_keys"], verbose=False,
            )
            self.assertEqual(config["create_new_adapter"], mode == "fresh")
            self.assertEqual(config["resume_from_checkpoint"],
                             str(self.checkpoint) if mode == "full-resume" else None)
            self.assertEqual(config["adapter_name_or_path"],
                             None if mode == "fresh" else str(self.checkpoint))


if __name__ == "__main__":
    unittest.main()

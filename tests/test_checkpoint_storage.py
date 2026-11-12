"""Disk lifecycle tests with tiny fake state files, never Qwen or a remote service."""
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import yaml

from scripts import checkpoint_storage as storage


class CheckpointStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "run/resume-source/parent"
        self.output = self.root / "run/checkpoints"
        self.source = self.checkpoint(self.cache / "checkpoint-338", step=338)
        self.best = self.checkpoint(self.cache / "checkpoint-300", step=300)
        self.config = self.output.parent / "runtime.yaml"

    def checkpoint(self, path, *, step):
        path.mkdir(parents=True)
        for name in ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "scaler.pt",
                     "rng_state_0.pth", "rng_state_1.pth"):
            (path / name).write_bytes((name + str(step)).encode())
        (path / "adapter_config.json").write_text('{"r":64}')
        (path / "trainer_state.json").write_text(json.dumps({
            "global_step": step, "epoch": 2.0, "best_metric": .09,
            "best_model_checkpoint": "/old/checkpoint-300",
        }))
        return path

    def stage(self, *, same_best=False):
        best = self.source if same_best else self.best
        if same_best:
            shutil.rmtree(self.best)  # Fixture only; no remote or user data.
        self.plan = storage.stage_resume(self.source, best, self.cache, self.output,
                                         run_id="a" * 32, artifact_path="training/checkpoint-338")
        prepared = Path(self.plan["prepared_output_dir"])
        self.settings = dict(output_dir=str(self.output), resume_from_checkpoint=str(prepared / self.source.name),
                             adapter_name_or_path=str(prepared / self.source.name), create_new_adapter=False,
                             num_train_epochs=4, save_total_limit=2, save_only_model=False,
                             load_best_model_at_end=True, eval_steps=50, save_steps=50,
                             learning_rate=.0001, lr_scheduler_type="cosine", fp16=True,
                             use_hard_example_loss=True)
        self.config.write_text(yaml.safe_dump(self.settings))
        return prepared

    def activate(self, prepared):
        with mock.patch.object(storage.shutil, "disk_usage", return_value=SimpleNamespace(free=10**12)):
            return Path(storage.activate_resume(prepared, self.output, self.config, self.root))

    def test_move_only_latest_and_best_then_rotation_releases_old_weights(self):
        original = (self.source / "trainer_state.json").read_bytes()
        prepared = self.stage()
        self.assertFalse(self.source.exists())
        self.assertFalse(self.best.exists())
        self.assertEqual((self.cache / "source_trainer_state.json").read_bytes(), original)
        active_plan = self.activate(prepared)
        self.assertFalse(prepared.exists())
        self.assertEqual(list(self.cache.rglob("*.safetensors")), [])
        self.assertEqual(list(self.cache.rglob("optimizer.pt")), [])
        self.assertTrue(json.loads(active_plan.read_text())["activated"])
        settings = yaml.safe_load(self.config.read_text())
        expected = dict(self.settings, resume_from_checkpoint=str(self.output / "checkpoint-338"),
                        adapter_name_or_path=str(self.output / "checkpoint-338"))
        self.assertEqual(settings, expected)  # No LR/loss/precision/epoch changes.
        state = json.loads((self.output / "checkpoint-338/trainer_state.json").read_text())
        self.assertEqual(state["global_step"], 338)
        self.assertEqual(state["best_metric"], .09)
        self.assertEqual(state["best_model_checkpoint"], str(self.output / "checkpoint-300"))
        for step, remove in ((350, 338), (400, 350), (450, 300), (500, 400)):
            self.checkpoint(self.output / f"checkpoint-{step}", step=step)
            self.assertEqual(len(list(self.output.glob("checkpoint-*"))), 3)
            shutil.rmtree(self.output / f"checkpoint-{remove}")
            weights = list(self.root.rglob("*.safetensors"))
            self.assertEqual(len(weights), 2)
            self.assertTrue(all(p.stat().st_nlink == 1 for p in weights))

    def test_best_equals_latest_uses_one_directory_and_no_extra_download(self):
        prepared = self.stage(same_best=True)
        self.assertEqual(len(list(prepared.glob("checkpoint-*"))), 1)
        self.activate(prepared)
        self.assertEqual(len(list(self.output.glob("checkpoint-*"))), 1)

    def test_no_best_still_stages_full_resume_without_hidden_cache(self):
        shutil.rmtree(self.best)
        plan = storage.stage_resume(self.source, None, self.cache, self.output,
                                     run_id="a" * 32, artifact_path="checkpoints/checkpoint-338")
        self.assertIsNone(plan["best_checkpoint_name"])
        self.assertFalse(self.source.exists())
        self.assertTrue((Path(plan["prepared_output_dir"]) / "checkpoint-338/optimizer.pt").is_file())

    def test_existing_output_is_never_overwritten(self):
        prepared = self.stage()
        self.output.mkdir()
        marker = self.output / "user-file"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.activate(prepared)
        self.assertEqual(marker.read_text(), "keep")
        self.assertTrue(prepared.exists())

    def test_tampered_stage_stops_before_move(self):
        prepared = self.stage()
        (prepared / "checkpoint-338/optimizer.pt").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Prepared resume"):
            self.activate(prepared)
        self.assertFalse(self.output.exists())

    def test_low_space_stops_before_activation_or_yaml_change(self):
        prepared = self.stage()
        original = self.config.read_bytes()
        with mock.patch.object(storage.shutil, "disk_usage", return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(RuntimeError, "Insufficient checkpoint disk"):
                storage.activate_resume(prepared, self.output, self.config, self.root)
        self.assertFalse(self.output.exists())
        self.assertTrue(prepared.exists())
        self.assertEqual(self.config.read_bytes(), original)

    def test_runtime_mismatch_does_not_fall_back_to_fresh(self):
        prepared = self.stage()
        self.settings["create_new_adapter"] = True
        self.config.write_text(yaml.safe_dump(self.settings))
        with self.assertRaisesRegex(ValueError, "Runtime YAML"):
            self.activate(prepared)
        self.assertFalse(self.output.exists())

    def test_guard_budgets_save_before_rotation_and_one_gib_reserve(self):
        self.output.mkdir()
        self.checkpoint(self.output / "checkpoint-450", step=450)
        self.checkpoint(self.output / "checkpoint-400", step=400)
        checkpoint_size = 2 * 1024**3
        with mock.patch.object(storage.shutil, "disk_usage", return_value=SimpleNamespace(free=3 * 1024**3)):
            storage.require_space(self.output, checkpoint_size, future_peak=True)
        with mock.patch.object(storage.shutil, "disk_usage", return_value=SimpleNamespace(free=3 * 1024**3 - 1)):
            with self.assertRaisesRegex(RuntimeError, "save-before-rotation"):
                storage.require_space(self.output, checkpoint_size, future_peak=True)

    def test_callback_checks_before_save_without_changing_training_control(self):
        prepared = self.stage()
        plan = self.activate(prepared)
        callback = storage.make_storage_callback(plan)
        args = SimpleNamespace(output_dir=str(self.output), save_total_limit=2)
        state = SimpleNamespace(is_world_process_zero=True)
        control = SimpleNamespace(should_save=False, should_training_stop=False)
        with mock.patch.object(storage, "require_space") as check:
            callback.on_train_begin(args, state, control)
            self.assertTrue(check.call_args.kwargs["future_peak"])
            check.reset_mock()
            callback.on_step_end(args, state, control)
            check.assert_not_called()
            control.should_save = True
            callback.on_step_end(args, state, control)
            self.assertFalse(check.call_args.kwargs["future_peak"])
        self.assertTrue(control.should_save)
        self.assertFalse(control.should_training_stop)
        before = storage._files(self.output)
        with mock.patch.object(storage.shutil, "disk_usage", return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(RuntimeError, "no partial checkpoint"):
                callback.on_step_end(args, state, control)
        self.assertEqual(storage._files(self.output), before)

    def test_actual_cpu_trainer_resumes_optimizer_rng_and_keeps_best_latest(self):
        import os
        import torch
        from transformers import TrainerCallback
        from test_hard_example_loss import (
            TinyCausalConfig, TinyCausalPreTrainedModel, TinyDictDataset, DummyProcessor,
            DummyTokenizer, collate_batch, FinetuningArguments, CustomSeq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
        dataset = TinyDictDataset([{
            "input_ids": torch.tensor([1, 2, 3, 4]), "attention_mask": torch.ones(4, dtype=torch.long),
            "labels": torch.tensor([-100, 2, 3, 4]),
        }] * 2)
        def trainer(output, steps, callbacks=None):
            args = Seq2SeqTrainingArguments(
                output_dir=str(output), max_steps=steps, per_device_train_batch_size=2,
                per_device_eval_batch_size=2, use_cpu=True, report_to="none", disable_tqdm=True,
                save_strategy="steps", eval_strategy="steps", save_steps=1, eval_steps=1,
                logging_steps=1, save_total_limit=2, load_best_model_at_end=True,
                metric_for_best_model="eval_loss", greater_is_better=False,
            )
            return CustomSeq2SeqTrainer(
                model=TinyCausalPreTrainedModel(TinyCausalConfig()), args=args,
                finetuning_args=FinetuningArguments(use_hard_example_loss=True),
                processor=DummyProcessor(), tokenizer=DummyTokenizer(), data_collator=collate_batch,
                train_dataset=dataset, eval_dataset=dataset, callbacks=callbacks,
            )
        first_output = self.root / "parent"
        with mock.patch.dict(os.environ):
            os.environ.pop("CHECKPOINT_STORAGE_PLAN", None)
            first = trainer(first_output, 2)
            first.train()
        current = self.root / "real-cache/checkpoint-2"
        shutil.copytree(first_output / "checkpoint-2", current)
        state = json.loads((current / "trainer_state.json").read_text())
        best_name = Path(state["best_model_checkpoint"]).name
        best = current if best_name == current.name else current.parent / best_name
        if best != current:
            shutil.copytree(first_output / best_name, best)
        output = self.root / "real-run/checkpoints"
        output.parent.mkdir()
        plan = storage.stage_resume(current, best, current.parent, output,
                                    run_id="b" * 32, artifact_path="checkpoints/checkpoint-2")
        prepared = Path(plan["prepared_output_dir"])
        config = output.parent / "runtime.yaml"
        config.write_text(yaml.safe_dump(dict(
            output_dir=str(output), resume_from_checkpoint=str(prepared / "checkpoint-2"),
            adapter_name_or_path=str(prepared / "checkpoint-2"), create_new_adapter=False,
            load_best_model_at_end=True, save_total_limit=2,
        )))
        plan_path = storage.activate_resume(prepared, output, config, self.root)
        observed = {}
        class ObserveResume(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                observed["step"] = state.global_step
                observed["optimizer_steps"] = [float(v["step"]) for v in
                                               kwargs["optimizer"].state_dict()["state"].values()]
                observed["scheduler_epoch"] = kwargs["lr_scheduler"].state_dict()["last_epoch"]
        with mock.patch.dict(os.environ, CHECKPOINT_STORAGE_PLAN=plan_path):
            resumed = trainer(output, 4, callbacks=[ObserveResume()])
        self.assertIn("CheckpointStorageCallback", [type(c).__name__ for c in resumed.callback_handler.callbacks])
        with mock.patch.object(resumed, "_load_rng_state", wraps=resumed._load_rng_state) as load_rng:
            result = resumed.train(resume_from_checkpoint=str(output / "checkpoint-2"))
        self.assertEqual(observed["step"], 2)
        self.assertTrue(observed["optimizer_steps"])
        self.assertTrue(all(step == 2 for step in observed["optimizer_steps"]))
        self.assertEqual(observed["scheduler_epoch"], 2)
        load_rng.assert_called()
        self.assertEqual(result.global_step, 4)
        self.assertTrue((output / "checkpoint-4/optimizer.pt").is_file())
        self.assertTrue(Path(resumed.state.best_model_checkpoint).is_dir())
        self.assertLessEqual(len(list(output.glob("checkpoint-*"))), 2)
        self.assertEqual(list(current.parent.rglob("optimizer.pt")), [])


if __name__ == "__main__":
    unittest.main()

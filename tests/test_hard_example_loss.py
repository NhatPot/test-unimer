"""CPU unit and trainer-integration tests for hard-example-weighted SFT loss."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers.modeling_outputs import CausalLMOutputWithPast


ROOT = Path(__file__).resolve().parents[1]
LLAMAFACTORY_SRC = ROOT / "train" / "LLaMA-Factory" / "src"
if str(LLAMAFACTORY_SRC) not in sys.path:
    sys.path.insert(0, str(LLAMAFACTORY_SRC))

from llamafactory.extras.constants import IGNORE_INDEX  # noqa: E402
from llamafactory.hparams import FinetuningArguments, ModelArguments  # noqa: E402
from llamafactory.train.sft.hard_example_loss import (  # noqa: E402
    DEFAULT_HARD_EXAMPLE_LOSS_ALPHA,
    DEFAULT_HARD_EXAMPLE_LOSS_MAX_WEIGHT,
    HardExampleLossDetails,
    compute_hard_example_weighted_loss,
    validate_hard_example_loss_config,
)
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer  # noqa: E402


class TinyCausalConfig(PretrainedConfig):
    model_type = "tiny_causal"

    def __init__(self, vocab_size: int = 32, hidden_size: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size


class TinyCausalPreTrainedModel(PreTrainedModel):
    config_class = TinyCausalConfig

    def __init__(self, config: TinyCausalConfig):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.forward_calls: list[dict[str, bool]] = []
        self.post_init()

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        return_dict: bool = True,
        **kwargs,
    ):
        self.forward_calls.append(
            {
                "has_labels": labels is not None,
                "has_num_items_in_batch": "num_items_in_batch" in kwargs,
                "training": self.training,
            }
        )
        hidden = self.embed(input_ids)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = ForCausalLMLoss(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                num_items_in_batch=kwargs.get("num_items_in_batch"),
                ignore_index=IGNORE_INDEX,
            )
        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)
        return CausalLMOutputWithPast(loss=loss, logits=logits)


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def save_pretrained(self, save_directory: str, **kwargs):
        return ()


class DummyProcessor:
    def save_pretrained(self, save_directory: str, **kwargs):
        return ()


class HardExampleLossHelperTests(unittest.TestCase):
    """Pure helper unit tests for `compute_hard_example_weighted_loss`."""

    def test_defaults_match_pilot_spec(self):
        self.assertEqual(DEFAULT_HARD_EXAMPLE_LOSS_ALPHA, 0.5)
        self.assertEqual(DEFAULT_HARD_EXAMPLE_LOSS_MAX_WEIGHT, 1.5)
        args = FinetuningArguments()
        self.assertFalse(args.use_hard_example_loss)
        self.assertEqual(args.hard_example_loss_alpha, 0.5)
        self.assertEqual(args.hard_example_loss_max_weight, 1.5)

    def test_uniform_weights_match_ordinary_shifted_ce_value_and_gradients(self):
        torch.manual_seed(123)
        batch_size, seq_len, vocab_size = 4, 19, 47
        base_logits = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.float32)
        labels = torch.full((batch_size, seq_len), IGNORE_INDEX, dtype=torch.long)
        # Variable-length valid answers with masked prompt prefixes and padding suffixes
        labels[0, 4:9] = torch.randint(0, vocab_size, (5,))
        labels[1, 2:17] = torch.randint(0, vocab_size, (15,))
        labels[2, 10:12] = torch.randint(0, vocab_size, (2,))
        labels[3, 1:19] = torch.randint(0, vocab_size, (18,))

        logits_ref = base_logits.clone().requires_grad_(True)
        loss_ref = ForCausalLMLoss(logits_ref, labels, vocab_size=vocab_size, ignore_index=IGNORE_INDEX)
        loss_ref.backward()

        for alpha, max_weight in ((0.0, 1.5), (0.5, 1.0), (0.0, 1.0)):
            with self.subTest(alpha=alpha, max_weight=max_weight):
                logits_test = base_logits.clone().requires_grad_(True)
                loss_test, details = compute_hard_example_weighted_loss(
                    logits_test,
                    labels,
                    alpha=alpha,
                    max_weight=max_weight,
                    return_details=True,
                )
                loss_test.backward()

                self.assertIsInstance(details, HardExampleLossDetails)
                self.assertTrue(torch.allclose(details.weights, torch.ones_like(details.weights)))
                self.assertEqual(details.valid_token_counts.tolist(), [5, 15, 2, 18])
                self.assertTrue(torch.allclose(loss_test, loss_ref, atol=1e-6, rtol=1e-6))
                self.assertTrue(torch.allclose(logits_test.grad, logits_ref.grad, atol=1e-6, rtol=1e-6))

    def test_ignored_prompt_padding_and_unshifted_positions_get_zero_gradients(self):
        torch.manual_seed(7)
        batch_size, seq_len, vocab_size = 2, 10, 23
        logits = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.float32, requires_grad=True)
        labels = torch.full((batch_size, seq_len), IGNORE_INDEX, dtype=torch.long)
        # Example 0: valid targets at label indices 4, 5 (predicted by logit indices 3, 4)
        labels[0, 4] = 3
        labels[0, 5] = 8
        # Example 1: valid target at label index 7 (predicted by logit index 6)
        labels[1, 7] = 11

        loss = compute_hard_example_weighted_loss(logits, labels, alpha=0.5, max_weight=1.5)
        loss.backward()

        grad = logits.grad
        self.assertIsNotNone(grad)
        # Example 0: only logit positions 3 and 4 should have non-zero gradients
        for pos in range(seq_len):
            if pos in (3, 4):
                self.assertGreater(grad[0, pos].abs().sum().item(), 0.0)
            else:
                self.assertEqual(grad[0, pos].abs().max().item(), 0.0)

        # Example 1: only logit position 6 should have non-zero gradients
        for pos in range(seq_len):
            if pos == 6:
                self.assertGreater(grad[1, pos].abs().sum().item(), 0.0)
            else:
                self.assertEqual(grad[1, pos].abs().max().item(), 0.0)

        # Final logit position (seq_len - 1) never predicts a next token
        self.assertEqual(grad[:, -1].abs().max().item(), 0.0)

    def test_causal_shift_correctness_and_first_label_ignored(self):
        vocab_size = 5
        logits = torch.zeros(1, 3, vocab_size, dtype=torch.float32, requires_grad=True)
        with torch.no_grad():
            # Position 0 strongly predicts token 2; position 1 strongly predicts token 4
            logits[0, 0, 2] = 10.0
            logits[0, 1, 4] = 10.0
            logits[0, 2, 1] = 10.0

        # labels[:, 0] is not predicted by any logit; labels[0, 1]=2 and labels[0, 2]=4 match shifted logits
        labels_a = torch.tensor([[0, 2, 4]], dtype=torch.long)
        labels_b = torch.tensor([[IGNORE_INDEX, 2, 4]], dtype=torch.long)

        loss_a = compute_hard_example_weighted_loss(logits, labels_a, alpha=0.5, max_weight=1.5)
        loss_a.backward()
        grad_a = logits.grad.detach().clone()

        logits.grad = None
        loss_b = compute_hard_example_weighted_loss(logits, labels_b, alpha=0.5, max_weight=1.5)
        loss_b.backward()
        grad_b = logits.grad.detach().clone()

        self.assertTrue(torch.allclose(loss_a, loss_b, atol=1e-7))
        self.assertTrue(torch.allclose(grad_a, grad_b, atol=1e-7))
        self.assertLess(loss_a.item(), 1e-3)

        # Wrong unshifted targets (if shift were missing) would have high loss
        labels_wrong = torch.tensor([[2, 4, IGNORE_INDEX]], dtype=torch.long)
        loss_wrong = compute_hard_example_weighted_loss(logits, labels_wrong, alpha=0.5, max_weight=1.5)
        self.assertGreater(loss_wrong.item(), 9.0)

    def test_bounded_weights_and_harder_example_gets_larger_weight(self):
        vocab_size = 8
        # 3 examples, seq_len=6
        logits = torch.zeros(3, 6, vocab_size, dtype=torch.float32, requires_grad=True)
        labels = torch.full((3, 6), IGNORE_INDEX, dtype=torch.long)

        with torch.no_grad():
            # Example 0 (easy, 4 valid tokens): strongly predicts target 1 -> very small per-token CE
            logits[0, :4, 1] = 8.0
            labels[0, 1:5] = 1

            # Example 1 (medium, 2 valid tokens): zero logits -> per-token CE = log(8) ≈ 2.0794
            labels[1, 1:3] = 2

            # Example 2 (very hard, 1 valid token): strongly predicts wrong token -> huge per-token CE
            logits[2, 0, 3] = 20.0
            labels[2, 1] = 4

        loss_weighted, details = compute_hard_example_weighted_loss(
            logits, labels, alpha=0.5, max_weight=1.5, return_details=True
        )
        loss_uniform = compute_hard_example_weighted_loss(logits, labels, alpha=0.0, max_weight=1.0)

        self.assertTrue(torch.all(details.weights >= 1.0))
        self.assertTrue(torch.all(details.weights <= 1.5))
        # Example 0 is below mean difficulty -> clamped to 1.0
        self.assertAlmostEqual(details.weights[0].item(), 1.0, places=6)
        # Harder example has strictly larger difficulty and larger weight
        self.assertLess(details.difficulties[0].item(), details.difficulties[1].item())
        self.assertLess(details.difficulties[1].item(), details.difficulties[2].item())
        self.assertGreater(details.weights[2].item(), details.weights[0].item())
        # Example 2 is extreme -> hits max_weight cap 1.5
        self.assertAlmostEqual(details.weights[2].item(), 1.5, places=6)
        # Upweighting the hard example increases the weighted mean CE over uniform mean CE
        self.assertGreater(loss_weighted.item(), loss_uniform.item())

    def test_difficulty_and_weights_are_detached_from_gradients(self):
        torch.manual_seed(99)
        batch_size, seq_len, vocab_size = 2, 7, 16
        base_logits = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.float32)
        with torch.no_grad():
            # Make example 1 moderately harder than example 0 so weight_1 is in the active (1.0, 1.5) linear region
            base_logits[0, :, 0] += 3.0
            base_logits[1, :, 0] -= 1.0
        labels = torch.full((batch_size, seq_len), IGNORE_INDEX, dtype=torch.long)
        labels[0, 1:5] = 0
        labels[1, 1:4] = 0

        logits = base_logits.clone().requires_grad_(True)
        loss, details = compute_hard_example_weighted_loss(
            logits, labels, alpha=0.5, max_weight=1.5, return_details=True
        )
        self.assertFalse(details.difficulties.requires_grad)
        self.assertFalse(details.mean_difficulty.requires_grad)
        self.assertFalse(details.weights.requires_grad)
        self.assertGreater(details.weights[1].item(), 1.0)
        self.assertLess(details.weights[1].item(), 1.5)
        loss.backward()

        # Reference where weights are treated as fixed constants
        logits_const = base_logits.clone().requires_grad_(True)
        ce0 = F.cross_entropy(logits_const[0, :4], labels[0, 1:5], reduction="sum")
        ce1 = F.cross_entropy(logits_const[1, :3], labels[1, 1:4], reduction="sum")
        w0, w1 = details.weights[0].item(), details.weights[1].item()
        loss_const = (w0 * ce0 + w1 * ce1) / (w0 * 4.0 + w1 * 3.0)
        loss_const.backward()

        self.assertTrue(torch.allclose(loss, loss_const, atol=1e-7))
        self.assertTrue(torch.allclose(logits.grad, logits_const.grad, atol=1e-7))

        # Reference where weights are NOT detached (gradients flow through difficulty/mean_difficulty)
        logits_attached = base_logits.clone().requires_grad_(True)
        ce0_a = F.cross_entropy(logits_attached[0, :4], labels[0, 1:5], reduction="sum")
        ce1_a = F.cross_entropy(logits_attached[1, :3], labels[1, 1:4], reduction="sum")
        d0_a = ce0_a / 4.0
        d1_a = ce1_a / 3.0
        mean_d_a = 0.5 * (d0_a + d1_a)
        w0_a = torch.clamp(1.0 + 0.5 * F.relu(d0_a / mean_d_a - 1.0), 1.0, 1.5)
        w1_a = torch.clamp(1.0 + 0.5 * F.relu(d1_a / mean_d_a - 1.0), 1.0, 1.5)
        loss_attached = (w0_a * ce0_a + w1_a * ce1_a) / (w0_a * 4.0 + w1_a * 3.0)
        loss_attached.backward()

        # Detached gradient must differ from the un-detached gradient
        self.assertFalse(torch.allclose(logits.grad, logits_attached.grad, atol=1e-4))

    def test_partial_and_full_all_masked_and_empty_examples(self):
        torch.manual_seed(2026)
        vocab_size = 11

        # 1. Partial all-masked batch: example 1 has all IGNORE_INDEX, examples 0 and 2 have valid tokens
        logits_3 = torch.randn(3, 6, vocab_size, dtype=torch.float32, requires_grad=True)
        labels_3 = torch.full((3, 6), IGNORE_INDEX, dtype=torch.long)
        labels_3[0, 2:5] = torch.tensor([1, 2, 3])
        labels_3[2, 1:3] = torch.tensor([4, 5])

        logits_2 = logits_3.detach()[[0, 2]].clone().requires_grad_(True)
        labels_2 = labels_3[[0, 2]].clone()

        loss_3, details_3 = compute_hard_example_weighted_loss(
            logits_3, labels_3, alpha=0.5, max_weight=1.5, return_details=True
        )
        loss_2, details_2 = compute_hard_example_weighted_loss(
            logits_2, labels_2, alpha=0.5, max_weight=1.5, return_details=True
        )
        loss_3.backward()
        loss_2.backward()

        # All-masked example 1 must not change mean_difficulty, valid weights, loss, or valid gradients
        self.assertTrue(torch.allclose(details_3.mean_difficulty, details_2.mean_difficulty, atol=1e-7))
        self.assertTrue(torch.allclose(details_3.weights[[0, 2]], details_2.weights, atol=1e-7))
        self.assertTrue(torch.allclose(loss_3, loss_2, atol=1e-7))
        self.assertEqual(logits_3.grad[1].abs().max().item(), 0.0)
        self.assertTrue(torch.allclose(logits_3.grad[[0, 2]], logits_2.grad, atol=1e-7))

        # 2. Full all-masked batch
        logits_masked = torch.randn(2, 5, vocab_size, dtype=torch.float32, requires_grad=True)
        labels_masked = torch.full((2, 5), IGNORE_INDEX, dtype=torch.long)
        loss_masked, details_masked = compute_hard_example_weighted_loss(
            logits_masked, labels_masked, alpha=0.5, max_weight=1.5, return_details=True
        )
        self.assertEqual(loss_masked.ndim, 0)
        self.assertTrue(torch.isfinite(loss_masked).item())
        self.assertEqual(loss_masked.item(), 0.0)
        self.assertTrue(loss_masked.requires_grad)
        loss_masked.backward()
        self.assertEqual(logits_masked.grad.abs().max().item(), 0.0)
        self.assertEqual(details_masked.valid_token_counts.tolist(), [0, 0])

        # 3. seq_len == 1 (no shifted tokens) and batch_size == 0
        for shape, lshape in (((2, 1, vocab_size), (2, 1)), ((0, 4, vocab_size), (0, 4))):
            with self.subTest(shape=shape):
                logits_empty = torch.randn(*shape, dtype=torch.float32, requires_grad=True)
                labels_empty = torch.zeros(*lshape, dtype=torch.long)
                loss_empty = compute_hard_example_weighted_loss(logits_empty, labels_empty)
                self.assertEqual(loss_empty.ndim, 0)
                self.assertTrue(torch.isfinite(loss_empty).item())
                self.assertEqual(loss_empty.item(), 0.0)
                loss_empty.backward()
                self.assertEqual(logits_empty.grad.abs().sum().item(), 0.0)

        # 4. Zero mean difficulty (perfect predictions where valid token CE == 0.0)
        logits_zero_ce = torch.full((2, 3, 4), -100.0, dtype=torch.float32, requires_grad=True)
        with torch.no_grad():
            logits_zero_ce[:, :, 1] = 100.0
        labels_zero_ce = torch.tensor([[IGNORE_INDEX, 1, 1], [IGNORE_INDEX, 1, IGNORE_INDEX]], dtype=torch.long)
        loss_zero_ce, details_zero_ce = compute_hard_example_weighted_loss(
            logits_zero_ce, labels_zero_ce, return_details=True
        )
        self.assertTrue(torch.isfinite(loss_zero_ce).item())
        self.assertEqual(loss_zero_ce.item(), 0.0)
        self.assertTrue(torch.allclose(details_zero_ce.weights, torch.ones_like(details_zero_ce.weights)))

    def test_fp16_cpu_input_upcasts_to_fp32_and_preserves_finite_gradients(self):
        torch.manual_seed(55)
        batch_size, seq_len, vocab_size = 2, 8, 32
        # Include large half-precision values (~30.0) to verify FP32 softmax stability
        logits_fp16 = (torch.randn(batch_size, seq_len, vocab_size, dtype=torch.float16) * 10.0).requires_grad_(True)
        logits_fp32 = logits_fp16.detach().float().requires_grad_(True)
        labels = torch.full((batch_size, seq_len), IGNORE_INDEX, dtype=torch.long)
        labels[0, 2:7] = torch.randint(0, vocab_size, (5,))
        labels[1, 4:6] = torch.randint(0, vocab_size, (2,))

        loss_fp16 = compute_hard_example_weighted_loss(logits_fp16, labels, alpha=0.5, max_weight=1.5)
        loss_fp32 = compute_hard_example_weighted_loss(logits_fp32, labels, alpha=0.5, max_weight=1.5)

        self.assertEqual(loss_fp16.dtype, torch.float32)
        self.assertEqual(loss_fp16.ndim, 0)
        self.assertTrue(torch.isfinite(loss_fp16).item())
        self.assertTrue(torch.allclose(loss_fp16, loss_fp32, atol=1e-5, rtol=1e-5))

        loss_fp16.backward()
        loss_fp32.backward()
        self.assertEqual(logits_fp16.grad.dtype, torch.float16)
        self.assertTrue(torch.isfinite(logits_fp16.grad).all().item())
        self.assertTrue(torch.allclose(logits_fp16.grad.float(), logits_fp32.grad, atol=1e-3, rtol=1e-3))

    def test_real_nonfinite_logits_are_not_silently_masked(self):
        logits = torch.zeros(2, 4, 6, dtype=torch.float32, requires_grad=True)
        with torch.no_grad():
            logits[0, 0, 1] = float("nan")
        labels = torch.tensor([[IGNORE_INDEX, 1, 2, 3], [IGNORE_INDEX, 2, 3, 4]], dtype=torch.long)
        loss = compute_hard_example_weighted_loss(logits, labels, alpha=0.5, max_weight=1.5)
        self.assertFalse(torch.isfinite(loss).item())

    def test_invalid_helper_arguments_rejected(self):
        logits = torch.zeros(2, 4, 6)
        labels = torch.zeros(2, 4, dtype=torch.long)

        for bad_alpha in (-0.1, -1, float("nan"), float("inf"), -float("inf"), True, False, None, "0.5"):
            with self.subTest(bad_alpha=bad_alpha), self.assertRaises(ValueError):
                validate_hard_example_loss_config(alpha=bad_alpha, max_weight=1.5)
            with self.subTest(bad_alpha_call=bad_alpha), self.assertRaises(ValueError):
                compute_hard_example_weighted_loss(logits, labels, alpha=bad_alpha, max_weight=1.5)

        for bad_max_weight in (0.99, 0.0, -1.0, float("nan"), float("inf"), -float("inf"), True, False, None, "1.5"):
            with self.subTest(bad_max_weight=bad_max_weight), self.assertRaises(ValueError):
                validate_hard_example_loss_config(alpha=0.5, max_weight=bad_max_weight)
            with self.subTest(bad_max_weight_call=bad_max_weight), self.assertRaises(ValueError):
                compute_hard_example_weighted_loss(logits, labels, alpha=0.5, max_weight=bad_max_weight)

        with self.assertRaises(TypeError):
            compute_hard_example_weighted_loss([[1.0]], labels)
        with self.assertRaises(ValueError):
            compute_hard_example_weighted_loss(torch.zeros(2, 4), labels)
        with self.assertRaises(ValueError):
            compute_hard_example_weighted_loss(logits, torch.zeros(2, 5, dtype=torch.long))


class TinyDictDataset(torch.utils.data.Dataset):
    def __init__(self, items: list[dict[str, torch.Tensor]]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        return {k: v.clone() for k, v in self.items[idx].items()}


def collate_batch(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([f["input_ids"] for f in features], dim=0),
        "attention_mask": torch.stack([f["attention_mask"] for f in features], dim=0),
        "labels": torch.stack([f["labels"] for f in features], dim=0),
    }


class CustomSeq2SeqTrainerHardExampleIntegrationTests(unittest.TestCase):
    """Actual CPU trainer-path integration tests without model downloads."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output_dir = str(Path(self.temp.name) / "trainer_out")
        torch.manual_seed(314)
        self.config = TinyCausalConfig(vocab_size=24, hidden_size=12)
        self.base_model = TinyCausalPreTrainedModel(self.config)
        # Skew lm_head weights so different target tokens have markedly different difficulty
        with torch.no_grad():
            self.base_model.lm_head.weight[2] *= 4.0
            self.base_model.lm_head.weight[19] *= -4.0

        self.batch = {
            "input_ids": torch.tensor(
                [
                    [1, 2, 3, 4, 5, 0],
                    [6, 7, 8, 9, 10, 11],
                ],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1],
                ],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                [
                    [IGNORE_INDEX, IGNORE_INDEX, 2, 2, 2, IGNORE_INDEX],
                    [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 19, 19, IGNORE_INDEX],
                ],
                dtype=torch.long,
            ),
        }

    def clone_model(self) -> TinyCausalPreTrainedModel:
        model = TinyCausalPreTrainedModel(self.config)
        model.load_state_dict(self.base_model.state_dict())
        return model

    def make_training_args(self, **overrides) -> Seq2SeqTrainingArguments:
        kwargs = dict(
            output_dir=self.output_dir,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=1e-3,
            num_train_epochs=1,
            max_steps=1,
            logging_steps=1,
            save_strategy="no",
            eval_strategy="no",
            report_to="none",
            use_cpu=True,
            remove_unused_columns=False,
        )
        kwargs.update(overrides)
        return Seq2SeqTrainingArguments(**kwargs)

    def make_trainer(
        self,
        model: nn.Module = None,
        finetuning_args: FinetuningArguments = None,
        training_args: Seq2SeqTrainingArguments = None,
        model_args: ModelArguments = None,
        processor=DummyProcessor(),
        train_dataset=None,
        eval_dataset=None,
        **kwargs,
    ) -> CustomSeq2SeqTrainer:
        model = self.clone_model() if model is None else model
        finetuning_args = FinetuningArguments() if finetuning_args is None else finetuning_args
        training_args = self.make_training_args() if training_args is None else training_args
        return CustomSeq2SeqTrainer(
            model=model,
            args=training_args,
            finetuning_args=finetuning_args,
            model_args=model_args,
            processor=processor,
            tokenizer=DummyTokenizer(),
            data_collator=collate_batch,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs,
        )

    def test_finetuning_args_and_trainer_reject_invalid_and_incompatible_configs(self):
        # Invalid FinetuningArguments fields
        for invalid_kwargs in (
            {"use_hard_example_loss": "yes"},
            {"hard_example_loss_alpha": -0.01},
            {"hard_example_loss_alpha": float("nan")},
            {"hard_example_loss_alpha": float("inf")},
            {"hard_example_loss_alpha": True},
            {"hard_example_loss_max_weight": 0.95},
            {"hard_example_loss_max_weight": float("nan")},
            {"hard_example_loss_max_weight": False},
            {"use_hard_example_loss": True, "stage": "pt"},
            {"use_hard_example_loss": True, "stage": "dpo"},
            {"use_hard_example_loss": True, "use_dft_loss": True},
        ):
            with self.subTest(invalid_kwargs=invalid_kwargs), self.assertRaises(ValueError):
                FinetuningArguments(**invalid_kwargs)

        # Incompatible label smoothing
        with self.assertRaisesRegex(ValueError, "label smoothing"):
            self.make_trainer(
                finetuning_args=FinetuningArguments(use_hard_example_loss=True),
                training_args=self.make_training_args(label_smoothing_factor=0.1),
            )

        # Incompatible custom compute_loss_func
        with self.assertRaisesRegex(ValueError, "compute_loss_func"):
            self.make_trainer(
                finetuning_args=FinetuningArguments(use_hard_example_loss=True),
                compute_loss_func=lambda outputs, labels, num_items_in_batch=None: torch.tensor(0.0),
            )

        # Incompatible liger kernel via ModelArguments or TrainingArguments
        model_args_liger = ModelArguments(model_name_or_path="dummy/model", enable_liger_kernel=True)
        with self.assertRaisesRegex(ValueError, "liger kernel"):
            self.make_trainer(
                finetuning_args=FinetuningArguments(use_hard_example_loss=True),
                model_args=model_args_liger,
            )

        training_args_liger = self.make_training_args()
        training_args_liger.use_liger_kernel = True
        with self.assertRaisesRegex(ValueError, "liger kernel"):
            self.make_trainer(
                finetuning_args=FinetuningArguments(use_hard_example_loss=True),
                training_args=training_args_liger,
            )

        # Missing labels in training inputs
        trainer = self.make_trainer(finetuning_args=FinetuningArguments(use_hard_example_loss=True))
        trainer.model.train()
        with self.assertRaisesRegex(ValueError, "requires `labels`"):
            trainer.compute_loss(trainer.model, {"input_ids": self.batch["input_ids"]})

        # Model omitting logits (e.g., fused CE kernel without logits)
        class NoLogitsModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(1))

            def forward(self, **kwargs):
                return {"loss": self.weight.sum()}

        no_logits_model = NoLogitsModel()
        no_logits_model.train()
        with self.assertRaisesRegex(ValueError, "logits"):
            trainer.compute_loss(no_logits_model, dict(self.batch))

        # Post-init mutation of label_smoother or compute_loss_func is also caught at compute_loss time
        trainer_mut = self.make_trainer(finetuning_args=FinetuningArguments(use_hard_example_loss=True))
        trainer_mut.model.train()
        trainer_mut.label_smoother = lambda *a, **kw: torch.tensor(0.0)
        with self.assertRaisesRegex(ValueError, "label smoothing"):
            trainer_mut.compute_loss(trainer_mut.model, dict(self.batch))
        trainer_mut.label_smoother = None
        trainer_mut.compute_loss_func = lambda *a, **kw: torch.tensor(0.0)
        with self.assertRaisesRegex(ValueError, "compute_loss_func"):
            trainer_mut.compute_loss(trainer_mut.model, dict(self.batch))

    def test_hf_argument_parser_parses_hard_example_loss_fields(self):
        from transformers import HfArgumentParser

        parser = HfArgumentParser((FinetuningArguments,))
        (parsed_default,) = parser.parse_dict({})
        self.assertFalse(parsed_default.use_hard_example_loss)
        self.assertEqual(parsed_default.hard_example_loss_alpha, 0.5)
        self.assertEqual(parsed_default.hard_example_loss_max_weight, 1.5)

        (parsed_custom,) = parser.parse_dict(
            {
                "stage": "sft",
                "use_hard_example_loss": True,
                "hard_example_loss_alpha": 0.75,
                "hard_example_loss_max_weight": 1.25,
            }
        )
        self.assertTrue(parsed_custom.use_hard_example_loss)
        self.assertEqual(parsed_custom.hard_example_loss_alpha, 0.75)
        self.assertEqual(parsed_custom.hard_example_loss_max_weight, 1.25)

    def test_tuple_model_output_and_past_index_handling(self):
        class TupleOutputModel(nn.Module):
            def __init__(self, base: TinyCausalPreTrainedModel):
                super().__init__()
                self.base = base

            def forward(self, input_ids, attention_mask=None, labels=None):
                out = self.base(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=False,
                )
                return out + ("past_state_sentinel",)

        tuple_model = TupleOutputModel(self.clone_model())
        tuple_model.train()
        training_args = self.make_training_args(past_index=2)
        trainer = self.make_trainer(
            model=tuple_model,
            finetuning_args=FinetuningArguments(use_hard_example_loss=True),
            training_args=training_args,
        )
        loss, outputs = trainer.compute_loss(tuple_model, dict(self.batch), return_outputs=True)
        self.assertIsInstance(outputs, tuple)
        self.assertIs(outputs[0], loss)
        self.assertEqual(outputs[1].shape, (2, 6, self.config.vocab_size))
        self.assertEqual(trainer._past, "past_state_sentinel")

    def test_flag_off_reproduces_superclass_in_train_and_eval(self):
        trainer = self.make_trainer(finetuning_args=FinetuningArguments(use_hard_example_loss=False))
        for is_train in (True, False):
            with self.subTest(is_train=is_train):
                model_a = self.clone_model()
                model_b = self.clone_model()
                model_a.train(is_train)
                model_b.train(is_train)

                loss_a = trainer.compute_loss(model_a, dict(self.batch))
                loss_b = Seq2SeqTrainer.compute_loss(trainer, model_b, dict(self.batch))
                loss_a.backward()
                loss_b.backward()

                self.assertTrue(torch.allclose(loss_a, loss_b, atol=1e-7))
                for ( _, p_a), (_, p_b) in zip(model_a.named_parameters(), model_b.named_parameters()):
                    self.assertTrue(torch.allclose(p_a.grad, p_b.grad, atol=1e-7))

    def test_model_training_mode_controls_branch_not_grad_enabled_alone(self):
        trainer_on = self.make_trainer(
            finetuning_args=FinetuningArguments(
                use_hard_example_loss=True,
                hard_example_loss_alpha=0.5,
                hard_example_loss_max_weight=1.5,
            )
        )
        trainer_off = self.make_trainer(finetuning_args=FinetuningArguments(use_hard_example_loss=False))

        # 1. model.eval() with torch.enable_grad() MUST use superclass unweighted loss
        model_eval_on = self.clone_model()
        model_eval_off = self.clone_model()
        model_eval_on.eval()
        model_eval_off.eval()

        with torch.enable_grad():
            loss_eval_on = trainer_on.compute_loss(model_eval_on, dict(self.batch))
            loss_eval_off = trainer_off.compute_loss(model_eval_off, dict(self.batch))
            loss_eval_on.backward()
            loss_eval_off.backward()

        self.assertTrue(torch.allclose(loss_eval_on, loss_eval_off, atol=1e-7))
        self.assertTrue(model_eval_on.forward_calls[-1]["has_labels"])
        for (_, p_on), (_, p_off) in zip(model_eval_on.named_parameters(), model_eval_off.named_parameters()):
            self.assertTrue(torch.allclose(p_on.grad, p_off.grad, atol=1e-7))

        # 2. model.train() with torch.no_grad() MUST still take the hard-example weighted branch
        model_train_on = self.clone_model()
        model_train_on.train()
        with torch.no_grad():
            loss_train_no_grad = trainer_on.compute_loss(model_train_on, dict(self.batch))
        self.assertFalse(model_train_on.forward_calls[-1]["has_labels"])
        self.assertFalse(torch.allclose(loss_train_no_grad, loss_eval_off, atol=1e-4))

    def test_enabled_training_changes_loss_and_gradients_and_skips_duplicate_model_ce(self):
        trainer_base = self.make_trainer(finetuning_args=FinetuningArguments(use_hard_example_loss=False))
        trainer_uniform = self.make_trainer(
            finetuning_args=FinetuningArguments(
                use_hard_example_loss=True,
                hard_example_loss_alpha=0.0,
                hard_example_loss_max_weight=1.0,
            )
        )
        trainer_weighted = self.make_trainer(
            finetuning_args=FinetuningArguments(
                use_hard_example_loss=True,
                hard_example_loss_alpha=0.5,
                hard_example_loss_max_weight=1.5,
            )
        )

        model_base = self.clone_model()
        model_uniform = self.clone_model()
        model_weighted = self.clone_model()
        model_base.train()
        model_uniform.train()
        model_weighted.train()

        batch_copy = dict(self.batch)
        loss_base = trainer_base.compute_loss(model_base, dict(self.batch))
        loss_uniform = trainer_uniform.compute_loss(model_uniform, dict(self.batch))
        loss_weighted, outputs_weighted = trainer_weighted.compute_loss(
            model_weighted, batch_copy, return_outputs=True
        )

        # Caller's input dictionary must still have "labels" intact
        self.assertIn("labels", batch_copy)
        # Weighted training branch must NOT pass labels to model.forward (avoiding duplicate CE graph)
        self.assertTrue(model_base.forward_calls[-1]["has_labels"])
        self.assertFalse(model_uniform.forward_calls[-1]["has_labels"])
        self.assertFalse(model_weighted.forward_calls[-1]["has_labels"])

        # return_outputs contract
        self.assertIs(outputs_weighted["loss"], loss_weighted)
        self.assertIs(outputs_weighted.loss, loss_weighted)
        self.assertIs(outputs_weighted[0], loss_weighted)
        self.assertEqual(outputs_weighted["logits"].shape, (2, 6, self.config.vocab_size))

        loss_base.backward()
        loss_uniform.backward()
        loss_weighted.backward()

        # Uniform weighted loss matches baseline in value and gradients
        self.assertTrue(torch.allclose(loss_uniform, loss_base, atol=1e-6))
        for (_, p_u), (_, p_b) in zip(model_uniform.named_parameters(), model_base.named_parameters()):
            self.assertTrue(torch.allclose(p_u.grad, p_b.grad, atol=1e-6))

        # Active hard-example weighting changes loss and parameter gradients
        self.assertGreater(loss_weighted.item(), loss_base.item())
        grad_changed = any(
            not torch.allclose(p_w.grad, p_b.grad, atol=1e-4)
            for (_, p_w), (_, p_b) in zip(model_weighted.named_parameters(), model_base.named_parameters())
        )
        self.assertTrue(grad_changed)

    def test_gradient_accumulation_and_num_items_in_batch_not_rescaled_twice(self):
        gas = 4
        num_items = torch.tensor(20)
        for processor in (DummyProcessor(), None):
            with self.subTest(has_processor=processor is not None):
                training_args = self.make_training_args(gradient_accumulation_steps=gas)
                finetuning_args = FinetuningArguments(
                    use_hard_example_loss=True,
                    hard_example_loss_alpha=0.5,
                    hard_example_loss_max_weight=1.5,
                )
                trainer = self.make_trainer(
                    finetuning_args=finetuning_args,
                    training_args=training_args,
                    processor=processor,
                )
                self.assertFalse(trainer.model_accepts_loss_kwargs)

                model_direct = self.clone_model()
                model_direct.train()
                unscaled_loss = trainer.compute_loss(
                    model_direct, dict(self.batch), num_items_in_batch=num_items
                )
                unscaled_loss.backward()

                trainer.model.zero_grad()
                step_loss = trainer.training_step(
                    trainer.model, dict(self.batch), num_items_in_batch=num_items
                )

                # training_step must divide unscaled_loss by gradient_accumulation_steps ONCE
                self.assertTrue(torch.allclose(step_loss, unscaled_loss.detach() / gas, atol=1e-6))
                for (_, p_step), (_, p_dir) in zip(
                    trainer.model.named_parameters(), model_direct.named_parameters()
                ):
                    self.assertTrue(torch.allclose(p_step.grad, p_dir.grad / gas, atol=1e-6))

    def test_train_evaluate_and_predict_end_to_end_on_cpu(self):
        examples = [
            {k: v[i].clone() for k, v in self.batch.items()}
            for i in range(self.batch["input_ids"].size(0))
        ]
        dataset = TinyDictDataset(examples)
        training_args_on = self.make_training_args(
            output_dir=str(Path(self.temp.name) / "run_on"),
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            max_steps=1,
        )
        training_args_off = self.make_training_args(
            output_dir=str(Path(self.temp.name) / "run_off"),
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            max_steps=1,
        )

        trainer_on = self.make_trainer(
            finetuning_args=FinetuningArguments(
                use_hard_example_loss=True,
                hard_example_loss_alpha=0.5,
                hard_example_loss_max_weight=1.5,
            ),
            training_args=training_args_on,
            train_dataset=dataset,
            eval_dataset=dataset,
        )
        trainer_off = self.make_trainer(
            finetuning_args=FinetuningArguments(use_hard_example_loss=False),
            training_args=training_args_off,
            train_dataset=dataset,
            eval_dataset=dataset,
        )

        # Before training, evaluate() and predict() on identical weights must match exactly
        eval_on = trainer_on.evaluate()
        eval_off = trainer_off.evaluate()
        self.assertAlmostEqual(eval_on["eval_loss"], eval_off["eval_loss"], places=6)

        pred_on = trainer_on.predict(dataset)
        pred_off = trainer_off.predict(dataset)
        self.assertAlmostEqual(pred_on.metrics["test_loss"], pred_off.metrics["test_loss"], places=6)

        # Run 1 training step on CPU and verify finite training loss and updated parameters
        initial_head = trainer_on.model.lm_head.weight.detach().clone()
        train_result = trainer_on.train()
        self.assertIn("train_loss", train_result.metrics)
        self.assertTrue(torch.isfinite(torch.tensor(train_result.metrics["train_loss"])).item())
        self.assertFalse(torch.allclose(trainer_on.model.lm_head.weight, initial_head))


if __name__ == "__main__":
    unittest.main()

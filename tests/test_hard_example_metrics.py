"""CPU-only contracts for observational weight metrics, including two ranks."""

import json
import math
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from test_hard_example_loss import compute_hard_example_weighted_loss
from llamafactory.train.sft.hard_example_metrics import HardExampleWeightStats


def distributed_worker(rank, rendezvous, output):
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    try:
        stats = HardExampleWeightStats()
        weights = [1.5] if rank == 0 else [1.0, 1.0, 1.0]
        stats.update(torch.tensor(weights), torch.ones(len(weights)))
        first = stats.take("cpu")
        # Rank zero has no observations; it must still enter both collectives.
        if rank == 1:
            stats.update(torch.tensor([1.4]), torch.tensor([2]))
        second = stats.take("cpu")
        Path(output, f"rank-{rank}.json").write_text(json.dumps([first, second]))
    finally:
        dist.destroy_process_group()


class HardExampleWeightStatsTests(unittest.TestCase):
    def test_counts_examples_not_tokens_or_microbatch_means_and_resets(self):
        stats = HardExampleWeightStats()
        stats.update(torch.tensor([1.5]), torch.tensor([1]))
        stats.update(torch.tensor([1.0, 1.0, 1.0, 99.0]), torch.tensor([100, 1, 9, 0]))
        result = stats.take("cpu")
        self.assertEqual(result["weight_samples"], 4)
        self.assertEqual(result["mean_weight"], 1.125)
        self.assertEqual(result["pct_weight_gt_1.1"], 25)
        self.assertEqual(result["pct_weight_gt_1.3"], 25)
        self.assertEqual(result["max_weight"], 1.5)
        self.assertEqual(stats.take("cpu"), {})

    def test_thresholds_are_strict_and_empty_answers_excluded(self):
        stats = HardExampleWeightStats()
        stats.update(torch.tensor([1.0, 1.1, 1.3, 1.4, 1.5]), torch.tensor([1, 1, 1, 1, 0]))
        result = stats.take("cpu")
        self.assertEqual(result["pct_weight_gt_1.1"], 50)
        self.assertEqual(result["pct_weight_gt_1.3"], 25)
        self.assertAlmostEqual(result["max_weight"], 1.4)

    def test_all_masked_does_not_create_fake_measurements(self):
        stats = HardExampleWeightStats()
        stats.update(torch.ones(4), torch.zeros(4))
        self.assertEqual(stats.take("cpu"), {})

    def test_details_and_metrics_preserve_loss_and_gradients(self):
        torch.manual_seed(42)
        a = torch.randn(4, 7, 13, requires_grad=True)
        b = a.detach().clone().requires_grad_(True)
        labels = torch.randint(0, 13, (4, 7))
        labels[0, :] = -100
        labels[1, :3] = -100
        plain = compute_hard_example_weighted_loss(a, labels)
        observed, details = compute_hard_example_weighted_loss(b, labels, return_details=True)
        stats = HardExampleWeightStats()
        stats.update(details.weights, details.valid_token_counts)
        self.assertFalse(stats._totals.requires_grad)
        self.assertEqual(stats.take("cpu")["weight_samples"], 3)
        plain.backward()
        observed.backward()
        self.assertTrue(torch.equal(plain, observed))
        self.assertTrue(torch.equal(a.grad, b.grad))

    def test_nonfinite_weights_are_not_reported_as_finite_percentages(self):
        stats = HardExampleWeightStats()
        stats.update(torch.tensor([1.0, float("nan")]), torch.ones(2))
        result = stats.take("cpu")
        for key in ("mean_weight", "max_weight", "pct_weight_gt_1.1", "pct_weight_gt_1.3"):
            self.assertTrue(math.isnan(result[key]))

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo unavailable")
    def test_two_ranks_unequal_counts_and_empty_rank(self):
        with tempfile.TemporaryDirectory() as temp:
            mp.spawn(distributed_worker, args=(str(Path(temp, "rendezvous")), temp), nprocs=2, join=True)
            outputs = [json.loads(Path(temp, f"rank-{rank}.json").read_text()) for rank in range(2)]
        self.assertEqual(outputs[0], outputs[1])
        first, second = outputs[0]
        self.assertEqual(first["mean_weight"], 1.125)
        self.assertEqual(first["pct_weight_gt_1.1"], 25)
        self.assertEqual(first["max_weight"], 1.5)
        self.assertEqual(second["weight_samples"], 1)
        self.assertAlmostEqual(second["mean_weight"], 1.4)
        self.assertEqual(second["pct_weight_gt_1.3"], 100)


if __name__ == "__main__":
    unittest.main()

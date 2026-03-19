"""Unit tests for the RLCF self-contained DPO training script."""

import torch

from tinker_cookbook.recipes.rlcf.train import Config, compute_dpo_loss, get_batches


class TestConfig:
    """Verify paper-faithful defaults match train_rlcf.sh."""

    def test_model(self) -> None:
        assert Config().model_name == "Qwen/Qwen2.5-7B-Instruct"

    def test_dpo_beta(self) -> None:
        assert Config().dpo_beta == 0.1

    def test_learning_rate(self) -> None:
        assert Config().learning_rate == 3e-6

    def test_num_epochs(self) -> None:
        assert Config().num_epochs == 2

    def test_max_length(self) -> None:
        assert Config().max_length == 2048

    def test_dataset(self) -> None:
        assert Config().dataset_name == "viswavi/rlcf"

    def test_lr_schedule(self) -> None:
        assert Config().lr_schedule == "linear"

    def test_save_every(self) -> None:
        assert Config().save_every == 32


class TestComputeDpoLoss:
    def test_chosen_preferred(self) -> None:
        """When chosen has higher logprobs, loss should be low and accuracy high."""
        chosen_lp = [torch.tensor(0.0)]
        rejected_lp = [torch.tensor(-2.0)]
        chosen_ref_lp = [torch.tensor(-1.0)]
        rejected_ref_lp = [torch.tensor(-1.0)]

        loss, metrics = compute_dpo_loss(
            chosen_lp, rejected_lp, chosen_ref_lp, rejected_ref_lp, dpo_beta=0.1
        )
        assert metrics["dpo/accuracy"] == 1.0
        assert metrics["dpo/margin"] > 0

    def test_rejected_preferred(self) -> None:
        """When rejected has higher logprobs, accuracy should be 0."""
        chosen_lp = [torch.tensor(-2.0)]
        rejected_lp = [torch.tensor(0.0)]
        chosen_ref_lp = [torch.tensor(-1.0)]
        rejected_ref_lp = [torch.tensor(-1.0)]

        loss, metrics = compute_dpo_loss(
            chosen_lp, rejected_lp, chosen_ref_lp, rejected_ref_lp, dpo_beta=0.1
        )
        assert metrics["dpo/accuracy"] == 0.0
        assert metrics["dpo/margin"] < 0

    def test_equal_logprobs(self) -> None:
        """When logprobs are equal, loss should be log(2)."""
        lp = [torch.tensor(0.0)]
        ref = [torch.tensor(0.0)]

        loss, metrics = compute_dpo_loss(lp, lp, ref, ref, dpo_beta=0.1)
        assert abs(loss.item() - torch.log(torch.tensor(2.0)).item()) < 1e-5

    def test_beta_scaling(self) -> None:
        """Higher beta should amplify the margin."""
        chosen_lp = [torch.tensor(0.0)]
        rejected_lp = [torch.tensor(-1.0)]
        ref = [torch.tensor(-0.5)]

        _, m_low = compute_dpo_loss(chosen_lp, rejected_lp, ref, ref, dpo_beta=0.1)
        _, m_high = compute_dpo_loss(chosen_lp, rejected_lp, ref, ref, dpo_beta=1.0)
        assert abs(m_high["dpo/margin"]) > abs(m_low["dpo/margin"])


class TestGetBatches:
    def _make_fake_datums(self, n_pairs: int) -> list:
        """Create a flat list of 2*n_pairs placeholder objects."""
        return list(range(2 * n_pairs))

    def test_correct_number_of_batches(self) -> None:
        datums = self._make_fake_datums(10)
        batches = get_batches(datums, batch_size=3, epoch_seed=0)
        assert len(batches) == 4  # ceil(10/3) = 4, last batch has 1 pair

    def test_pairs_stay_together(self) -> None:
        datums = self._make_fake_datums(5)
        batches = get_batches(datums, batch_size=2, epoch_seed=42)
        for batch in batches:
            for i in range(0, len(batch), 2):
                chosen = batch[i]
                rejected = batch[i + 1]
                assert chosen % 2 == 0
                assert rejected == chosen + 1

    def test_different_seeds_shuffle_differently(self) -> None:
        datums = self._make_fake_datums(20)
        b0 = get_batches(datums, batch_size=20, epoch_seed=0)
        b1 = get_batches(datums, batch_size=20, epoch_seed=1)
        assert b0[0] != b1[0]

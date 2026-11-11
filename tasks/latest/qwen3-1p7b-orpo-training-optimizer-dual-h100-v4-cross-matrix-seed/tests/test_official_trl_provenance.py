from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from conftest import LAB_ROOT, TASK_ROOT
from runtime.api import ObjectiveContract, build_official_trl_objective
from runtime import hf_reference


EXPECTED_SOURCE_SHA256 = (
    "ac04e1e4b530af2f8fc64e5d47202b4a123a74bf08f1ccc0b60ba250a670c0fe"
)
EXPECTED_WHEEL_SHA256 = (
    "502b958ac95313bd3e1a31ce91bc852b14f90303bc9b973ede1ee959f40c5c5a"
)


def test_runtime_binds_exact_official_trl_source():
    provenance = hf_reference.official_trl_provenance()
    assert provenance["distribution"] == "trl"
    assert provenance["version"] == "0.12.1"
    assert provenance["tag"] == "v0.12.1"
    assert provenance["commit"] == "c1e9ea8ecfdaf1486e2fc84bc55379b2338d5320"
    assert provenance["module"] == "trl.trainer.orpo_trainer"
    assert provenance["source_sha256"] == EXPECTED_SOURCE_SHA256
    assert provenance["source_path"].endswith("/trl/trainer/orpo_trainer.py")

    from trl.trainer.orpo_trainer import ORPOTrainer

    assert ORPOTrainer.get_batch_logps.__module__ == "trl.trainer.orpo_trainer"
    assert ORPOTrainer.odds_ratio_loss.__module__ == "trl.trainer.orpo_trainer"
    assert inspect.getsourcefile(ORPOTrainer.get_batch_logps) == provenance["source_path"]
    assert inspect.getsourcefile(ORPOTrainer.odds_ratio_loss) == provenance["source_path"]


def test_aligned_adapter_branch_equals_official_decoder_only_shift():
    from trl.trainer.orpo_trainer import ORPOTrainer

    generator = torch.Generator().manual_seed(20260815)
    aligned_logits = torch.randn(4, 5, 13, generator=generator)
    aligned_labels = torch.randint(13, (4, 5), generator=generator)
    aligned_labels[:, :2] = -100
    # Decoder-only TRL discards the final logit and first label.  Append/prepend
    # those unused positions to reconstruct its normal unaligned call shape.
    full_logits = torch.cat((aligned_logits, torch.randn(4, 1, 13, generator=generator)), dim=1)
    full_labels = torch.cat((torch.full((4, 1), -100), aligned_labels), dim=1)

    decoder_only = ORPOTrainer.get_batch_logps(
        full_logits,
        full_labels,
        average_log_prob=True,
        label_pad_token_id=-100,
        is_encoder_decoder=False,
    )
    already_aligned = ORPOTrainer.get_batch_logps(
        aligned_logits,
        aligned_labels,
        average_log_prob=True,
        label_pad_token_id=-100,
        is_encoder_decoder=True,
    )
    torch.testing.assert_close(already_aligned, decoder_only)


def test_tensor_adapter_matches_official_concatenated_forward_loss_and_gradients():
    """Qualify the adapter against TRL's real decoder-only forward path.

    The benchmark ABI receives already shifted hidden states.  This test starts
    one level earlier with the official ``concatenated_forward`` method, then
    checks that the adapter preserves every loss-bearing value and gradient
    after the trusted next-token alignment.
    """

    from trl.trainer.orpo_trainer import ORPOTrainer

    torch.manual_seed(7)
    pair_batch, sequence_length, hidden_size, vocab_size = 2, 6, 5, 13
    chosen_ids = torch.tensor(
        [[1, 2, 3, 4, 5, 0], [2, 3, 4, 5, 6, 7]], dtype=torch.long
    )
    rejected_ids = torch.tensor(
        [[1, 2, 8, 9, 10, 0], [2, 3, 9, 10, 11, 12]], dtype=torch.long
    )
    chosen_mask = chosen_ids.ne(0).long()
    rejected_mask = rejected_ids.ne(0).long()
    chosen_labels = chosen_ids.clone()
    rejected_labels = rejected_ids.clone()
    chosen_labels[:, :2] = -100
    rejected_labels[:, :2] = -100
    chosen_labels.masked_fill_(chosen_mask.eq(0), -100)
    rejected_labels.masked_fill_(rejected_mask.eq(0), -100)
    batch = {
        "chosen_input_ids": chosen_ids,
        "chosen_attention_mask": chosen_mask,
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_ids,
        "rejected_attention_mask": rejected_mask,
        "rejected_labels": rejected_labels,
    }

    raw_hidden = torch.randn(2 * pair_batch, sequence_length, hidden_size)
    raw_weight = torch.randn(vocab_size, hidden_size)

    class _FixedHiddenModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = torch.nn.Parameter(raw_hidden.clone())
            self.weight = torch.nn.Parameter(raw_weight.clone())

        def forward(self, input_ids, **_):
            assert input_ids.shape == (2 * pair_batch, sequence_length)
            return SimpleNamespace(logits=self.hidden @ self.weight.t())

    official_model = _FixedHiddenModel()
    official = object.__new__(ORPOTrainer)
    official.is_encoder_decoder = False
    official.label_pad_token_id = -100
    official.padding_value = 0
    official.accelerator = SimpleNamespace(device=torch.device("cpu"))
    official.aux_loss_enabled = False
    official.beta = 0.1

    forward = official.concatenated_forward(official_model, batch)
    losses, chosen_rewards, rejected_rewards, ratio_mean, log_odds_mean = (
        official.odds_ratio_loss(forward[0], forward[1])
    )
    official_loss = forward[4] - losses.mean()
    official_loss.backward()

    concatenated_ids = torch.cat((chosen_ids, rejected_ids), dim=0)
    concatenated_mask = torch.cat((chosen_mask, rejected_mask), dim=0)
    concatenated_labels = torch.cat((chosen_labels, rejected_labels), dim=0)
    aligned_hidden = raw_hidden[:, :-1].clone().requires_grad_(True)
    aligned_weight = raw_weight.clone().requires_grad_(True)
    or_labels = concatenated_labels[:, 1:]
    nll_target = concatenated_ids[:, 1:].clone()
    nll_target.masked_fill_(concatenated_mask[:, 1:].eq(0), -100)

    adapted = build_official_trl_objective(ObjectiveContract(beta=0.1))(
        aligned_weight,
        aligned_hidden,
        or_labels,
        None,
        nll_target,
    )
    adapted.loss.backward()

    torch.testing.assert_close(adapted.loss, official_loss)
    expected_aux = (
        forward[0],
        forward[1],
        forward[2][:, :-1].detach().mean(),
        forward[3][:, :-1].detach().mean(),
        forward[4],
        chosen_rewards,
        rejected_rewards,
        ratio_mean,
        log_odds_mean,
    )
    for actual, expected in zip(adapted.aux, expected_aux, strict=True):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(aligned_weight.grad, official_model.weight.grad)
    torch.testing.assert_close(
        aligned_hidden.grad,
        official_model.hidden.grad[:, :-1],
    )
    torch.testing.assert_close(
        official_model.hidden.grad[:, -1],
        torch.zeros_like(official_model.hidden.grad[:, -1]),
    )


def test_reference_construction_fails_closed_on_source_hash_mismatch(monkeypatch):
    hf_reference._clear_official_trl_binding_cache_for_tests()
    monkeypatch.setattr(hf_reference, "_sha256_file", lambda _: "0" * 64)
    with pytest.raises(RuntimeError, match="source hash mismatch"):
        build_official_trl_objective(ObjectiveContract())
    hf_reference._clear_official_trl_binding_cache_for_tests()


def test_work_image_pins_and_checks_the_published_official_stack():
    dockerfile = (TASK_ROOT / "images/work/Dockerfile").read_text(encoding="utf-8")
    for pin in (
        "trl==0.12.1",
        "transformers==4.51.3",
        "accelerate==1.6.0",
        "datasets==2.21.0",
        "tokenizers==0.21.1",
        "huggingface-hub==0.30.2",
        EXPECTED_WHEEL_SHA256,
        EXPECTED_SOURCE_SHA256,
    ):
        assert pin in dockerfile


def test_v001_is_only_a_factory_and_contains_no_loss_implementation():
    source = (
        LAB_ROOT / "solutions/objective_v001/candidate.py"
    ).read_text(encoding="utf-8")
    assert "build_official_trl_objective" in source
    for handwritten_math in (
        "log_softmax",
        "logsigmoid",
        "log_odds",
        "cross_entropy",
        "torch.exp",
        "torch.log1p",
        "hidden_states @",
    ):
        assert handwritten_math not in source

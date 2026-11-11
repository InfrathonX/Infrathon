from __future__ import annotations

import torch

from conftest import LAB_ROOT
from runtime.api import AUX_NAMES, ObjectiveContract, normalize_result
from runtime.hf_reference import reference_orpo
from runtime.loader import load_objective
from runtime.train_step import TrainStepModule, derive_aligned_orpo_targets


def test_official_contract_fixes_ignore_index_at_minus_100():
    assert ObjectiveContract(beta=0.13).ignore_index == -100
    try:
        ObjectiveContract(beta=0.13, ignore_index=-17)
    except ValueError as exc:
        assert "ignore_index=-100" in str(exc)
    else:  # pragma: no cover - an explicit assertion gives a clearer failure
        raise AssertionError("non-official ignore index was accepted")


def test_seed_loss_aux_and_gradients_match_official_trl_reference():
    torch.manual_seed(19)
    contract = ObjectiveContract(beta=0.13)
    loaded = load_objective(LAB_ROOT, contract)
    raw_weight = torch.randn(53, 11)
    raw_hidden = torch.randn(6, 7, 11)
    labels = torch.randint(53, (6, 7))
    labels[:, :2] = -100
    labels[:, -1] = -100
    nll_target = torch.randint(53, (6, 7))
    nll_target[:, -1] = -100
    bias = torch.randn(53)

    def execute(fn):
        weight = raw_weight.clone().requires_grad_(True)
        hidden = raw_hidden.clone().requires_grad_(True)
        local_bias = bias.clone().requires_grad_(True)
        result = normalize_result(fn(weight, hidden, labels, local_bias, nll_target))
        result.loss.backward()
        return result, weight.grad, hidden.grad, local_bias.grad

    expected = execute(
        lambda w, h, y, b, n: reference_orpo(w, h, y, b, n, contract=contract)
    )
    actual = execute(loaded.module)
    assert len(actual[0].aux) == len(AUX_NAMES) == 9
    torch.testing.assert_close(actual[0].loss, expected[0].loss)
    for got, want in zip(actual[0].aux, expected[0].aux, strict=True):
        torch.testing.assert_close(got, want)
    for got, want in zip(actual[1:], expected[1:], strict=True):
        torch.testing.assert_close(got, want)


def test_real_target_derivation_keeps_prompt_only_in_nll_and_masks_padding():
    input_ids = torch.tensor(
        [
            [10, 11, 12, 20, 21, 0],
            [10, 11, 12, 30, 31, 0],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 0],
        ]
    )
    completion_labels = input_ids.clone()
    completion_labels[:, :3] = -100
    completion_labels.masked_fill_(attention_mask.eq(0), -100)

    or_labels, nll_target = derive_aligned_orpo_targets(
        input_ids,
        attention_mask,
        completion_labels,
    )

    assert or_labels.tolist() == [
        [-100, -100, 20, 21, -100],
        [-100, -100, 30, 31, -100],
    ]
    assert nll_target.tolist() == [
        [11, 12, 20, 21, -100],
        [11, 12, 30, 31, -100],
    ]


def test_real_train_step_passes_the_separate_nll_target():
    class _Backbone(torch.nn.Module):
        def forward(self, input_ids, **_):
            hidden = torch.nn.functional.one_hot(input_ids, num_classes=40).float()
            return type("Output", (), {"last_hidden_state": hidden})()

    class _TinyCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Backbone()
            self.lm_head = torch.nn.Linear(40, 40, bias=False)

    class _CaptureObjective(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_labels = None
            self.seen_nll_target = None

        def forward(self, weight, hidden, labels, bias, nll_target):
            self.seen_labels = labels.detach().clone()
            self.seen_nll_target = nll_target.detach().clone()
            zero = (hidden.sum() + weight.sum()) * 0.0
            return zero, (zero,) * len(AUX_NAMES)

    input_ids = torch.tensor([[10, 11, 20, 21, 0], [10, 11, 30, 31, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0]])
    labels = input_ids.clone()
    labels[:, :2] = -100
    labels.masked_fill_(attention_mask.eq(0), -100)
    capture = _CaptureObjective()

    TrainStepModule(_TinyCausalLM(), capture)(input_ids, attention_mask, labels)

    assert capture.seen_labels is not None
    assert capture.seen_nll_target is not None
    assert capture.seen_labels.tolist() == [
        [-100, 20, 21, -100],
        [-100, 30, 31, -100],
    ]
    assert capture.seen_nll_target.tolist() == [
        [11, 20, 21, -100],
        [11, 30, 31, -100],
    ]


def test_prompt_targets_affect_chosen_nll_but_not_or_logps():
    contract = ObjectiveContract(beta=0.1)
    objective = load_objective(LAB_ROOT, contract).module
    vocab = 7
    weight = torch.eye(vocab)
    hidden = torch.zeros(2, 4, vocab)
    for position, token in enumerate((1, 2, 3, 4)):
        hidden[0, position, token] = 8.0
        hidden[1, position, token] = 4.0
    labels = torch.tensor(
        [
            [-100, -100, 3, 4],
            [-100, -100, 3, 4],
        ]
    )
    original_nll = torch.tensor(
        [
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ]
    )
    changed_prompt_nll = original_nll.clone()
    changed_prompt_nll[0, 0] = 6

    original = objective(weight, hidden, labels, None, original_nll)
    changed = objective(weight, hidden, labels, None, changed_prompt_nll)

    torch.testing.assert_close(original.aux[0], changed.aux[0])
    torch.testing.assert_close(original.aux[1], changed.aux[1])
    assert not torch.isclose(original.aux[4], changed.aux[4])
    assert not torch.isclose(original.loss, changed.loss)

    rejected_only_nll = original_nll.clone()
    rejected_only_nll[1] = torch.tensor([6, 6, 6, 6])
    rejected_changed = objective(weight, hidden, labels, None, rejected_only_nll)
    torch.testing.assert_close(original.aux[4], rejected_changed.aux[4])
    torch.testing.assert_close(original.loss, rejected_changed.loss)

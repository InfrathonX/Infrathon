from __future__ import annotations

import pytest
import torch

from conftest import LAB_ROOT
from runtime.api import AUX_NAMES, ObjectiveContract, normalize_result
from runtime.hf_reference import reference_gkd
from runtime.loader import load_objective
from runtime.train_step import TrainStepModule, derive_aligned_gkd_labels


def test_official_contract_fixes_ignore_index_and_bounds_hyperparameters():
    assert ObjectiveContract(beta=0.13).ignore_index == -100
    try:
        ObjectiveContract(beta=0.13, ignore_index=-17)
    except ValueError as exc:
        assert "ignore_index=-100" in str(exc)
    else:  # pragma: no cover - an explicit assertion gives a clearer failure
        raise AssertionError("non-official ignore index was accepted")
    # beta is inclusive on both ends; temperature must be strictly positive.
    assert ObjectiveContract(beta=0.0).beta == 0.0
    assert ObjectiveContract(beta=1.0).beta == 1.0
    with pytest.raises(ValueError):
        ObjectiveContract(beta=1.5)
    with pytest.raises(ValueError):
        ObjectiveContract(beta=-0.01)
    with pytest.raises(ValueError):
        ObjectiveContract(temperature=0.0)


def test_seed_loss_aux_and_gradients_match_official_trl_reference():
    torch.manual_seed(19)
    contract = ObjectiveContract(beta=0.13, temperature=1.7)
    loaded = load_objective(LAB_ROOT, contract)
    raw_student_weight = torch.randn(53, 11)
    raw_student_hidden = torch.randn(6, 7, 11)
    raw_teacher_weight = torch.randn(53, 13)
    raw_teacher_hidden = torch.randn(6, 7, 13)
    labels = torch.randint(53, (6, 7))
    labels[:, :2] = -100
    labels[:, -1] = -100
    raw_student_bias = torch.randn(53)
    raw_teacher_bias = torch.randn(53)

    def execute(fn):
        student_weight = raw_student_weight.clone().requires_grad_(True)
        student_hidden = raw_student_hidden.clone().requires_grad_(True)
        student_bias = raw_student_bias.clone().requires_grad_(True)
        result = normalize_result(
            fn(
                student_weight,
                student_hidden,
                raw_teacher_weight.clone(),
                raw_teacher_hidden.clone(),
                labels,
                student_bias,
                raw_teacher_bias.clone(),
            )
        )
        result.loss.backward()
        return result, student_weight.grad, student_hidden.grad, student_bias.grad

    expected = execute(
        lambda sw, sh, tw, th, y, sb, tb: reference_gkd(
            sw, sh, tw, th, y, sb, tb, contract=contract
        )
    )
    actual = execute(loaded.module)
    assert len(actual[0].aux) == len(AUX_NAMES) == 9
    torch.testing.assert_close(actual[0].loss, expected[0].loss)
    for got, want in zip(actual[0].aux, expected[0].aux, strict=True):
        torch.testing.assert_close(got, want)
    for got, want in zip(actual[1:], expected[1:], strict=True):
        torch.testing.assert_close(got, want)


def test_real_label_derivation_shifts_and_masks_prompt_and_padding():
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

    gkd_labels = derive_aligned_gkd_labels(
        input_ids,
        attention_mask,
        completion_labels,
    )

    assert gkd_labels.tolist() == [
        [-100, -100, 20, 21, -100],
        [-100, -100, 30, 31, -100],
    ]


def test_real_train_step_passes_shifted_labels_and_frozen_teacher_tensors():
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
            self.seen_teacher_weight = None
            self.seen_teacher_hidden = None
            self.seen_teacher_bias = "unset"

        def forward(
            self,
            student_weight,
            student_hidden,
            teacher_weight,
            teacher_hidden,
            labels,
            student_bias=None,
            teacher_bias=None,
        ):
            self.seen_labels = labels.detach().clone()
            self.seen_teacher_weight = teacher_weight
            self.seen_teacher_hidden = teacher_hidden
            self.seen_teacher_bias = teacher_bias
            zero = (student_hidden.sum() + student_weight.sum()) * 0.0
            return zero, (zero,) * len(AUX_NAMES)

    input_ids = torch.tensor([[10, 11, 20, 21, 0], [10, 11, 30, 31, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0]])
    labels = input_ids.clone()
    labels[:, :2] = -100
    labels.masked_fill_(attention_mask.eq(0), -100)
    teacher_weight = torch.randn(40, 8)
    teacher_hidden = torch.randn(2, 4, 8)
    capture = _CaptureObjective()
    step = TrainStepModule(_TinyCausalLM(), capture, teacher_weight, None)

    step(input_ids, attention_mask, labels, teacher_hidden)

    assert capture.seen_labels is not None
    assert capture.seen_labels.tolist() == [
        [-100, 20, 21, -100],
        [-100, 30, 31, -100],
    ]
    assert capture.seen_teacher_weight is teacher_weight
    assert capture.seen_teacher_hidden is teacher_hidden
    assert capture.seen_teacher_bias is None

    # The teacher boundary is frozen on both construction and forward.
    with pytest.raises(ValueError, match="not require gradients"):
        TrainStepModule(
            _TinyCausalLM(), capture, teacher_weight.clone().requires_grad_(True), None
        )
    with pytest.raises(ValueError, match="not require gradients"):
        step(input_ids, attention_mask, labels, teacher_hidden.clone().requires_grad_(True))
    with pytest.raises(ValueError, match="already-shifted"):
        step(input_ids, attention_mask, labels, torch.randn(2, 5, 8))


def test_masked_positions_are_inert_and_temperature_only_rescales_distributions():
    torch.manual_seed(23)
    contract = ObjectiveContract()
    objective = load_objective(LAB_ROOT, contract).module
    student_weight = torch.randn(19, 5)
    student_hidden = torch.randn(2, 4, 5)
    teacher_weight = torch.randn(19, 6)
    teacher_hidden = torch.randn(2, 4, 6)
    labels = torch.tensor(
        [
            [-100, -100, 3, 4],
            [-100, -100, 5, 6],
        ]
    )

    base = objective(student_weight, student_hidden, teacher_weight, teacher_hidden, labels)

    # Changing hidden states only at ignore-index positions changes nothing:
    # official GKD scores completion positions alone.
    prompt_perturbed = student_hidden.clone()
    prompt_perturbed[:, :2, :] += 3.0
    perturbed = objective(
        student_weight, prompt_perturbed, teacher_weight, teacher_hidden, labels
    )
    torch.testing.assert_close(perturbed.loss, base.loss)
    for got, want in zip(perturbed.aux, base.aux, strict=True):
        torch.testing.assert_close(got, want)

    # A scored-position change must reach the loss.
    scored_perturbed = student_hidden.clone()
    scored_perturbed[:, 2:, :] += 3.0
    scored = objective(
        student_weight, scored_perturbed, teacher_weight, teacher_hidden, labels
    )
    assert not torch.isclose(scored.loss, base.loss)

    # Temperature rescales the loss and JSD witnesses but never the raw
    # (unscaled) logits means the telemetry pins.
    hot = load_objective(LAB_ROOT, ObjectiveContract(temperature=2.3)).module
    hot_result = hot(student_weight, student_hidden, teacher_weight, teacher_hidden, labels)
    assert not torch.isclose(hot_result.loss, base.loss)
    torch.testing.assert_close(hot_result.aux[2], base.aux[2])
    torch.testing.assert_close(hot_result.aux[3], base.aux[3])
    assert not torch.allclose(hot_result.aux[4], base.aux[4])


def test_teacher_tensors_requiring_gradients_are_rejected():
    torch.manual_seed(29)
    objective = load_objective(LAB_ROOT, ObjectiveContract()).module
    student_weight = torch.randn(19, 5)
    student_hidden = torch.randn(2, 4, 5)
    teacher_weight = torch.randn(19, 6)
    teacher_hidden = torch.randn(2, 4, 6)
    teacher_bias = torch.randn(19)
    labels = torch.tensor([[-100, 3, 4, -100], [-100, 5, 6, -100]])

    with pytest.raises(ValueError, match="must not require gradients"):
        objective(
            student_weight,
            student_hidden,
            teacher_weight.clone().requires_grad_(True),
            teacher_hidden,
            labels,
        )
    with pytest.raises(ValueError, match="must not require gradients"):
        objective(
            student_weight,
            student_hidden,
            teacher_weight,
            teacher_hidden.clone().requires_grad_(True),
            labels,
        )
    with pytest.raises(ValueError, match="must not require gradients"):
        objective(
            student_weight,
            student_hidden,
            teacher_weight,
            teacher_hidden,
            labels,
            None,
            teacher_bias.clone().requires_grad_(True),
        )

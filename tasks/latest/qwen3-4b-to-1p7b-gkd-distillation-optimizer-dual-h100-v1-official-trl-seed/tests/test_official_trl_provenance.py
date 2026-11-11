from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from conftest import LAB_ROOT, TASK_ROOT
from runtime.api import ObjectiveContract, build_official_trl_objective
from runtime import hf_reference


EXPECTED_SOURCE_SHA256 = (
    "09e6e8932297ca3793aa5c429852bb46e3f395f9f3aaa48479d3b7965a028f1a"
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
    assert provenance["module"] == "trl.trainer.gkd_trainer"
    assert provenance["source_sha256"] == EXPECTED_SOURCE_SHA256
    assert provenance["source_path"].endswith("/trl/trainer/gkd_trainer.py")

    from trl.trainer.gkd_trainer import GKDTrainer

    method = inspect.getattr_static(GKDTrainer, "generalized_jsd_loss")
    assert isinstance(method, staticmethod)
    assert GKDTrainer.generalized_jsd_loss.__module__ == "trl.trainer.gkd_trainer"
    assert (
        inspect.getsourcefile(GKDTrainer.generalized_jsd_loss)
        == provenance["source_path"]
    )


def test_aligned_adapter_call_equals_official_prompt_sliced_shift():
    """The ABI's already-shifted tensors reproduce TRL's compute_loss slicing.

    ``GKDTrainer.compute_loss`` slices full logits ``[:, p-1:-1]`` against
    labels ``[:, p:]``.  The MLSys Forge driver instead hands the objective the
    full next-token alignment ``[:, :-1]`` / ``[:, 1:]`` with prompt positions
    ignore-masked; the official mask must make both calls identical.
    """

    from trl.trainer.gkd_trainer import GKDTrainer

    generator = torch.Generator().manual_seed(20260822)
    batch, sequence, vocab, prompt = 3, 9, 13, 4
    student_full = torch.randn(batch, sequence, vocab, generator=generator)
    teacher_full = torch.randn(batch, sequence, vocab, generator=generator)
    full_labels = torch.randint(vocab, (batch, sequence), generator=generator)
    full_labels[:, :prompt] = -100
    full_labels[:, -1] = -100  # trailing padding stays unscored

    official = GKDTrainer.generalized_jsd_loss(
        student_full[:, prompt - 1 : -1],
        teacher_full[:, prompt - 1 : -1],
        labels=full_labels[:, prompt:],
        beta=0.3,
        temperature=1.3,
    )
    already_aligned = GKDTrainer.generalized_jsd_loss(
        student_full[:, :-1],
        teacher_full[:, :-1],
        labels=full_labels[:, 1:],
        beta=0.3,
        temperature=1.3,
    )
    torch.testing.assert_close(already_aligned, official)


def test_tensor_adapter_matches_official_compute_loss_and_gradients():
    """Qualify the adapter against TRL's real student/teacher forward path.

    The benchmark ABI receives already shifted hidden states.  This test starts
    one level earlier with the official ``compute_loss`` method over fixed
    student and teacher models, then checks that the adapter preserves the
    loss, every student gradient, and the internal witness identities after
    the trusted next-token alignment.
    """

    from trl.trainer.gkd_trainer import GKDTrainer

    torch.manual_seed(7)
    batch, sequence_length, prompt_length = 2, 6, 2
    student_hidden_size, teacher_hidden_size, vocab_size = 5, 7, 13
    input_ids = torch.tensor(
        [[1, 2, 3, 4, 5, 0], [2, 3, 4, 5, 6, 7]], dtype=torch.long
    )
    attention_mask = input_ids.ne(0).long()
    labels = input_ids.clone()
    labels[:, :prompt_length] = -100
    labels.masked_fill_(attention_mask.eq(0), -100)
    inputs = {
        "prompts": input_ids[:, :prompt_length],
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    raw_student_hidden = torch.randn(batch, sequence_length, student_hidden_size)
    raw_student_weight = torch.randn(vocab_size, student_hidden_size)
    raw_teacher_hidden = torch.randn(batch, sequence_length, teacher_hidden_size)
    raw_teacher_weight = torch.randn(vocab_size, teacher_hidden_size)

    class _FixedHiddenModel(torch.nn.Module):
        def __init__(self, hidden, weight):
            super().__init__()
            self.hidden = torch.nn.Parameter(hidden.clone())
            self.weight = torch.nn.Parameter(weight.clone())

        def forward(self, input_ids, **_):
            assert input_ids.shape == (batch, sequence_length)
            return SimpleNamespace(logits=self.hidden @ self.weight.t())

    student_model = _FixedHiddenModel(raw_student_hidden, raw_student_weight)
    teacher_model = _FixedHiddenModel(raw_teacher_hidden, raw_teacher_weight)
    official = object.__new__(GKDTrainer)
    official.teacher_model = teacher_model
    official.beta = 0.5

    # compute_loss invokes generalized_jsd_loss with its default temperature.
    official_loss = official.compute_loss(student_model, inputs)
    official_loss.backward()

    aligned_student_hidden = raw_student_hidden[:, :-1].clone().requires_grad_(True)
    aligned_student_weight = raw_student_weight.clone().requires_grad_(True)
    aligned_teacher_hidden = raw_teacher_hidden[:, :-1].clone()
    aligned_teacher_weight = raw_teacher_weight.clone()
    aligned_labels = labels[:, 1:]

    adapted = build_official_trl_objective(
        ObjectiveContract(beta=0.5, temperature=1.0)
    )(
        aligned_student_weight,
        aligned_student_hidden,
        aligned_teacher_weight,
        aligned_teacher_hidden,
        aligned_labels,
    )
    adapted.loss.backward()

    torch.testing.assert_close(adapted.loss, official_loss)
    torch.testing.assert_close(aligned_student_weight.grad, student_model.weight.grad)
    torch.testing.assert_close(
        aligned_student_hidden.grad,
        student_model.hidden.grad[:, :-1],
    )
    torch.testing.assert_close(
        student_model.hidden.grad[:, -1],
        torch.zeros_like(student_model.hidden.grad[:, -1]),
    )
    assert teacher_model.weight.grad is None
    assert teacher_model.hidden.grad is None

    # Witness identities over the same temperature-scaled distributions the
    # official method consumes: the KL mixture means recombine into the loss,
    # and the count-weighted per-sequence JSD equals the batchmean loss.
    beta = 0.5
    torch.testing.assert_close(
        beta * adapted.aux[5] + (1.0 - beta) * adapted.aux[6],
        adapted.loss.detach(),
    )
    scored_per_sequence = aligned_labels.ne(-100).sum(dim=-1).to(torch.float32)
    torch.testing.assert_close(
        (adapted.aux[4] * scored_per_sequence).sum() / scored_per_sequence.sum(),
        adapted.loss.detach(),
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
        "kl_div",
        "logsumexp",
        "generalized_jsd",
        "cross_entropy",
        "torch.exp",
        "torch.log",
        "hidden @",
    ):
        assert handwritten_math not in source

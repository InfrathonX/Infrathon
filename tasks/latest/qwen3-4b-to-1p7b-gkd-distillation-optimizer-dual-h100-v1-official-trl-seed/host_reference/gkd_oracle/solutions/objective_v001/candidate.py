"""Host-only positive control: Liger-style chunked GKD with TRL-exact math.

This objective never materializes the full [BT, V] logits for either model.
It streams row chunks, performs both vocabulary projections per chunk,
computes the official TRL generalized JSD (mixture = beta*student +
(1-beta)*teacher; jsd = beta*KL(teacher||M) + (1-beta)*KL(student||M);
temperature scaling; batchmean over scored tokens), and accumulates student
gradients during the forward pass exactly like Liger's fused-linear
distillation base — so backward stores no chunk graphs at all.

The stock ``LigerFusedLinearJSDLoss`` weighting differs from TRL for
beta != 0.5; this control therefore implements the TRL formula directly.
Host-only: this source is never copied into the Work or Judge image.
"""

from __future__ import annotations

import torch

from runtime.api import (
    AUX_NAMES,
    GKDResult,
    ObjectiveContract,
    validate_inputs,
)


CHUNK_ROWS = 1024


def _chunk_terms(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    beta: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token KL components and log-probabilities for one row chunk."""

    student_log_probs = torch.log_softmax(
        student_logits.to(torch.float32) / temperature, dim=-1
    )
    teacher_log_probs = torch.log_softmax(
        teacher_logits.to(torch.float32) / temperature, dim=-1
    )
    beta_tensor = torch.tensor(beta, dtype=student_log_probs.dtype)
    mixture_log_probs = torch.logsumexp(
        torch.stack(
            [
                student_log_probs + torch.log(beta_tensor),
                teacher_log_probs + torch.log(1 - beta_tensor),
            ]
        ),
        dim=0,
    )
    kl_teacher = (
        teacher_log_probs.exp() * (teacher_log_probs - mixture_log_probs)
    ).sum(dim=-1)
    kl_student = (
        student_log_probs.exp() * (student_log_probs - mixture_log_probs)
    ).sum(dim=-1)
    return kl_teacher, kl_student, student_log_probs, teacher_log_probs


class _ChunkedGKDFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        student_weight: torch.Tensor,
        student_hidden: torch.Tensor,
        teacher_weight: torch.Tensor,
        teacher_hidden: torch.Tensor,
        labels: torch.Tensor,
        student_bias: torch.Tensor | None,
        teacher_bias: torch.Tensor | None,
        beta: float,
        temperature: float,
        ignore_index: int,
    ):
        batch, seq_len, student_hidden_size = student_hidden.shape
        rows = batch * seq_len
        flat_student = student_hidden.reshape(rows, student_hidden_size)
        flat_teacher = teacher_hidden.reshape(rows, teacher_hidden.shape[2])
        flat_labels = labels.reshape(rows)
        scored = flat_labels.ne(ignore_index)
        n_scored = scored.sum().clamp_min(1).to(torch.float32)
        gather_index = flat_labels.clamp_min(0).unsqueeze(-1)

        grad_hidden = torch.zeros_like(flat_student)
        grad_weight = (
            torch.zeros_like(student_weight)
            if student_weight.requires_grad
            else None
        )
        grad_bias = (
            torch.zeros_like(student_bias)
            if student_bias is not None and student_bias.requires_grad
            else None
        )

        loss_total = torch.zeros((), dtype=torch.float32, device=flat_student.device)
        kl_teacher_sum = torch.zeros_like(loss_total)
        kl_student_sum = torch.zeros_like(loss_total)
        student_logit_sum = torch.zeros((), dtype=torch.float64, device=flat_student.device)
        teacher_logit_sum = torch.zeros_like(student_logit_sum)
        scored_vocab_count = torch.zeros_like(student_logit_sum)
        student_logp_rows = torch.zeros(rows, dtype=torch.float32, device=flat_student.device)
        teacher_logp_rows = torch.zeros_like(student_logp_rows)
        jsd_rows = torch.zeros_like(student_logp_rows)
        student_entropy_sum = torch.zeros_like(loss_total)
        teacher_entropy_sum = torch.zeros_like(loss_total)

        for start in range(0, rows, CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, rows)
            hidden_chunk = flat_student[start:stop].detach().requires_grad_(True)
            weight_leaf = student_weight.detach().requires_grad_(
                grad_weight is not None
            )
            bias_leaf = (
                None
                if student_bias is None
                else student_bias.detach().requires_grad_(grad_bias is not None)
            )
            chunk_scored = scored[start:stop]
            with torch.enable_grad():
                student_logits = hidden_chunk @ weight_leaf.t()
                if bias_leaf is not None:
                    student_logits = student_logits + bias_leaf
                with torch.no_grad():
                    teacher_logits = flat_teacher[start:stop] @ teacher_weight.t()
                    if teacher_bias is not None:
                        teacher_logits = teacher_logits + teacher_bias
                kl_teacher, kl_student, student_log_probs, teacher_log_probs = (
                    _chunk_terms(student_logits, teacher_logits, beta, temperature)
                )
                jsd_token = beta * kl_teacher + (1.0 - beta) * kl_student
                chunk_loss = (
                    jsd_token * chunk_scored.to(jsd_token.dtype)
                ).sum() / n_scored
            if hidden_chunk.numel() > 0:
                chunk_loss.backward()
            loss_total = loss_total + chunk_loss.detach()
            grad_hidden[start:stop] = (
                hidden_chunk.grad
                if hidden_chunk.grad is not None
                else torch.zeros_like(hidden_chunk)
            )
            if grad_weight is not None and weight_leaf.grad is not None:
                grad_weight.add_(weight_leaf.grad)
            if grad_bias is not None and bias_leaf is not None and bias_leaf.grad is not None:
                grad_bias.add_(bias_leaf.grad)

            with torch.no_grad():
                mask_f32 = chunk_scored.to(torch.float32)
                kl_teacher_sum += (kl_teacher.detach() * mask_f32).sum()
                kl_student_sum += (kl_student.detach() * mask_f32).sum()
                jsd_rows[start:stop] = (
                    beta * kl_teacher.detach() + (1.0 - beta) * kl_student.detach()
                ) * mask_f32
                chunk_index = gather_index[start:stop]
                student_logp_rows[start:stop] = (
                    student_log_probs.detach().gather(-1, chunk_index).squeeze(-1)
                    * mask_f32
                )
                teacher_logp_rows[start:stop] = (
                    teacher_log_probs.detach().gather(-1, chunk_index).squeeze(-1)
                    * mask_f32
                )
                student_entropy_sum += (
                    -(student_log_probs.detach().exp() * student_log_probs.detach()).sum(
                        dim=-1
                    )
                    * mask_f32
                ).sum()
                teacher_entropy_sum += (
                    -(teacher_log_probs.detach().exp() * teacher_log_probs.detach()).sum(
                        dim=-1
                    )
                    * mask_f32
                ).sum()
                student_logit_sum += (
                    student_logits.detach().to(torch.float64)
                    * mask_f32.unsqueeze(-1).to(torch.float64)
                ).sum()
                teacher_logit_sum += (
                    teacher_logits.detach().to(torch.float64)
                    * mask_f32.unsqueeze(-1).to(torch.float64)
                ).sum()
                scored_vocab_count += mask_f32.sum().to(torch.float64) * float(
                    student_weight.shape[0]
                )

        ctx.save_for_backward(
            grad_weight if grad_weight is not None else torch.empty(0),
            grad_hidden,
            grad_bias if grad_bias is not None else torch.empty(0),
        )
        ctx.has_weight_grad = grad_weight is not None
        ctx.has_bias_grad = grad_bias is not None
        ctx.hidden_shape = student_hidden.shape

        counts = scored.reshape(batch, seq_len).sum(dim=-1).clamp_min(1).to(torch.float32)
        student_completion_logps = (
            student_logp_rows.reshape(batch, seq_len).sum(dim=-1) / counts
        )
        teacher_completion_logps = (
            teacher_logp_rows.reshape(batch, seq_len).sum(dim=-1) / counts
        )
        jsd_per_sequence = jsd_rows.reshape(batch, seq_len).sum(dim=-1) / counts
        source_dtype = student_hidden.dtype
        student_logits_mean = (
            (student_logit_sum / scored_vocab_count).to(source_dtype)
        )
        teacher_logits_mean = (
            (teacher_logit_sum / scored_vocab_count).to(source_dtype)
        )
        return (
            loss_total,
            student_completion_logps,
            teacher_completion_logps,
            student_logits_mean,
            teacher_logits_mean,
            jsd_per_sequence,
            kl_teacher_sum / n_scored,
            kl_student_sum / n_scored,
            student_entropy_sum / n_scored,
            teacher_entropy_sum / n_scored,
        )

    @staticmethod
    def backward(ctx, grad_loss, *unused_aux_grads):
        grad_weight, grad_hidden, grad_bias = ctx.saved_tensors
        scale = grad_loss.to(grad_hidden.dtype)
        return (
            (grad_weight * scale) if ctx.has_weight_grad else None,
            (grad_hidden * scale).reshape(ctx.hidden_shape),
            None,
            None,
            None,
            (grad_bias * scale) if ctx.has_bias_grad else None,
            None,
            None,
            None,
            None,
        )


class ChunkedGKDObjective(torch.nn.Module):
    """TRL-exact chunked GKD objective with forward-pass gradient fusion."""

    def __init__(self, contract: ObjectiveContract):
        super().__init__()
        self.contract = contract

    def forward(
        self,
        student_weight: torch.Tensor,
        student_hidden: torch.Tensor,
        teacher_weight: torch.Tensor,
        teacher_hidden: torch.Tensor,
        labels: torch.Tensor,
        student_bias: torch.Tensor | None = None,
        teacher_bias: torch.Tensor | None = None,
    ) -> GKDResult:
        validate_inputs(
            student_weight,
            student_hidden,
            teacher_weight,
            teacher_hidden,
            labels,
            student_bias,
            teacher_bias,
            self.contract,
        )
        outputs = _ChunkedGKDFunction.apply(
            student_weight,
            student_hidden,
            teacher_weight,
            teacher_hidden,
            labels,
            student_bias,
            teacher_bias,
            float(self.contract.beta),
            float(self.contract.temperature),
            int(self.contract.ignore_index),
        )
        loss = outputs[0]
        aux = tuple(item.detach() for item in outputs[1:])
        if len(aux) != len(AUX_NAMES):
            raise AssertionError("chunked oracle produced an unexpected aux count")
        return GKDResult(loss, aux)


def build_objective(contract: ObjectiveContract) -> torch.nn.Module:
    return ChunkedGKDObjective(contract)

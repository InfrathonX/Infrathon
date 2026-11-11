"""Placeholder loss. Replace with on-policy context distillation (see AGENT_TASK.md)."""


def loss_function(args, batch, logits, sum_of_sample_mean):
    raise NotImplementedError(
        "On-policy context distillation is not implemented yet: the teacher must score the context-prefixed "
        "sequence and the loss is the analytic reverse KL over the pre-update student's top-256 tokens."
    )

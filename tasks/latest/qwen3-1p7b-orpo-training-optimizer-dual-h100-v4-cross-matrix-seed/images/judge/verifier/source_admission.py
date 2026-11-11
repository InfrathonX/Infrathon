"""Public/Judge-identical bridge to the canonical source admission policy."""

from runtime._source_policy_impl import AdmissionError, admit, locate_lab

__all__ = ["AdmissionError", "admit", "locate_lab"]

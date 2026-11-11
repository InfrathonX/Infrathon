"""LLM integrity review: a CPU-only gate in front of the GPU Judge.

When a submission arrives, an LLM compares the submitted files with the
agent's starting files, the instructions the agent received and the task's
integrity policy, and answers one question: did the agent work inside the
scope the task defines, or did it circumvent it (bypass the required
framework, special-case the evaluator, read forbidden data, ...)?  It never
grades quality.  See docs/LLM_JUDGE.md.

Modules:

- ``config``: ``SFORGE_LLM_JUDGE_*`` settings (imported by
  ``sforge.harness.config``, so this package's ``__init__`` stays free of
  imports to avoid a cycle).
- ``policy``: per-task ``llm_judge/POLICY.md`` files.
- ``bundle``: the submitted files against the Work image baseline.
- ``prompt``: reviewer instructions, rendering, verdict parsing.
- ``client``: OpenAI-compatible chat-completions client.
- ``reviewer``: orchestration, audit trail, cache, terminal reports.
"""

# LOCAL-ONLY forced-oracle Judge image.  Never pushed, never published, never
# retagged over :local.  It is the task's private Judge image (holdout layer
# included) plus:
#   1. the pinned upstream Liger-Kernel package (pure Python + Triton JIT
#      kernels; Triton compiles them for the running sm_90 device at first use),
#   2. an admission override that records and bypasses the per-file source scan
#      (import allowlist etc.) of runtime/_source_policy_impl.py, with the
#      protected-file manifest updated to the patched hash.
# Everything else (hidden CPU objective cases, formal E2E controller, gates,
# aggregation, report) is the unmodified Judge.
#
# Build (see run_forced_oracle.sh):
#   docker build \
#     --build-context liger=<Liger-Kernel checkout at LIGER_COMMIT> \
#     --build-arg JUDGE_IMAGE=qwen3_1p7b_orpo_dual_h100.judge.qwen3_1p7b_orpo_training_optimizer_dual_h100_v4:local \
#     -t qwen3_1p7b_orpo_dual_h100.judge.qwen3_1p7b_orpo_training_optimizer_dual_h100_v4:forced-oracle \
#     host_reference/forced_judge
ARG JUDGE_IMAGE=qwen3_1p7b_orpo_dual_h100.judge.qwen3_1p7b_orpo_training_optimizer_dual_h100_v4:local
FROM ${JUDGE_IMAGE}

ARG LIGER_COMMIT=780e76b8115bfd7b78592569853b245a33cc7bf0
USER root

COPY --from=liger src/liger_kernel /opt/conda/lib/python3.11/site-packages/liger_kernel
COPY admission_override.py /tmp/forced_oracle/admission_override.py

RUN chown -R root:root /opt/conda/lib/python3.11/site-packages/liger_kernel \
    && find /opt/conda/lib/python3.11/site-packages/liger_kernel -type d -exec chmod 755 {} + \
    && find /opt/conda/lib/python3.11/site-packages/liger_kernel -type f -exec chmod 644 {} + \
    && python3 -c "import liger_kernel.chunked_loss, liger_kernel.transformers.rms_norm, liger_kernel.transformers.rope, liger_kernel.transformers.swiglu" \
    && chmod 644 /app/workdir/orpo_lab/runtime/_source_policy_impl.py /opt/orpo_judge/protected_hashes.txt \
    && python3 /tmp/forced_oracle/admission_override.py \
         --lab /app/workdir/orpo_lab \
         --manifest /opt/orpo_judge/protected_hashes.txt \
    && chmod 444 /app/workdir/orpo_lab/runtime/_source_policy_impl.py \
    && chmod 600 /opt/orpo_judge/protected_hashes.txt \
    && rm -rf /tmp/forced_oracle \
    && find /app/workdir/orpo_lab/runtime -name __pycache__ -prune -exec rm -rf {} + \
    && python /opt/orpo_judge/verifier/verify_orpo.py --verify-protected-files

LABEL org.infrathon.forced-oracle="local-only" \
      org.infrathon.forced-oracle.liger-commit="${LIGER_COMMIT}" \
      org.infrathon.forced-oracle.bypassed="source-admission per-file scan; LLM integrity review (host side, SFORGE_LLM_JUDGE_MODE=off)"

WORKDIR /app/workdir/orpo_lab
ENTRYPOINT []

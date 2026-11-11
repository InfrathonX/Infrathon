# LOCAL-ONLY forced-oracle Judge image for the paged-GQA decode task.
#
# FROM the task's PRIVATE Judge image (it holds the frozen holdout), plus the
# FlashAttention-3 (hopper, sm_90a) package built by ../Dockerfile, plus the
# forced verifier.  The Judge's own files are not modified:
# verify_paged_gqa.py, candidate_worker.py, source_admission.py,
# run_judge.py, protected_hashes.txt and the runtime stay byte-identical (the
# protected-hash check still runs and passes).  Built only by
# run_forced_oracle.sh as <judge image>:forced-oracle; never push it, never
# retag it as :local.
#
# FA3_IMAGE is the paper-reference image that ../run_fa3_reference.sh builds
# from a flash-attention checkout at 0251105a2fb19d2957484b7f023cd8c115286ced
# (same pytorch 2.8.0 / CUDA 12.9 / Python 3.11 base as the Judge image).
ARG JUDGE_IMAGE=standalone_paged_gqa_decode.judge.standalone_paged_gqa_decode_bf16_h100_scratch_v2:local
ARG FA3_IMAGE=standalone_paged_gqa_decode.paper-fa3:0251105a
FROM ${FA3_IMAGE} AS fa3

FROM ${JUDGE_IMAGE}
COPY --from=fa3 /opt/conda/lib/python3.11/site-packages/flash_attn_3 /opt/conda/lib/python3.11/site-packages/flash_attn_3
COPY --from=fa3 /opt/conda/lib/python3.11/site-packages/flash_attn_3-3.0.0.dist-info /opt/conda/lib/python3.11/site-packages/flash_attn_3-3.0.0.dist-info
COPY --from=fa3 /opt/conda/lib/python3.11/site-packages/flash_attn_interface.py /opt/conda/lib/python3.11/site-packages/flash_attn_config.py /opt/conda/lib/python3.11/site-packages/
COPY --from=fa3 /opt/conda/lib/python3.11/site-packages/einops /opt/conda/lib/python3.11/site-packages/einops
COPY --from=fa3 /opt/conda/lib/python3.11/site-packages/einops-0.8.2.dist-info /opt/conda/lib/python3.11/site-packages/einops-0.8.2.dist-info
RUN python -c "import torch, flash_attn_3._C; assert hasattr(torch.ops.flash_attn_3, 'fwd'); print('fa3 ok')"

COPY forced_admission.py forced_verify.py forced_worker.py /opt/standalone_paged_gqa_decode_judge/
RUN chown root:root /opt/standalone_paged_gqa_decode_judge/forced_admission.py \
       /opt/standalone_paged_gqa_decode_judge/forced_verify.py \
       /opt/standalone_paged_gqa_decode_judge/forced_worker.py \
    && chmod 0400 /opt/standalone_paged_gqa_decode_judge/forced_admission.py \
       /opt/standalone_paged_gqa_decode_judge/forced_verify.py \
    && chmod 0444 /opt/standalone_paged_gqa_decode_judge/forced_worker.py

LABEL org.infrathon.forced-oracle="local-only; source admission bypassed; sanitizer and edge failures recorded, not fatal; do not publish"

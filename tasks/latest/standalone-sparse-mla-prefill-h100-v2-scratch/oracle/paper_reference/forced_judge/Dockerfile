# FORCED-ORACLE Judge image (LOCAL ONLY: never push, never tag :local).
#
# = the task's PRIVATE Judge image (public Judge + frozen holdouts, built or pulled
#   by scripts/build_images.sh) with exactly two changes:
#   1. upstream FlashMLA (deepseek-ai/FlashMLA @ ba89a3466e9470ad08ab39738d4e7bb66989e1e7,
#      sm_90a, + ../flashmla_out_abi.patch) copied in as an installed Python package,
#      taken from the paper-reference image that ../Dockerfile builds
#      (`../run_flashmla_reference.sh --build-only`, FLASHMLA_SOURCE = upstream checkout);
#   2. the Judge's source_admission.py replaced by forced_source_admission.py (the
#      original is kept beside it as source_admission_enforcing.py and still makes
#      every structural decision), with protected_hashes.txt updated for that one file.
# verify_sparse_mla.py, run_judge.py, candidate_worker.py, the trusted runtime, the
# compute-sanitizer pass, the holdouts, timing and aggregation are untouched.
# Built only by run_forced_oracle.sh in this directory.
ARG JUDGE_IMAGE=standalone_sparse_mla_prefill.judge.standalone_sparse_mla_prefill_h100_scratch_v2:local
ARG PAPER_IMAGE=standalone_sparse_mla_prefill.paper-oracle-flashmla:local
FROM ${PAPER_IMAGE} AS paper

FROM ${JUDGE_IMAGE}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG SITE=/opt/conda/lib/python3.11/site-packages
ARG JUDGE_ROOT=/opt/standalone_sparse_mla_prefill_judge

COPY --from=paper /opt/conda/lib/python3.11/site-packages/flash_mla /opt/conda/lib/python3.11/site-packages/flash_mla
COPY --from=paper /opt/conda/lib/python3.11/site-packages/flash_mla-1.0.0+ba89a34.dist-info /opt/conda/lib/python3.11/site-packages/flash_mla-1.0.0+ba89a34.dist-info
COPY forced_source_admission.py /tmp/forced_source_admission.py

RUN set -eux; \
    python -c "import torch, triton; assert torch.__version__.split('+')[0] == '2.8.0'; assert torch.version.cuda == '12.9'; assert triton.__version__ == '3.4.0'"; \
    chmod -R a+rX "${SITE}/flash_mla" "${SITE}/flash_mla-1.0.0+ba89a34.dist-info"; \
    cd / && python -c "import flash_mla, flash_mla.cuda as c; assert hasattr(c, 'sparse_prefill_fwd_out'); print('flash_mla', flash_mla.__file__)"; \
    cd "${JUDGE_ROOT}"; \
    test -f source_admission.py && test ! -e source_admission_enforcing.py; \
    old="$(sha256sum source_admission.py | awk '{print $1}')"; \
    grep -qx "${old} judge:source_admission.py" protected_hashes.txt; \
    mv source_admission.py source_admission_enforcing.py; \
    install -o root -g root -m 0400 /tmp/forced_source_admission.py source_admission.py; \
    rm -f /tmp/forced_source_admission.py; \
    new="$(sha256sum source_admission.py | awk '{print $1}')"; \
    sed -i "s/^${old} judge:source_admission.py\$/${new} judge:source_admission.py/" protected_hashes.txt; \
    grep -qx "${new} judge:source_admission.py" protected_hashes.txt; \
    chmod 0400 source_admission_enforcing.py protected_hashes.txt; \
    chown root:root source_admission_enforcing.py protected_hashes.txt; \
    test -f private/final_cases.json && test -f private/edge_cases.json

LABEL org.infrathon.forced-oracle="true" \
      org.infrathon.forced-oracle.paper="deepseek-ai/FlashMLA@ba89a3466e9470ad08ab39738d4e7bb66989e1e7" \
      org.infrathon.forced-oracle.bypassed="source admission verdict; LLM integrity review is skipped by the run script"

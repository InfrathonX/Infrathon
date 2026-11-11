# FORCED-ORACLE Judge image (LOCAL ONLY: never push, never tag :local).
#
# = the task's PRIVATE Judge image (public Judge + frozen holdouts, built or pulled
#   by scripts/build_images.sh) with exactly two changes:
#   1. upstream DeepGEMM (deepseek-ai/DeepGEMM @ 559d79fb6994a58b8a15b4b93bf13ccc16edf247,
#      v2.6.1, built in place by develop.sh) copied in at /opt/deepgemm from the
#      paper-reference image that ../Dockerfile builds (DEEPGEMM_SOURCE = upstream
#      checkout), and made importable by a site-packages symlink deep_gemm ->
#      /opt/deepgemm/deep_gemm (the Judge passes no PYTHONPATH to the worker).  DeepGEMM JIT-compiles its kernels with
#      the image's nvcc at first use, into the worker's ~/.deep_gemm (cold cache).
#   2. the Judge's source_admission.py replaced by forced_source_admission.py (the
#      original is kept beside it as source_admission_enforcing.py and still makes
#      every structural decision), with protected_hashes.txt updated for that one file.
# verify_grouped.py, run_judge.py, candidate_worker.py, the trusted runtime, the
# compute-sanitizer pass, the holdouts, timing and aggregation are untouched.
# Built only by run_forced_oracle.sh in this directory.
ARG JUDGE_IMAGE=standalone_grouped_fp8.judge.standalone_dynamic_grouped_fp8_h100_scratch_v3:local
ARG PAPER_IMAGE=standalone_grouped_fp8.paper-reference:local
FROM ${PAPER_IMAGE} AS paper

FROM ${JUDGE_IMAGE}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG SITE=/opt/conda/lib/python3.11/site-packages
ARG JUDGE_ROOT=/opt/standalone_grouped_fp8_judge

COPY --from=paper /opt/deepgemm /opt/deepgemm
COPY forced_source_admission.py /tmp/forced_source_admission.py

RUN set -eux; \
    python -c "import torch, triton; assert torch.__version__.split('+')[0] == '2.8.0'; assert torch.version.cuda == '12.9'; assert triton.__version__ == '3.4.0'"; \
    chmod -R a+rX /opt/deepgemm; \
    test ! -e "${SITE}/deep_gemm"; \
    ln -s /opt/deepgemm/deep_gemm "${SITE}/deep_gemm"; \
    cd / && python -c "import deep_gemm; assert deep_gemm.__version__ == '2.6.1', deep_gemm.__version__; assert hasattr(deep_gemm, 'm_grouped_fp8_gemm_nt_masked'); assert hasattr(deep_gemm, 'm_grouped_fp8_gemm_nt_contiguous'); print('deep_gemm', deep_gemm.__file__)"; \
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
      org.infrathon.forced-oracle.paper="deepseek-ai/DeepGEMM@559d79fb6994a58b8a15b4b93bf13ccc16edf247" \
      org.infrathon.forced-oracle.bypassed="source admission verdict; LLM integrity review is skipped by the run script"

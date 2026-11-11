# LOCAL-ONLY forced-oracle Judge image for the K-grouped FP8 wgrad task.
#
# FROM the task's PRIVATE Judge image (it holds the frozen holdout), plus
# upstream DeepGEMM built for sm_90a, plus the forced verifier.  The Judge's
# own files are not modified: verify_wgrad.py, candidate_worker.py,
# source_admission.py, run_judge.py, protected_hashes.txt and the runtime stay
# byte-identical (the protected-hash check still runs and passes).  Built only
# by oracle/forced_judge/run_forced_oracle.sh as <judge image>:forced-oracle;
# never push it, never retag it as :local.
#
# Named build contexts:
#   deepgemm  a DeepGEMM checkout at 559d79fb6994a58b8a15b4b93bf13ccc16edf247
#             with third-party/cutlass f3fde583 and third-party/fmt 553ec11e
#   (default) this directory (forced_*.py)
ARG JUDGE_IMAGE=standalone_k_grouped_wgrad.judge.standalone_k_grouped_fp8_wgrad_h100_scratch_v2:local

# Stage 1 is instruction-for-instruction the head of oracle/Dockerfile, so an
# existing oracle image build is reused from the Docker cache.
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel AS deepgemm_build

ARG DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda PATH=/usr/local/cuda/bin:${PATH} \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 DG_USE_NVIDIA_TOOLS=0 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends build-essential ca-certificates git ninja-build \
    && rm -rf /var/lib/apt/lists/* && git config --system safe.directory '*'

COPY --from=deepgemm . /opt/deepgemm/
RUN cd /opt/deepgemm && DG_JIT_CACHE_DIR=/tmp/deepgemm-image-jit bash develop.sh \
    && PYTHONPATH=/opt/deepgemm python -c "import deep_gemm; assert deep_gemm.__version__ == '2.6.1', deep_gemm.__version__" \
    && rm -rf /tmp/deepgemm-image-jit

FROM ${JUDGE_IMAGE}

# DeepGEMM in place (develop.sh layout: build/ .so symlinked into deep_gemm/,
# CUTLASS includes symlinked from third-party/).  The Judge's worker clears
# PYTHONPATH, so a .pth file puts /opt/deepgemm on sys.path.  Its JIT cache
# defaults to $HOME/.deep_gemm, i.e. /home/agent/.deep_gemm for the worker
# (cold at the start of every Judge container, like the Triton cache).
COPY --from=deepgemm_build /opt/deepgemm /opt/deepgemm
RUN chmod -R a+rX /opt/deepgemm \
    && echo /opt/deepgemm > /opt/conda/lib/python3.11/site-packages/zz_forced_oracle_deepgemm.pth \
    && python -c "import deep_gemm; assert deep_gemm.__version__ == '2.6.1', deep_gemm.__version__"

COPY forced_admission.py forced_verify.py /opt/standalone_k_grouped_wgrad_judge/
RUN chown root:root /opt/standalone_k_grouped_wgrad_judge/forced_admission.py \
       /opt/standalone_k_grouped_wgrad_judge/forced_verify.py \
    && chmod 0400 /opt/standalone_k_grouped_wgrad_judge/forced_admission.py \
       /opt/standalone_k_grouped_wgrad_judge/forced_verify.py

# Per-case isolation makes the run longer than the Judge's 3300 s budget; the
# trial/warm-up counts and the per-process candidate timeout stay the Judge's.
ENV STANDALONE_KGROUPED_WGRAD_TOTAL_TIMEOUT=13800 \
    STANDALONE_KGROUPED_WGRAD_JUDGE_TIMEOUT=14100 \
    FORCED_EDGE_CASE_TIMEOUT=240 \
    FORCED_SANITIZER_CASE_TIMEOUT=300

LABEL org.infrathon.forced-oracle="local-only; source admission bypassed; per-case isolation; do not publish"

# Paper-reference image: the task's own Judge image (same torch 2.8.0 / CUDA 12.9 /
# Triton 3.4.0, same trusted runtime at /opt/standalone_grouped_fp8_worker) + upstream
# DeepGEMM 559d79f built in place.  Host-only; never published, never a scored Judge,
# never a Work or Judge build input.  Built only by oracle/paper_reference/run_paper_reference.sh.
ARG JUDGE_IMAGE=standalone_grouped_fp8.judge.standalone_dynamic_grouped_fp8_h100_scratch_v3:local
FROM ${JUDGE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda PATH=/usr/local/cuda/bin:${PATH} \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 DG_USE_NVIDIA_TOOLS=0

RUN apt-get update && apt-get install -y --no-install-recommends git ninja-build build-essential \
    && rm -rf /var/lib/apt/lists/* && git config --system safe.directory '*'

# upstream checkout (with third-party/cutlass f3fde58 and third-party/fmt 553ec11)
COPY --from=deepgemm . /opt/deepgemm/
RUN cd /opt/deepgemm && DG_JIT_CACHE_DIR=/tmp/dg-build-jit bash develop.sh \
    && PYTHONPATH=/opt/deepgemm python -c "import deep_gemm; assert deep_gemm.__version__ == '2.6.1', deep_gemm.__version__" \
    && rm -rf /tmp/dg-build-jit

COPY oracle_runner.py /opt/paper_harness/oracle_runner.py
ENV PYTHONPATH=/opt/deepgemm DG_JIT_CACHE_DIR=/tmp/deepgemm-jit
WORKDIR /opt/paper_harness
ENTRYPOINT []

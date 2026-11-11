# Oracle image: base + upstream DeepGEMM + the trusted runtime.
#
# This image is NOT the task's Work or Judge image.  It is built only by
# oracle/run_oracle.sh, carries its own tag, and exists so a host
# operator can measure an upstream oracle under the Judge's own protocol.
# Same base as the Work/Judge images (pytorch 2.8.0 / CUDA 12.9 / Triton 3.4.0).
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda PATH=/usr/local/cuda/bin:${PATH} \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 DG_USE_NVIDIA_TOOLS=0 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends build-essential ca-certificates git ninja-build \
    && rm -rf /var/lib/apt/lists/* && git config --system safe.directory '*'

# Upstream DeepGEMM 559d79fb6994a58b8a15b4b93bf13ccc16edf247 (v2.6.1) with submodules
# third-party/cutlass f3fde583 and third-party/fmt 553ec11e, built in place.
COPY --from=deepgemm . /opt/deepgemm/
RUN cd /opt/deepgemm && DG_JIT_CACHE_DIR=/tmp/deepgemm-image-jit bash develop.sh \
    && PYTHONPATH=/opt/deepgemm python -c "import deep_gemm; assert deep_gemm.__version__ == '2.6.1', deep_gemm.__version__" \
    && rm -rf /tmp/deepgemm-image-jit

COPY --from=task_runtime k_grouped_wgrad_runtime.py /opt/oracle/k_grouped_wgrad_runtime.py
COPY --from=task_baseline candidates/kernel_v001/submission.py /opt/oracle/candidates/baseline_triton_v001/submission.py
COPY oracle_runner.py /opt/oracle/oracle_runner.py
COPY candidates /opt/oracle/candidates/

ENV PYTHONPATH=/opt/deepgemm:/opt/oracle DG_JIT_CACHE_DIR=/tmp/deepgemm-jit
WORKDIR /opt/oracle
ENTRYPOINT []

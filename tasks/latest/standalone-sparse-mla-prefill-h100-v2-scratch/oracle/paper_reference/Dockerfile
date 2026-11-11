# Host-only paper-reference image: the Judge's base image + upstream FlashMLA
# (deepseek-ai/FlashMLA ba89a3466e9470ad08ab39738d4e7bb66989e1e7, sm_90a) with
# a small ABI glue patch, plus the task's trusted runtime.
#
# This image is NOT the task's Work or Judge image.  It is built only by
# oracle/paper_reference/run_flashmla_reference.sh, carries its own tag, and
# exists so a host operator can time upstream FlashMLA with the Judge's own
# runtime.  Build contexts: `flashmla` = a FlashMLA checkout at the pinned
# commit with csrc/cutlass initialised; `task_runtime` = the task's runtime/;
# the default context is this directory.
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    FLASH_MLA_DISABLE_SM100=1 \
    NVCC_THREADS=4 \
    MAX_JOBS=8 \
    TORCH_CUDA_ARCH_LIST=9.0a

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates git ninja-build patch \
    && rm -rf /var/lib/apt/lists/*

# Same torch / CUDA / Triton as the Judge image.
RUN python -c "import torch, triton; assert torch.__version__.split('+')[0] == '2.8.0'; assert torch.version.cuda == '12.9'; assert triton.__version__ == '3.4.0'"

COPY --from=flashmla . /opt/flashmla/
COPY flashmla_out_abi.patch /opt/flashmla_out_abi.patch
RUN cd /opt/flashmla \
    && patch -p1 < /opt/flashmla_out_abi.patch \
    && test -f csrc/cutlass/include/cute/tensor.hpp \
    && (pip install --no-build-isolation --no-deps -v . 2>&1 | tail -60) \
    && pip show flash_mla \
    && cd / && python -c "import flash_mla, flash_mla.cuda as c; assert hasattr(c, 'sparse_prefill_fwd_out'); print('flash_mla ok')"

COPY --from=task_runtime sparse_mla_prefill_runtime.py /opt/oracle/sparse_mla_prefill_runtime.py
COPY oracle_runner.py /opt/oracle/oracle_runner.py
COPY candidates /opt/oracle/candidates/
WORKDIR /opt/oracle
ENTRYPOINT []

FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRITON_CACHE_DIR=/app/workdir/sparse_mla_prefill_lab/build/triton-cache \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl git jq ninja-build procps ripgrep xz-utils \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -s /bin/bash agent \
    && git config --system safe.directory '*'

# This image is agent-free: the agent CLI (Codex, Claude Code, ...) arrives
# as a host-provided runtime that SForge bind-mounts read-only at
# /opt/agent-runtime when SFORGE_AGENT_RUNTIME_DIR is set.  Switching agent
# kind or version never rebuilds this image.
RUN python - <<'PY'
import importlib.util
import torch
import triton

assert torch.__version__ == "2.8.0+cu129", torch.__version__
assert torch.version.cuda == "12.9", torch.version.cuda
assert triton.__version__ == "3.4.0", triton.__version__
for absent in ("flash_mla", "flash_attn", "deep_gemm", "flashinfer", "xformers"):
    assert importlib.util.find_spec(absent) is None, absent
PY

COPY benchmark/ /workspace/benchmark/
COPY workdir/ /app/workdir/
COPY AGENT_TASK.md /app/workdir/sparse_mla_prefill_lab/AGENT_TASK.md
COPY benchmark/prepare_submit.sh /app/workdir/sparse_mla_prefill_lab/.sforge/prepare_submit.sh
COPY --chown=agent:agent codex-config.toml /home/agent/.codex/config.toml

# The public --edge and --sanitize suites need the shipped public edge manifest
# (validated against the pinned runtime here) and compute-sanitizer.
RUN cd /workspace/benchmark && python - <<'PY'
import os

import sparse_mla_prefill_runtime as runtime
import public_runner

cases = public_runner.public_edge_cases()
assert all(runtime.validate_edge_case(case)["native"] for case in cases)
assert any(case["sanitize"] for case in cases)
assert os.access(public_runner.SANITIZER, os.X_OK), public_runner.SANITIZER
PY

RUN chmod 600 /home/agent/.codex/config.toml \
    && chmod 755 /workspace/benchmark/*.py /workspace/benchmark/*.sh \
       /app/workdir/sparse_mla_prefill_lab/tools/*.sh \
    && mkdir -p /app/workdir/sparse_mla_prefill_lab/build/triton-cache /logs/agent \
    && chown -R agent:agent /app/workdir /logs/agent \
    && chown -R root:root /workspace/benchmark \
    && chmod -R a-w /workspace/benchmark \
    && chown -R root:root /app/workdir/sparse_mla_prefill_lab/.sforge \
    && chmod 555 /app/workdir/sparse_mla_prefill_lab/.sforge \
       /app/workdir/sparse_mla_prefill_lab/.sforge/prepare_submit.sh

USER agent
WORKDIR /app/workdir/sparse_mla_prefill_lab
ENTRYPOINT []
CMD ["bash"]

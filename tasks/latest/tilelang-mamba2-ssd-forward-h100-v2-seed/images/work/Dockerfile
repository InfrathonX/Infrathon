FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TILELANG_CACHE_DIR=/app/workdir/mamba2_ssd_lab/build/tilelang-cache \
    TILELANG_TMP_DIR=/app/workdir/mamba2_ssd_lab/build/tilelang-tmp \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl git jq ninja-build procps ripgrep xz-utils \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -s /bin/bash agent \
    && git config --system safe.directory '*'

# TileLang compiles through nvcc at JIT time, so this image must carry a full
# CUDA toolkit with headers, not just a driver.  The -devel base supplies it;
# a runtime base fails with `fatal error: cuda_runtime.h: No such file`.
RUN pip install --no-cache-dir tilelang==0.1.13

# This image is agent-free: the agent CLI (Codex, Claude Code, ...) arrives
# as a host-provided runtime that SForge bind-mounts read-only at
# /opt/agent-runtime when SFORGE_AGENT_RUNTIME_DIR is set.  Switching agent
# kind or version never rebuilds this image.
RUN python - <<'PY'
import importlib.util
import torch
import triton
import tilelang

assert torch.__version__ == "2.8.0+cu129", torch.__version__
assert torch.version.cuda == "12.9", torch.version.cuda
assert triton.__version__ == "3.4.0", triton.__version__
assert tilelang.__version__ == "0.1.13", tilelang.__version__
for absent in ("mamba_ssm", "fla", "helion", "flash_attn", "flashinfer", "deep_gemm", "xformers"):
    assert importlib.util.find_spec(absent) is None, absent
PY

COPY benchmark/ /workspace/benchmark/
# The public captured layer (fixtures/recipe/real_layers/, target `public`).
COPY --from=public_layer public-real-layer16.pt real_layers.manifest.json /workspace/benchmark/public_layer/
COPY workdir/ /app/workdir/
COPY AGENT_TASK.md /app/workdir/mamba2_ssd_lab/AGENT_TASK.md
COPY benchmark/prepare_submit.sh /app/workdir/mamba2_ssd_lab/.sforge/prepare_submit.sh
COPY benchmark/prepare_submit.sh /app/workdir/mamba2_ssd_lab/.sforge/prepare_submit.sh
COPY --chown=agent:agent codex-config.toml /home/agent/.codex/config.toml

RUN chmod 600 /home/agent/.codex/config.toml \
    && chmod 755 /workspace/benchmark/*.py /workspace/benchmark/*.sh \
       /app/workdir/mamba2_ssd_lab/tools/*.sh \
    && mkdir -p /app/workdir/mamba2_ssd_lab/build/tilelang-cache \
       /app/workdir/mamba2_ssd_lab/build/tilelang-tmp /logs/agent \
    && chown -R agent:agent /app/workdir /logs/agent \
    && chown -R root:root /workspace/benchmark \
    && chmod -R a-w /workspace/benchmark \
    && chown -R root:root /app/workdir/mamba2_ssd_lab/.sforge \
    && chmod 555 /app/workdir/mamba2_ssd_lab/.sforge \
       /app/workdir/mamba2_ssd_lab/.sforge/prepare_submit.sh

USER agent
WORKDIR /app/workdir/mamba2_ssd_lab
ENTRYPOINT []
CMD ["bash"]

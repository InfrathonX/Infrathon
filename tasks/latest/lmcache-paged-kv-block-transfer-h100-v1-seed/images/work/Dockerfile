FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl git jq procps ripgrep xz-utils \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -s /bin/bash agent \
    && git config --system safe.directory '*'

# Submissions are single CUDA translation units compiled with the frozen nvcc
# command, so this image must carry the full toolkit; the -devel base supplies
# it.  There is no Python dependency beyond torch (harness-side only).
RUN python - <<'PY'
import subprocess
import torch

assert torch.__version__ == "2.8.0+cu129", torch.__version__
assert torch.version.cuda == "12.9", torch.version.cuda
release = subprocess.run(
    ["/usr/local/cuda/bin/nvcc", "--version"], capture_output=True, text=True, check=True
).stdout
assert "release 12.9" in release, release
PY

# This image is agent-free: the agent CLI arrives as a host-provided runtime
# bind-mounted read-only at /opt/agent-runtime when SFORGE_AGENT_RUNTIME_DIR
# is set.  Switching agent kind or version never rebuilds this image.
COPY benchmark/ /workspace/benchmark/
COPY workdir/ /app/workdir/
COPY AGENT_TASK.md /app/workdir/kv_transfer_lab/AGENT_TASK.md
COPY benchmark/prepare_submit.sh /app/workdir/kv_transfer_lab/.sforge/prepare_submit.sh
COPY --chown=agent:agent codex-config.toml /home/agent/.codex/config.toml

RUN chmod 600 /home/agent/.codex/config.toml \
    && chmod 755 /workspace/benchmark/*.py /workspace/benchmark/*.sh \
       /app/workdir/kv_transfer_lab/tools/*.sh \
    && mkdir -p /app/workdir/kv_transfer_lab/build /logs/agent \
    && chown -R agent:agent /app/workdir /logs/agent \
    && chown -R root:root /workspace/benchmark \
    && chmod -R a-w /workspace/benchmark \
    && chown -R root:root /app/workdir/kv_transfer_lab/.sforge \
    && chmod 555 /app/workdir/kv_transfer_lab/.sforge \
       /app/workdir/kv_transfer_lab/.sforge/prepare_submit.sh

USER agent
WORKDIR /app/workdir/kv_transfer_lab
ENTRYPOINT []
CMD ["bash"]

FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTHONDONTWRITEBYTECODE=1
ENV TRITON_CACHE_DIR=/tmp/triton-cache-agent
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      git \
      jq \
      procps \
      ripgrep \
      sudo \
      xz-utils \
    && rm -rf /var/lib/apt/lists/* \
    && (id -u agent >/dev/null 2>&1 || useradd -m -s /bin/bash agent) \
    && echo 'agent ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/mlsysforge-agent \
    && chmod 440 /etc/sudoers.d/mlsysforge-agent \
    && git config --system safe.directory '*'

# This image is agent-free: the agent CLI (Codex, Claude Code, ...) arrives
# as a host-provided runtime that SForge bind-mounts read-only at
# /opt/agent-runtime when SFORGE_AGENT_RUNTIME_DIR is set.  Switching agent
# kind or version never rebuilds this image.
COPY workdir/ /app/workdir/
COPY public_tests/ /workspace/public_tests/
COPY AGENT_TASK.md /app/workdir/flash_kmeans_lab/AGENT_TASK.md
COPY public_tests/prepare_submit.sh /app/workdir/flash_kmeans_lab/.sforge/prepare_submit.sh
COPY --chown=agent:agent codex-config.toml /home/agent/.codex/config.toml

RUN mkdir -p \
      /app/workdir/flash_kmeans_lab/kernels \
      /workspace/public_tests/results \
      /logs/agent \
      /tmp/triton-cache-agent \
    && chmod 555 \
      /workspace/public_tests \
      /workspace/public_tests/prepare_submit.sh \
      /workspace/public_tests/run_public_tests.sh \
    && chmod 444 \
      /workspace/public_tests/public_test.py \
      /workspace/public_tests/public_fresh_worker.py \
    && chown -R agent:agent \
      /app/workdir \
      /workspace/public_tests/results \
      /logs/agent \
      /tmp/triton-cache-agent \
      /home/agent/.codex \
    && chown -R root:root \
      /app/workdir/flash_kmeans_lab/.sforge \
    && chmod 555 \
      /app/workdir/flash_kmeans_lab/.sforge \
      /app/workdir/flash_kmeans_lab/.sforge/prepare_submit.sh \
    && chmod 600 /home/agent/.codex/config.toml

USER agent
WORKDIR /app/workdir/flash_kmeans_lab

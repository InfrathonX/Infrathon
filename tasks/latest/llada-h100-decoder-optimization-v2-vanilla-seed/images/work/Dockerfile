FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive
# Every architecture inside the capability envelope (sm_80 and up), so a
# candidate that compiles CUDA/Triton kernels is not silently pinned to one
# GPU generation.  Override with --build-arg for a single-target build.
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0"

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV TOKENIZERS_PARALLELISM=false
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

RUN python3 -m pip install --no-cache-dir \
      accelerate==1.4.0 \
      huggingface-hub==0.29.3 \
      safetensors==0.5.3 \
      tokenizers==0.21.0 \
      transformers==4.49.0

# This image is agent-free: the agent CLI (Codex, Claude Code, ...) arrives
# as a host-provided runtime that SForge bind-mounts read-only at
# /opt/agent-runtime when SFORGE_AGENT_RUNTIME_DIR is set.  Switching agent
# kind or version never rebuilds this image.
COPY workdir/ /app/workdir/
COPY public_tests/ /workspace/public_tests/
COPY AGENT_TASK.md /app/workdir/decoder_lab/AGENT_TASK.md
# Cooperative pre-submit drain hook of the single-accelerator profile.  It is
# installed root-owned and non-writable inside the task cwd; sforge-submit
# invokes it before archive/HTTP admission when SFORGE_PRE_SUBMIT_HOOK names it.
COPY public_tests/prepare_submit.sh /app/workdir/decoder_lab/.sforge/prepare_submit.sh
COPY --chown=agent:agent codex-config.toml /home/agent/.codex/config.toml

RUN mkdir -p /workspace/public_tests/results /logs/agent /tmp/triton-cache-agent \
    && chmod 555 /workspace/public_tests /workspace/public_tests/run_public_tests.sh \
    && find /workspace/public_tests -type f ! -name run_public_tests.sh -exec chmod 444 {} + \
    && chown -R agent:agent \
      /app/workdir \
      /workspace/public_tests/results \
      /logs/agent \
      /tmp/triton-cache-agent \
      /home/agent/.codex \
    && chmod 600 /home/agent/.codex/config.toml \
    && chown -R root:root /app/workdir/decoder_lab/.sforge \
    && chmod 555 /app/workdir/decoder_lab/.sforge /app/workdir/decoder_lab/.sforge/prepare_submit.sh

USER agent
WORKDIR /app/workdir/decoder_lab

FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive
# Capability envelope, not one card: every architecture the task admits
# (sm_80 A100, sm_86 A6000, sm_89 L40S, sm_90 H100/H20/H200, sm_100 B200).
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0"
ENV CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:${PATH} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false TRITON_CACHE_DIR=/tmp/triton-cache-agent \
    EAGLE3_TARGET_REPO=/opt/models/qwen3-8b-hf EAGLE3_HEAD_DIR=/opt/models/eagle3-head

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git jq procps ripgrep sudo xz-utils \
 && rm -rf /var/lib/apt/lists/* \
 && (id -u agent >/dev/null 2>&1 || useradd -m -u 1000 -s /bin/bash agent) \
 && echo 'agent ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/mlsysforge-agent \
 && chmod 440 /etc/sudoers.d/mlsysforge-agent \
 && test -x /usr/local/cuda/bin/nvcc

RUN python3 -m pip install --no-cache-dir transformers==4.53.1 tokenizers==0.21.4 \
    safetensors==0.8.0 accelerate==1.14.0 numpy==2.4.6 huggingface-hub==0.33.5 pytest==9.1.1

COPY workdir/ /app/workdir/
COPY public_tests/ /workspace/public_tests/
COPY AGENT_TASK.md /app/workdir/spec_lab/AGENT_TASK.md
COPY public_tests/prepare_submit.sh /app/workdir/spec_lab/.sforge/prepare_submit.sh
COPY --chown=agent:agent codex-config.toml /home/agent/.codex/config.toml
RUN mkdir -p /workspace/public_tests/results /logs/agent /tmp/triton-cache-agent /app/workdir/spec_lab/build \
 && chown -R agent:agent /app/workdir /workspace/public_tests/results /logs/agent /tmp/triton-cache-agent /home/agent/.codex \
 && chown -R root:root /app/workdir/spec_lab/runtime /app/workdir/spec_lab/public_prompts \
 && find /app/workdir/spec_lab/runtime /app/workdir/spec_lab/public_prompts -type d -exec chmod 0555 {} + \
 && find /app/workdir/spec_lab/runtime /app/workdir/spec_lab/public_prompts -type f -exec chmod 0444 {} + \
 && chmod 0555 /workspace/public_tests /workspace/public_tests/run_public_tests.sh \
 && find /workspace/public_tests -type f ! -name run_public_tests.sh -exec chmod 0444 {} + \
 && chmod 0600 /home/agent/.codex/config.toml \
 && chown -R root:root /app/workdir/spec_lab/.sforge \
 && chmod 0555 /app/workdir/spec_lab/.sforge /app/workdir/spec_lab/.sforge/prepare_submit.sh
USER agent
WORKDIR /app/workdir/spec_lab

FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV TOKENIZERS_PARALLELISM=false
ENV TRITON_CACHE_DIR=/tmp/triton-cache-agent
ENV GKD_STUDENT_MODEL_PATH=/models/Qwen3-1.7B
ENV GKD_TEACHER_MODEL_PATH=/models/Qwen3-4B
ENV GKD_DATA_PATH=/datasets/tldr-gkd
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      git \
      jq \
      ninja-build \
      procps \
      ripgrep \
      sudo \
      xz-utils \
    && rm -rf /var/lib/apt/lists/* \
    && (id -u agent >/dev/null 2>&1 || useradd -m -s /bin/bash agent) \
    && echo 'agent ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/mlsysforge-agent \
    && chmod 440 /etc/sudoers.d/mlsysforge-agent \
    && git config --system safe.directory '*'

# This is the frozen official Hugging Face TRL 0.12.1 stack.  Verify both the
# published wheel and the exact upstream GKDTrainer source during the build;
# runtime/hf_reference.py repeats the source check before constructing v001.
RUN mkdir -p /tmp/trl-wheel \
    && python3 -m pip download --no-deps --only-binary=:all: \
      --dest /tmp/trl-wheel trl==0.12.1 \
    && echo '502b958ac95313bd3e1a31ce91bc852b14f90303bc9b973ede1ee959f40c5c5a  /tmp/trl-wheel/trl-0.12.1-py3-none-any.whl' \
      | sha256sum --check --strict \
    && python3 -m pip install --no-cache-dir \
      accelerate==1.6.0 \
      datasets==2.21.0 \
      huggingface-hub==0.30.2 \
      pytest==8.3.5 \
      safetensors==0.5.3 \
      tokenizers==0.21.1 \
      transformers==4.51.3 \
      /tmp/trl-wheel/trl-0.12.1-py3-none-any.whl \
    && python3 -c "import hashlib, inspect, pathlib, trl; from trl.trainer.gkd_trainer import GKDTrainer; source = pathlib.Path(inspect.getfile(GKDTrainer)); assert trl.__version__ == '0.12.1'; assert hashlib.sha256(source.read_bytes()).hexdigest() == '09e6e8932297ca3793aa5c429852bb46e3f395f9f3aaa48479d3b7965a028f1a'" \
    && rm -rf /tmp/trl-wheel

# This image is agent-free: the agent CLI (Codex, Claude Code, ...) arrives
# as a host-provided runtime that SForge bind-mounts read-only at
# /opt/agent-runtime when SFORGE_AGENT_RUNTIME_DIR is set.  Switching agent
# kind or version never rebuilds this image.
COPY workdir/ /app/workdir/
COPY public_tests/ /workspace/public_tests/
COPY AGENT_TASK.md /app/workdir/gkd_lab/AGENT_TASK.md
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
    && chmod 600 /home/agent/.codex/config.toml

USER agent
WORKDIR /app/workdir/gkd_lab

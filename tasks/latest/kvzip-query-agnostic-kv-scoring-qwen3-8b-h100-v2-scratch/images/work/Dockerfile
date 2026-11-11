FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel
ARG DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda PATH=/usr/local/cuda/bin:${PATH} \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 KV_SCORING_MODEL_DIR=/opt/models/qwen3-8b/snapshots/unknown \
    KV_SCORING_CORPUS_DIR=/workspace/benchmark/corpus
RUN apt-get update && apt-get install -y --no-install-recommends build-essential ca-certificates curl git jq ninja-build procps ripgrep xz-utils \
 && rm -rf /var/lib/apt/lists/* && useradd -m -s /bin/bash agent && git config --system safe.directory '*'
RUN pip install --no-cache-dir transformers==4.57.1 accelerate==1.10.1 safetensors==0.6.2 rouge==1.0.1 rouge-score==0.1.2 tree-sitter==0.21.3 tree-sitter-languages==1.10.2 nltk==3.9.1
RUN python - <<'PY'
import importlib.util, torch, triton, transformers
assert torch.__version__ == "2.8.0+cu129"
assert torch.version.cuda == "12.9" and triton.__version__ == "3.4.0"
assert transformers.__version__ == "4.57.1"
for absent in ("kvzip", "flash_attn", "xformers", "flashinfer"):
    assert importlib.util.find_spec(absent) is None, absent
PY
COPY --from=task_runtime kv_scoring_runtime.py kv_scoring_metric.py /workspace/runtime/
COPY --from=corpus_public . /workspace/benchmark/corpus/
COPY benchmark/ /workspace/benchmark/
COPY workdir/ /app/workdir/
COPY AGENT_TASK.md /app/workdir/kv_scoring_lab/AGENT_TASK.md
COPY benchmark/prepare_submit.sh /app/workdir/kv_scoring_lab/.sforge/prepare_submit.sh
COPY --chown=agent:agent codex-config.toml /home/agent/.codex/config.toml
RUN chmod 600 /home/agent/.codex/config.toml && chmod 755 /workspace/benchmark/*.py /workspace/benchmark/*.sh /app/workdir/kv_scoring_lab/tools/*.sh \
 && mkdir -p /app/workdir/kv_scoring_lab/build /logs/agent && chown -R agent:agent /app/workdir /logs/agent \
 && chown -R root:root /workspace && chmod -R a-w /workspace \
    && chown -R root:root /app/workdir/kv_scoring_lab/.sforge \
    && chmod 555 /app/workdir/kv_scoring_lab/.sforge \
       /app/workdir/kv_scoring_lab/.sforge/prepare_submit.sh
USER agent
WORKDIR /app/workdir/kv_scoring_lab
ENTRYPOINT []
CMD ["bash"]

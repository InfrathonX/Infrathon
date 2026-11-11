ARG ON_DEMAND_SOFTWARE_IMAGE
FROM ${ON_DEMAND_SOFTWARE_IMAGE}
USER root
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    TRITON_CACHE_DIR=/app/workdir/minference_prefill_lab/build/triton-cache \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
# This build accepts only the measured clean software parent. Fail if a
# historical task tree is present; deleting it would not remove parent layers.
RUN test ! -e /opt/minference_prefill_judge \
    && test ! -e /workspace/benchmark \
    && id agent && test "$(id -u agent)" = 1000
RUN CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void python - <<'PY'
from importlib.metadata import version
import torch,triton,transformers
assert torch.__version__=='2.8.0+cu129' and torch.version.cuda=='12.9'
assert triton.__version__=='3.4.0' and transformers.__version__=='4.57.1'
assert version('flash_attn')=='2.8.3' and version('einops')=='0.8.1'
PY
COPY benchmark/ /workspace/benchmark/
COPY --chmod=0755 commands/sforge-submit commands/sforge-save-version /usr/local/bin/
COPY workdir/ /app/workdir/
COPY AGENT_TASK.md /app/workdir/minference_prefill_lab/AGENT_TASK.md
COPY benchmark/prepare_submit.sh /app/workdir/minference_prefill_lab/.sforge/prepare_submit.sh
COPY --chown=agent:agent codex-config.toml /home/agent/.codex/config.toml
RUN mkdir -p /workspace/benchmark/public-prompts \
      /workspace/benchmark/model \
      /app/workdir/minference_prefill_lab/build/triton-cache /logs/agent \
    && chown -R agent:agent /app/workdir /logs/agent \
    && find /workspace/benchmark -type d -exec chmod 0555 {} + \
    && find /workspace/benchmark -type f -exec chmod 0444 {} + \
    && chmod 0555 /workspace/benchmark/*.py /workspace/benchmark/*.sh \
       /app/workdir/minference_prefill_lab/tools/*.sh \
    && chmod 0600 /home/agent/.codex/config.toml \
    && chown -R root:root /app/workdir/minference_prefill_lab/.sforge \
    && chmod 0555 /app/workdir/minference_prefill_lab/.sforge \
       /app/workdir/minference_prefill_lab/.sforge/prepare_submit.sh
USER agent
WORKDIR /app/workdir/minference_prefill_lab
ENTRYPOINT []
CMD ["bash"]

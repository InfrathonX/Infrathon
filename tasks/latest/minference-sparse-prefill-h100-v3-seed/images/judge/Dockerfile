ARG ON_DEMAND_SOFTWARE_IMAGE
FROM ${ON_DEMAND_SOFTWARE_IMAGE}
USER root
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    MINFERENCE_PREFILL_WORKER_TIMEOUT=3600 MINFERENCE_PREFILL_JUDGE_TIMEOUT=18000
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
COPY --chmod=0400 --from=installed_sources / /opt/minference_prefill_judge/
RUN mkdir -p /workspace/benchmark/performance-prompts \
      /opt/minference_prefill_judge/external-assets/kernel-prompts \
      /opt/minference_prefill_judge/model/d10aef7999a2b5ba950ab3974312feeedbfe0b77 \
      /logs/verifier /app/workdir/minference_prefill_lab /home/agent/.cache \
    && find /opt/minference_prefill_judge -type d -exec chmod 0500 {} + \
    && chmod 0511 /opt/minference_prefill_judge /opt/minference_prefill_judge/runtime \
       /opt/minference_prefill_judge/scripts \
    && chmod 0500 /opt/minference_prefill_judge/candidate_worker.py \
    && chmod 0555 /workspace/benchmark /workspace/benchmark/performance-prompts \
    && chmod 0700 /logs/verifier \
    && chown -R agent:agent /home/agent/.cache
WORKDIR /app/workdir/minference_prefill_lab
USER root
ENTRYPOINT []

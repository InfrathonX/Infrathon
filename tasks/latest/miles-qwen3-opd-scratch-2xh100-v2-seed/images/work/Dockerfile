# Work image: Miles 78527b9 with on-policy distillation removed (the starter patch), the
# editable starter, public checks and data. Models, the converted initial student
# checkpoint and the agent runtime (Codex) are mounted read-only at launch, never baked in.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
USER root
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    WANDB_MODE=disabled RAY_USAGE_STATS_ENABLED=0 CUDA_DEVICE_MAX_CONNECTIONS=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility HOME=/home/agent
COPY init/ /app/workdir/miles_lab/repo/
COPY workdir/ /app/workdir/
COPY AGENT_TASK.md /app/workdir/miles_lab/AGENT_TASK.md
COPY INTERFACE.md /app/workdir/miles_lab/docs/INTERFACE.md
COPY public/ /opt/miles_opd_public/
COPY public-data/ /data/public/
COPY hook/opd_audit_hook.py hook/opd_audit_hook.pth hook/opd_provenance_hook.py hook/opd_provenance_hook.pth \
     /usr/local/lib/python3.12/dist-packages/
RUN mkdir -p /scratch /logs/agent /home/agent/.cache \
    && chown -R agent:agent /app/workdir /scratch /logs/agent /home/agent \
    && chown -R root:root /opt/miles_opd_public /data/public \
    && chmod -R a-w /opt/miles_opd_public /data/public \
    && test ! -e /data/public/train.jsonl \
    && test ! -e /root/miles \
    && test -z "$(find /app/workdir/miles_lab -type l)" \
    && python -c "import importlib.util as u; assert u.find_spec('miles') is None" \
    && ! grep -rqiE "opd|on_policy_distill|teacher_log" /app/workdir/miles_lab/repo/miles \
    && ! grep -rqiE "opcd|context.distill" /app/workdir/miles_lab/repo/miles \
    && ! command -v codex && ! command -v node
USER agent
WORKDIR /app/workdir/miles_lab
ENTRYPOINT []
CMD ["bash"]

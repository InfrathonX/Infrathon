# Work image: miles_diffusion@29fde6ee (no teacher path, no distillation loss), the editable starter, public checks
# and small public data. Models, teachers and the large public replay/probe recordings (/data/public/{probe,replay})
# are mounted read-only at launch; the agent runtime (Codex) is host-mounted, never baked in.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
USER root
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    WANDB_MODE=disabled RAY_USAGE_STATS_ENABLED=0 NVIDIA_DRIVER_CAPABILITIES=compute,utility HOME=/home/agent \
    OPD_PUBLIC_DATA=/data/public OPD_APP=/app/workdir/miles_lab
COPY init/ /app/workdir/miles_lab/repo/
COPY workdir/ /app/workdir/
COPY AGENT_TASK.md /app/workdir/miles_lab/AGENT_TASK.md
COPY INTERFACE.md /app/workdir/miles_lab/docs/INTERFACE.md
COPY public/ /opt/opd_public/
COPY md_pristine/ /opt/opd_public/md_pristine/
COPY public-data/ /data/public/
COPY hook/opd_audit_hook.py hook/opd_audit_hook.pth /usr/local/lib/python3.12/dist-packages/
RUN mkdir -p /scratch /logs/agent /home/agent/.cache /data/public/probe /data/public/replay \
    && chown -R agent:agent /app/workdir /scratch /logs/agent /home/agent \
    && chown -R root:root /opt/opd_public /data/public \
    && chmod -R a+rX,a-w /opt/opd_public /data/public \
    && test -z "$(find /app/workdir/miles_lab -type l)" \
    && test ! -e /opt/opd_public/judge/run_judge.py \
    && python -c "import importlib.util as u; assert u.find_spec('miles') is None" \
    && ! grep -rqiE "teacher_model|teacher-model|loss_type.*opd|distill_kl|teacher-key|opd_dump|diffusionopd" /app/workdir/miles_lab/repo \
    && ! command -v codex && ! command -v node
USER agent
WORKDIR /app/workdir/miles_lab
ENTRYPOINT []
CMD ["bash"]

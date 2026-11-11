# Work image: miles_diffusion@29fde6ee (no Dance transition, no seed-group RNG, no HPSv3) with the starter
# repo/task_entry.py and repo/engine_overlay/README.md, the public checks (same code as the Judge, run as the agent) and
# the in-engine audit hook (inert unless WAN_AUDIT_DIR is set). Models and the public data root (/data/public) are
# mounted read-only at launch; the agent runtime (Codex) is host-mounted, never baked in.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
USER root
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    WANDB_MODE=disabled RAY_USAGE_STATS_ENABLED=0 NVIDIA_DRIVER_CAPABILITIES=compute,utility HOME=/home/agent \
    WAN_PUBLIC_DATA=/data/public WAN_APP=/app/workdir/miles_lab
COPY init/ /app/workdir/miles_lab/repo/
COPY workdir/ /app/workdir/
COPY AGENT_TASK.md /app/workdir/miles_lab/AGENT_TASK.md
COPY INTERFACE.md /app/workdir/miles_lab/docs/INTERFACE.md
COPY public/ /opt/wan_public/
COPY hook/wan_audit_hook.py hook/wan_audit_hook.pth /usr/local/lib/python3.12/dist-packages/
RUN mkdir -p /scratch /logs/agent /home/agent/.cache /data/public \
    && chown -R agent:agent /app/workdir /scratch /logs/agent /home/agent \
    && chown -R root:root /opt/wan_public \
    && chmod -R a+rX,a-w /opt/wan_public \
    && test -z "$(find /app/workdir/miles_lab -type l)" \
    && python -c "import importlib.util as u; assert u.find_spec('miles') is None" \
    && ! grep -rqE "DanceSdeStepBackend|rollout_sde_noise_seed|class HPSv3|hpsv3_frame_mu" /app/workdir/miles_lab/repo \
    && ! command -v codex && ! command -v node
USER agent
WORKDIR /app/workdir/miles_lab
ENTRYPOINT []
CMD ["bash"]

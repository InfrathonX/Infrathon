# Judge image: same runtime; the Judge's private data, checks and the reference
# implementation (upstream Miles OPD, /opt/opd_judge/reference) live in a root-only
# directory. Candidate code is copied from the extracted submission and run as `agent`;
# the reference runs as `agent` from a temporary copy deleted before any candidate code runs.
# Public Judge: no private/ (hidden eval sets, probe batches); it may be published.
# images/judge/Dockerfile.private adds private/ on the host that builds it.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
USER root
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    WANDB_MODE=disabled RAY_USAGE_STATS_ENABLED=0 CUDA_DEVICE_MAX_CONNECTIONS=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility HOME=/root
COPY init/ /app/workdir/miles_lab/repo/
COPY workdir/ /app/workdir/
COPY judge/ /opt/opd_judge/
COPY public-data/ /data/public/
COPY hook/opd_audit_hook.py hook/opd_audit_hook.pth hook/opd_provenance_hook.py hook/opd_provenance_hook.pth \
     /usr/local/lib/python3.12/dist-packages/
RUN mkdir -p /scratch /logs/verifier /home/agent/.cache \
    && chown -R agent:agent /home/agent \
    && chown -R root:root /opt/opd_judge /data/public \
    && chmod -R go-rwx /opt/opd_judge \
    && chmod -R a-w /data/public \
    && chmod 700 /logs/verifier \
    && test -f /opt/opd_judge/reference/task_entry.py \
    && test -f /opt/opd_judge/reference/repo/miles/rollout/on_policy_distillation.py \
    && test -f /opt/opd_judge/reference/repo/miles/opd_task/step_dump.py \
    && test -z "$(find /opt/opd_judge/reference -type l)"
USER root
WORKDIR /app/workdir/miles_lab
ENTRYPOINT []
CMD ["bash"]

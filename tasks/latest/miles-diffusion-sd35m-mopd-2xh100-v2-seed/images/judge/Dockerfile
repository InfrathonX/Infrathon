# Judge image: same runtime; checks, the fp32 oracle and the replay server live in a root-only directory; the private
# data root is bind-mounted read-only at /judge_private (host copy go-rwx). Candidate code (extracted submission) is
# copied and run as `agent`; never imported.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
USER root
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    WANDB_MODE=disabled RAY_USAGE_STATS_ENABLED=0 NVIDIA_DRIVER_CAPABILITIES=compute,utility HOME=/root \
    PYTHONPATH=/opt/opd_judge/md_pristine
COPY judge/ /opt/opd_judge/
COPY md_pristine/ /opt/opd_judge/md_pristine/
COPY hook/opd_audit_hook.py hook/opd_audit_hook.pth /usr/local/lib/python3.12/dist-packages/
RUN mkdir -p /scratch /logs/verifier /home/agent/.cache \
    && chown -R agent:agent /home/agent \
    && mkdir -m 700 /judge_private \
    && chown -R root:root /opt/opd_judge \
    && chmod -R go-rwx /opt/opd_judge \
    && chmod 700 /logs/verifier
USER root
WORKDIR /app/workdir/miles_lab
ENTRYPOINT []
CMD ["bash"]

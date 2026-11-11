# Judge image: same runtime; the checks, the fp32 oracle helpers, the held-out quality eval (quality_eval.py,
# hpsv3_scorer.py, metric.json, eval_prompts/) and the audit hook live in a root-only directory; the private data root
# (hidden dumps, oracles, qualification) is bind-mounted read-only at /judge_private (host copy go-rwx). Candidate code
# (extracted submission) is copied and run as `agent` (uid 1001) via setpriv; never imported.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
USER root
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    WANDB_MODE=disabled RAY_USAGE_STATS_ENABLED=0 NVIDIA_DRIVER_CAPABILITIES=compute,utility HOME=/root
COPY judge/ /opt/wan_judge/
COPY hook/wan_audit_hook.py hook/wan_audit_hook.pth /usr/local/lib/python3.12/dist-packages/
RUN mkdir -p /scratch /logs/verifier /home/agent/.cache \
    && chown -R agent:agent /home/agent \
    && mkdir -m 700 /judge_private \
    && chown -R root:root /opt/wan_judge \
    && chmod -R go-rwx /opt/wan_judge \
    && chmod 700 /logs/verifier \
    && test "$(id -u agent)" = 1001
USER root
WORKDIR /app/workdir/miles_lab
ENTRYPOINT []
CMD ["bash"]

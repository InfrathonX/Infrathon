# Judge image: same runtime; the Judge's private data and checks live in a root-only
# directory. Candidate code is copied from the extracted submission and run as `agent`.
# Public Judge: no private/ (hidden eval sets, probe batches, games); it may be published.
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
COPY judge/ /opt/alfworld_judge/
COPY public-data/ /data/public/
COPY hook/ /usr/local/lib/python3.12/dist-packages/
RUN mkdir -p /scratch /logs/verifier /home/agent/.cache \
    && chown -R agent:agent /home/agent \
    && chown -R root:root /opt/alfworld_judge /data/public \
    && chmod -R go-rwx /opt/alfworld_judge \
    && chmod -R a-w /data/public \
    && chmod 700 /logs/verifier
USER root
WORKDIR /app/workdir/miles_lab
ENTRYPOINT []
CMD ["bash"]

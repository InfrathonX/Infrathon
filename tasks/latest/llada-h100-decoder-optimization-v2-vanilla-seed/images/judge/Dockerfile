ARG WORK_IMAGE=llada_decoder.work.llada_h100_decoder_optimization_v2:local
FROM ${WORK_IMAGE}

# Public Judge: everything but private_data/ (hidden cases, quality references,
# policies).  This image may be published; Dockerfile.private adds private_data/
# on the host that builds or pulls it (scripts/build_images.sh).

USER root

RUN rm -f /etc/sudoers.d/mlsysforge-agent \
    && sed -i '/^agent ALL=(ALL) NOPASSWD:ALL$/d' /etc/sudoers 2>/dev/null \
    && groupadd --gid 60001 mlsysforge-grader \
    && useradd --uid 60001 --gid 60001 --no-create-home \
        --home-dir /nonexistent --shell /usr/sbin/nologin mlsysforge-grader

ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV TOKENIZERS_PARALLELISM=false
ENV TRITON_CACHE_DIR=/tmp/triton-cache-agent

COPY run_judge.py /opt/decoder_judge/run_judge.py
COPY verifier/ /opt/decoder_judge/verifier/
COPY protected_hashes.txt /opt/decoder_judge/protected_hashes.txt

RUN chown -R root:root /opt/decoder_judge \
    && chmod 500 /opt/decoder_judge /opt/decoder_judge/verifier \
    && find /opt/decoder_judge -type f -exec chmod 400 {} + \
    && chmod 500 /opt/decoder_judge/run_judge.py \
    && mkdir -p /tmp/triton-cache-agent \
    && chown agent:agent /tmp/triton-cache-agent \
    && chown -R root:root /app/workdir/decoder_lab/runtime \
    && chmod 555 /app/workdir/decoder_lab/runtime \
    && find /app/workdir/decoder_lab/runtime -type f -exec chmod 444 {} +

WORKDIR /app/workdir/decoder_lab
ENTRYPOINT []

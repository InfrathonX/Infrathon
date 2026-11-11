FROM ubuntu:24.04@sha256:52df9b1ee71626e0088f7d400d5c6b5f7bb916f8f0c82b474289a4ece6cf3faf

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bubblewrap \
        build-essential \
        ca-certificates \
        curl \
        git \
        jq \
        nodejs \
        npm \
        procps \
        python3 \
        python3-aiohttp \
        python3-fastapi \
        python3-httpx \
        python3-uvicorn \
        python3-venv \
        tini \
        util-linux \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --system --uid 900 --no-create-home --shell /usr/sbin/nologin bridge \
    && existing_user="$(getent passwd 1000 | cut -d: -f1)" \
    && if [ -n "${existing_user}" ]; then \
         existing_group="$(id -gn "${existing_user}")"; \
         usermod --login candidate --home /home/candidate --move-home "${existing_user}"; \
         if [ "${existing_group}" != candidate ]; then \
           groupmod --new-name candidate "${existing_group}"; \
         fi; \
       else \
         groupadd --gid 1000 candidate; \
         useradd --create-home --uid 1000 --gid candidate --shell /bin/bash candidate; \
       fi \
    && usermod --groups candidate candidate \
    && install -d -o root -g root -m 0755 /app \
    && install -d -o candidate -g candidate -m 0755 /app/submission \
    && install -d -o candidate -g candidate -m 0755 /app/public_evals \
    && install -d -o root -g root -m 0755 /app/feedback

COPY container-entrypoint.sh /usr/local/bin/kv-task-entrypoint
COPY loopback_bridge.py /opt/kvbench/loopback_bridge.py
COPY kvbench.py /opt/kvbench/kvbench.py
COPY kvbench /usr/local/bin/kvbench
# Candidate-visible Direct/FIFO starting point.  This is deliberately separate
# from the author-only solution/ tree, which is outside this build context.
COPY --chown=candidate:candidate starter_submission/ /app/submission/

RUN chmod 0755 \
        /usr/local/bin/kv-task-entrypoint \
        /usr/local/bin/kvbench \
        /opt/kvbench/loopback_bridge.py \
        /opt/kvbench/kvbench.py \
        /app/submission/launch.sh \
    && chmod 0644 /app/submission/gateway.py \
    && chown root:root /app \
    && chmod 0755 /app \
    && chown -R candidate:candidate /app/submission

WORKDIR /app
ENTRYPOINT ["/usr/local/bin/kv-task-entrypoint"]

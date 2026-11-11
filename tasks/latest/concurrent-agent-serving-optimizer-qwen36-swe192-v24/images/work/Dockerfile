FROM ubuntu:24.04@sha256:52df9b1ee71626e0088f7d400d5c6b5f7bb916f8f0c82b474289a4ece6cf3faf

ARG DEBIAN_FRONTEND=noninteractive

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Etc/UTC

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        jq \
        less \
        lsof \
        netcat-openbsd \
        procps \
        python3 \
        python3-aiohttp \
        python3-httpx \
        ripgrep \
        sudo \
        tini \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

# This image is agent-free: the agent CLI (Codex, Claude Code, ...) arrives
# as a host-provided runtime that SForge bind-mounts read-only at
# /opt/agent-runtime when SFORGE_AGENT_RUNTIME_DIR is set.  Switching agent
# kind or version never rebuilds this image.

RUN existing_user="$(getent passwd 1000 | cut -d: -f1)" \
    && if [ -n "${existing_user}" ]; then \
         existing_group="$(id -gn "${existing_user}")"; \
         usermod --login agent --home /home/agent --move-home "${existing_user}"; \
         if [ "${existing_group}" != agent ]; then \
           groupmod --new-name agent "${existing_group}"; \
         fi; \
       else \
         groupadd --gid 1000 agent; \
         useradd --create-home --uid 1000 --gid agent --shell /bin/bash agent; \
       fi \
    && usermod --groups agent agent \
    && printf 'agent ALL=(ALL) NOPASSWD:ALL\n' >/etc/sudoers.d/mlsysforge-agent \
    && chmod 0440 /etc/sudoers.d/mlsysforge-agent \
    && install -d -o root -g root -m 0755 /app /workspace \
    && install -d -o agent -g agent -m 0755 \
        /app/submission \
        /app/public_evals \
        /home/agent/.codex \
        /logs/agent \
    && git config --system safe.directory '*'

# This context deliberately contains only agent-visible material.  The
# unprivileged kvbench client can ask the external trusted Judge origin to run
# the public workload, but no model socket, in-container benchmark controller,
# private workload, Judge implementation, or author reference gateway is
# copied here.
COPY --chown=agent:agent submission/ /app/submission/
COPY AGENT_TASK.md /app/AGENT_TASK.md
COPY RUNTIME_CONTRACT.json /app/RUNTIME_CONTRACT.json
COPY --chown=agent:agent RUN_BUDGET.json /app/RUN_BUDGET.json
COPY protocol_tests/ /workspace/protocol_tests/
COPY public_benchmark/ /opt/kvbench-public/
COPY --chown=agent:agent codex-config.toml /home/agent/.codex/config.toml

RUN install -o root -g root -m 0755 \
        /opt/kvbench-public/kvbench \
        /usr/local/bin/kvbench \
    && chmod 0755 \
        /app/submission/launch.sh \
        /workspace/protocol_tests/run_protocol_smoke.sh \
        /workspace/protocol_tests/protocol_smoke.py \
        /opt/kvbench-public/kvbench.py \
    && chmod 0644 \
        /app/submission/gateway.py \
        /app/submission/ARCHITECTURE.md \
        /app/AGENT_TASK.md \
        /app/RUN_BUDGET.json \
    && chmod 0444 \
        /app/RUNTIME_CONTRACT.json \
    && chmod -R a-w /workspace/protocol_tests \
    && chmod -R a-w /opt/kvbench-public \
    && chmod 0600 /home/agent/.codex/config.toml \
    && chown -R agent:agent \
        /app/submission \
        /app/public_evals \
        /logs/agent \
        /home/agent/.codex

USER agent
WORKDIR /app
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]

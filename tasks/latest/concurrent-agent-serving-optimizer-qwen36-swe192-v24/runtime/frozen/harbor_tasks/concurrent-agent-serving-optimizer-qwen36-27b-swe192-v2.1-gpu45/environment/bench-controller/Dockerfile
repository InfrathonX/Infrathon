FROM python:3.12.11-slim-bookworm@sha256:c00fc7b44d844b6da22861ec24af43968a5200eac4ec607b4725d585165d6b49

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir aiohttp==3.12.15 jinja2==3.1.6

COPY server.py /opt/bench-controller/server.py
COPY runner /opt/bench-controller/runner
COPY profiles /opt/bench-controller/profiles
COPY vendor/minisweagent /opt/bench-controller/vendor/minisweagent
COPY vendor/mini-swe-agent.LICENSE.md vendor/VENDORING.md /opt/bench-controller/vendor/
RUN chmod -R a-w /opt/bench-controller \
    && chmod 0555 /opt/bench-controller/server.py

EXPOSE 7000
ENV PYTHONPATH=/opt/bench-controller:/opt/bench-controller/vendor \
    MSWEA_SILENT_STARTUP=1
ENTRYPOINT ["python3", "/opt/bench-controller/server.py"]

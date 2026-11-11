FROM python:3.12.11-slim-bookworm@sha256:c00fc7b44d844b6da22861ec24af43968a5200eac4ec607b4725d585165d6b49

RUN apt-get update \
    && apt-get install -y --no-install-recommends bubblewrap git util-linux \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir \
        certifi==2026.6.17 \
        charset-normalizer==3.4.9 \
        docker==7.1.0 \
        idna==3.18 \
        requests==2.34.2 \
        urllib3==2.7.0 \
    && chmod 4755 /usr/bin/bwrap \
    && install -d -o root -g root -m 0700 /run/kvbench-sandbox /run/kvbench-workspaces

COPY worker.py /opt/sandbox-worker/worker.py
RUN chmod 0555 /opt/sandbox-worker/worker.py

ENTRYPOINT ["python3", "/opt/sandbox-worker/worker.py"]

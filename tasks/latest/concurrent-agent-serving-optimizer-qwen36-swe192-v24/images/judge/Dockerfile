ARG WORK_IMAGE=kvbench_qwen36.work.concurrent_agent_serving_optimizer_qwen36_swe192_v24:local
FROM ${WORK_IMAGE}

USER root

# Only the trusted Judge image receives a Docker client.  The host socket and
# frozen runtime files are injected per Judge container by the runtime adapter;
# they are never baked into either image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        docker-compose-v2 \
        docker.io \
        util-linux \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /etc/sudoers.d/mlsysforge-agent

COPY judge/run_judge.py /opt/kvbench-mlsysforge/run_judge.py
COPY judge/run_public.py /opt/kvbench-mlsysforge/run_public.py
COPY scripts/run_public_candidate.py /opt/kvbench-mlsysforge/run_public_candidate.py
RUN chmod 0755 \
        /opt/kvbench-mlsysforge/run_judge.py \
        /opt/kvbench-mlsysforge/run_public.py \
        /opt/kvbench-mlsysforge/run_public_candidate.py \
    && install -d -o root -g root -m 0700 /logs/verifier

WORKDIR /app
ENTRYPOINT []
CMD ["bash"]

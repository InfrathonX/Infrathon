ARG WORK_IMAGE=flash_kmeans.work.flash_kmeans_paper_h100_kernel_lab_v7_scratch:local
FROM ${WORK_IMAGE}

# Public Judge: everything but hidden_cases.py (the hidden benchmark and
# deployment cases).  This image may be published; Dockerfile.private adds it on
# the host that builds or pulls it (scripts/build_images.sh).

USER root

# The work image needs sudo for agent installation. The candidate evaluator
# permanently drops to this account and must not retain a path back to root.
RUN rm -f /etc/sudoers.d/mlsysforge-agent \
    && sed -i '/^agent ALL=(ALL) NOPASSWD:ALL$/d' /etc/sudoers 2>/dev/null || true

ENV FLASH_KMEANS_VERIFY_SUITE=full
ENV FLASH_KMEANS_VERIFY_WARMUP=3
ENV FLASH_KMEANS_VERIFY_ITERS=5
ENV FLASH_KMEANS_VERIFY_REPEATS=5
ENV FLASH_KMEANS_VERIFY_TIMEOUT=3300
ENV FLASH_KMEANS_DEPLOYMENT_CASE_TIMEOUT=180
ENV PYTHONDONTWRITEBYTECODE=1
ENV TRITON_CACHE_DIR=/tmp/triton-cache-agent

COPY verify_flash_kmeans.py /opt/flash_kmeans_judge/verify_flash_kmeans.py
COPY run_judge.py /opt/flash_kmeans_judge/run_judge.py
COPY protected_hashes.txt /opt/flash_kmeans_judge/protected_hashes.txt
COPY benchmark/ /opt/flash_kmeans_benchmark/

# All benchmark, reference, timing, memory, case, and admission code enters
# only through the Judge build context. Nothing is inherited from the Work
# image. Benchmark code remains root-only. Fresh-process children import the
# harness while privileged, then irreversibly drop to agent before importing
# any candidate source.
RUN chown -R root:root \
      /opt/flash_kmeans_judge \
      /opt/flash_kmeans_benchmark \
    && chmod 500 /opt/flash_kmeans_judge \
    && chmod 400 /opt/flash_kmeans_judge/* \
    && chmod 500 /opt/flash_kmeans_benchmark \
    && find /opt/flash_kmeans_benchmark -type f -exec chmod 400 {} + \
    && mkdir -p /tmp/triton-cache-agent \
    && chown agent:agent /tmp/triton-cache-agent

WORKDIR /app/workdir/flash_kmeans_lab
ENTRYPOINT []

FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

# Public Judge: everything but the private holdout.  This image may be published;
# Dockerfile.private adds the Judge-only inputs on the host that builds or
# pulls it (scripts/build_images.sh).

ENV CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LMCACHE_KV_CANDIDATE_TIMEOUT=3600 \
    LMCACHE_KV_TOTAL_TIMEOUT=4200 \
    LMCACHE_KV_JUDGE_TIMEOUT=4500

RUN useradd -m -s /bin/bash agent \
    && python -c "import torch; assert torch.__version__.split('+')[0] == '2.8.0'; assert torch.version.cuda == '12.9'" \
    && /usr/local/cuda/bin/nvcc --version | grep -q "release 12.9"

COPY --from=task_runtime kv_transfer_runtime.py /opt/lmcache_kv_transfer_worker/kv_transfer_runtime.py
COPY --from=task_runtime transfer_abi.h /opt/lmcache_kv_transfer_worker/transfer_abi.h
COPY run_judge.py verify_kv_transfer.py candidate_worker.py source_admission.py \
     protected_hashes.txt public_cases.json \
     /opt/lmcache_kv_transfer_judge/

RUN chown -R root:root /opt/lmcache_kv_transfer_judge /opt/lmcache_kv_transfer_worker \
    && chmod 0511 /opt/lmcache_kv_transfer_judge \
    && chmod 0400 /opt/lmcache_kv_transfer_judge/run_judge.py \
       /opt/lmcache_kv_transfer_judge/verify_kv_transfer.py \
       /opt/lmcache_kv_transfer_judge/source_admission.py \
       /opt/lmcache_kv_transfer_judge/protected_hashes.txt \
    && chmod 0444 /opt/lmcache_kv_transfer_judge/candidate_worker.py \
       /opt/lmcache_kv_transfer_judge/public_cases.json \
    && chmod 0555 /opt/lmcache_kv_transfer_worker \
    && chmod 0444 /opt/lmcache_kv_transfer_worker/kv_transfer_runtime.py \
       /opt/lmcache_kv_transfer_worker/transfer_abi.h \
    && mkdir -p /app/workdir/kv_transfer_lab /logs/verifier /home/agent/.cache \
    && chown -R agent:agent /home/agent/.cache \
    && chmod 0700 /logs/verifier

WORKDIR /app/workdir/kv_transfer_lab
USER root
ENTRYPOINT []

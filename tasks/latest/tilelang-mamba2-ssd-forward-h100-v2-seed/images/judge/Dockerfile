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
    TILELANG_SSD_VERIFY_TRIALS=3 \
    TILELANG_SSD_VERIFY_WARMUPS=1 \
    TILELANG_SSD_CANDIDATE_TIMEOUT=4500 \
    TILELANG_SSD_TOTAL_TIMEOUT=4900 \
    TILELANG_SSD_JUDGE_TIMEOUT=5100

RUN useradd -m -s /bin/bash agent \
    && pip install --no-cache-dir tilelang==0.1.13 \
    && python -c "import torch, triton, tilelang; assert torch.__version__.split('+')[0] == '2.8.0'; assert torch.version.cuda == '12.9'; assert triton.__version__ == '3.4.0'; assert tilelang.__version__ == '0.1.13'"

COPY --from=task_runtime mamba2_ssd_runtime.py /opt/tilelang_mamba2_ssd_worker/mamba2_ssd_runtime.py
COPY run_judge.py verify_ssd.py candidate_worker.py source_admission.py protected_hashes.txt /opt/tilelang_mamba2_ssd_judge/

RUN chown -R root:root /opt/tilelang_mamba2_ssd_judge /opt/tilelang_mamba2_ssd_worker \
    && chmod 0511 /opt/tilelang_mamba2_ssd_judge \
    && chmod 0400 /opt/tilelang_mamba2_ssd_judge/run_judge.py \
       /opt/tilelang_mamba2_ssd_judge/verify_ssd.py \
       /opt/tilelang_mamba2_ssd_judge/source_admission.py \
       /opt/tilelang_mamba2_ssd_judge/protected_hashes.txt \
    && chmod 0444 /opt/tilelang_mamba2_ssd_judge/candidate_worker.py \
    && chmod 0555 /opt/tilelang_mamba2_ssd_worker \
    && chmod 0444 /opt/tilelang_mamba2_ssd_worker/mamba2_ssd_runtime.py \
    && mkdir -p /app/workdir/mamba2_ssd_lab /logs/verifier /home/agent/.cache \
    && chown -R agent:agent /home/agent/.cache \
    && chmod 0700 /logs/verifier

WORKDIR /app/workdir/mamba2_ssd_lab
USER root
ENTRYPOINT []

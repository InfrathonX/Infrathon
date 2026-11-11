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
    FLUX_MOEAG_VERIFY_TRIALS=7 \
    FLUX_MOEAG_VERIFY_WARMUPS=1 \
    FLUX_MOEAG_CANDIDATE_TIMEOUT=5400 \
    FLUX_MOEAG_TOTAL_TIMEOUT=5700 \
    FLUX_MOEAG_JUDGE_TIMEOUT=6000

RUN useradd -m -s /bin/bash agent \
    && python -c \
      "import torch, triton; assert torch.__version__.split('+')[0] == '2.8.0'; assert torch.version.cuda == '12.9'; assert triton.__version__ == '3.4.0'"

COPY --from=task_runtime moe_ag_runtime.py /opt/flux_moe_ag_worker/moe_ag_runtime.py
COPY run_judge.py verify_moe_ag.py candidate_worker.py source_admission.py protected_hashes.txt /opt/flux_moe_ag_judge/

RUN chown -R root:root /opt/flux_moe_ag_judge /opt/flux_moe_ag_worker \
    && chmod 0511 /opt/flux_moe_ag_judge \
    && chmod 0400 /opt/flux_moe_ag_judge/run_judge.py \
       /opt/flux_moe_ag_judge/verify_moe_ag.py \
       /opt/flux_moe_ag_judge/source_admission.py \
       /opt/flux_moe_ag_judge/protected_hashes.txt \
    && chmod 0444 /opt/flux_moe_ag_judge/candidate_worker.py \
    && chmod 0555 /opt/flux_moe_ag_worker \
    && chmod 0444 /opt/flux_moe_ag_worker/moe_ag_runtime.py \
    && mkdir -p /app/workdir/moe_ag_lab /logs/verifier /home/agent/.cache \
    && chown -R agent:agent /home/agent/.cache \
    && chmod 0700 /logs/verifier

WORKDIR /app/workdir/moe_ag_lab
USER root
ENTRYPOINT []

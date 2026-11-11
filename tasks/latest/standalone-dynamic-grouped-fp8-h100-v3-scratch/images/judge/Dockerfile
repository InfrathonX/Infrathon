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
    STANDALONE_GROUPED_FP8_VERIFY_TRIALS=5 \
    STANDALONE_GROUPED_FP8_VERIFY_WARMUPS=1 \
    STANDALONE_GROUPED_FP8_VERIFY_ROUNDS=3 \
    STANDALONE_GROUPED_FP8_CANDIDATE_TIMEOUT=5400 \
    STANDALONE_GROUPED_FP8_SANITIZER_TIMEOUT=780 \
    STANDALONE_GROUPED_FP8_TOTAL_TIMEOUT=6300 \
    STANDALONE_GROUPED_FP8_JUDGE_TIMEOUT=6400

RUN useradd -m -s /bin/bash agent \
    && python -c \
      "import torch, triton; assert torch.__version__.split('+')[0] == '2.8.0'; assert torch.version.cuda == '12.9'; assert triton.__version__ == '3.4.0'" \
    && test -x /usr/local/cuda/compute-sanitizer/compute-sanitizer

COPY --from=task_runtime grouped_fp8_runtime.py /opt/standalone_grouped_fp8_worker/grouped_fp8_runtime.py
COPY run_judge.py verify_grouped.py candidate_worker.py source_admission.py protected_hashes.txt /opt/standalone_grouped_fp8_judge/

RUN chown -R root:root /opt/standalone_grouped_fp8_judge /opt/standalone_grouped_fp8_worker \
    && chmod 0511 /opt/standalone_grouped_fp8_judge \
    && chmod 0400 /opt/standalone_grouped_fp8_judge/run_judge.py \
       /opt/standalone_grouped_fp8_judge/verify_grouped.py \
       /opt/standalone_grouped_fp8_judge/source_admission.py \
       /opt/standalone_grouped_fp8_judge/protected_hashes.txt \
    && chmod 0444 /opt/standalone_grouped_fp8_judge/candidate_worker.py \
    && chmod 0555 /opt/standalone_grouped_fp8_worker \
    && chmod 0444 /opt/standalone_grouped_fp8_worker/grouped_fp8_runtime.py \
    && mkdir -p /app/workdir/grouped_fp8_lab /logs/verifier /home/agent/.cache \
    && chown -R agent:agent /home/agent/.cache \
    && chmod 0700 /logs/verifier

WORKDIR /app/workdir/grouped_fp8_lab
USER root
ENTRYPOINT []

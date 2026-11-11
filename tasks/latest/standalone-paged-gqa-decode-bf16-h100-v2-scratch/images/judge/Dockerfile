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
    STANDALONE_PAGED_GQA_VERIFY_TRIALS=3 \
    STANDALONE_PAGED_GQA_VERIFY_WARMUPS=1 \
    STANDALONE_PAGED_GQA_CANDIDATE_TIMEOUT=7200 \
    STANDALONE_PAGED_GQA_SANITIZER_TIMEOUT=780 \
    STANDALONE_PAGED_GQA_TOTAL_TIMEOUT=8100 \
    STANDALONE_PAGED_GQA_JUDGE_TIMEOUT=8200

RUN useradd -m -s /bin/bash agent \
    && python -c \
      "import torch, triton; assert torch.__version__.split('+')[0] == '2.8.0'; assert torch.version.cuda == '12.9'; assert triton.__version__ == '3.4.0'" \
    && test -x /usr/local/cuda/compute-sanitizer/compute-sanitizer

COPY --from=task_runtime paged_gqa_decode_runtime.py /opt/standalone_paged_gqa_decode_worker/paged_gqa_decode_runtime.py
COPY run_judge.py verify_paged_gqa.py candidate_worker.py source_admission.py protected_hashes.txt /opt/standalone_paged_gqa_decode_judge/

RUN chown -R root:root /opt/standalone_paged_gqa_decode_judge /opt/standalone_paged_gqa_decode_worker \
    && chmod 0511 /opt/standalone_paged_gqa_decode_judge \
    && chmod 0400 /opt/standalone_paged_gqa_decode_judge/run_judge.py \
       /opt/standalone_paged_gqa_decode_judge/verify_paged_gqa.py \
       /opt/standalone_paged_gqa_decode_judge/source_admission.py \
       /opt/standalone_paged_gqa_decode_judge/protected_hashes.txt \
    && chmod 0444 /opt/standalone_paged_gqa_decode_judge/candidate_worker.py \
    && chmod 0555 /opt/standalone_paged_gqa_decode_worker \
    && chmod 0444 /opt/standalone_paged_gqa_decode_worker/paged_gqa_decode_runtime.py \
    && mkdir -p /app/workdir/paged_gqa_decode_lab /logs/verifier /home/agent/.cache \
    && chown -R agent:agent /home/agent/.cache \
    && chmod 0700 /logs/verifier

WORKDIR /app/workdir/paged_gqa_decode_lab
USER root
ENTRYPOINT []

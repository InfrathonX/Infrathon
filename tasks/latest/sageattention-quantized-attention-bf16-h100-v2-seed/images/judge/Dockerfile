FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

ENV CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    SAGE_QATTN_VERIFY_TRIALS=3 \
    SAGE_QATTN_VERIFY_WARMUPS=1 \
    SAGE_QATTN_CANDIDATE_TIMEOUT=5400 \
    SAGE_QATTN_TOTAL_TIMEOUT=5700 \
    SAGE_QATTN_JUDGE_TIMEOUT=5800 \
    SAGE_QATTN_FIXTURE_DIR=/opt/sageattention_quant_attention_judge/fixtures

RUN useradd -m -s /bin/bash agent \
    && python -c \
      "import torch, triton; assert torch.__version__.split('+')[0] == '2.8.0'; assert torch.version.cuda == '12.9'; assert triton.__version__ == '3.4.0'"

COPY --from=task_runtime quant_attention_runtime.py /opt/sageattention_quant_attention_worker/quant_attention_runtime.py
COPY run_judge.py verify_quant_attention.py candidate_worker.py source_admission.py protected_hashes.txt /opt/sageattention_quant_attention_judge/
# The e8 edge input is captured per deployment from fixtures/recipe/RECIPE.json.
# RECIPE.json travels next to the trusted runtime; the capture manifest travels
# next to the capture, and the runtime binds one to the other before every run.
COPY --from=fixtures recipe/RECIPE.json /opt/sageattention_quant_attention_worker/RECIPE.json
COPY --from=fixtures e8_wan13b_p0_b15_s00_cond_8192.pt /opt/sageattention_quant_attention_judge/fixtures/e8_wan13b_p0_b15_s00_cond_8192.pt
COPY --from=fixtures e8_wan13b_p0_b15_s00_cond_8192.manifest.json /opt/sageattention_quant_attention_judge/fixtures/e8_wan13b_p0_b15_s00_cond_8192.manifest.json
# Real-model layers: every recipe travels next to the trusted runtime, which binds
# each capture (mounted read-only at run time, see scripts/sforge_judge_entry.py) to it.
COPY --from=fixtures recipe/real_layers/RECIPE_*.json /opt/sageattention_quant_attention_worker/real_layers/

RUN chown -R root:root /opt/sageattention_quant_attention_judge /opt/sageattention_quant_attention_worker \
    && chmod 0511 /opt/sageattention_quant_attention_judge \
    && chmod 0400 /opt/sageattention_quant_attention_judge/run_judge.py \
       /opt/sageattention_quant_attention_judge/verify_quant_attention.py \
       /opt/sageattention_quant_attention_judge/source_admission.py \
       /opt/sageattention_quant_attention_judge/protected_hashes.txt \
    && chmod 0444 /opt/sageattention_quant_attention_judge/candidate_worker.py \
       /opt/sageattention_quant_attention_judge/fixtures/e8_wan13b_p0_b15_s00_cond_8192.pt \
       /opt/sageattention_quant_attention_judge/fixtures/e8_wan13b_p0_b15_s00_cond_8192.manifest.json \
    && chmod -R a-w,a+rX /opt/sageattention_quant_attention_worker/real_layers \
    && chmod 0555 /opt/sageattention_quant_attention_worker \
    && chmod 0444 /opt/sageattention_quant_attention_worker/quant_attention_runtime.py \
       /opt/sageattention_quant_attention_worker/RECIPE.json \
    && mkdir -p /app/workdir/quant_attention_lab /logs/verifier /home/agent/.cache \
    && chown -R agent:agent /home/agent/.cache \
    && chmod 0700 /logs/verifier
# This is the publishable public Judge: images/judge/Dockerfile.private adds the final
# holdout on the host, and the scored real layers are mounted at run time.

WORKDIR /app/workdir/quant_attention_lab
USER root
ENTRYPOINT []

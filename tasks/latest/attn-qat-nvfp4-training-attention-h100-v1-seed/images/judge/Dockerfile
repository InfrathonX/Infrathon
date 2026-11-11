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
    ATTN_QAT_VERIFY_TRIALS=3 \
    ATTN_QAT_VERIFY_WARMUPS=1 \
    ATTN_QAT_CANDIDATE_TIMEOUT=3600 \
    ATTN_QAT_TOTAL_TIMEOUT=3900 \
    ATTN_QAT_JUDGE_TIMEOUT=4000 \
    ATTN_QAT_FIXTURE_DIR=/opt/attn_qat_judge/fixtures

RUN useradd -m -s /bin/bash agent \
    && python - <<'PY'
import importlib.util
import torch
import triton

assert torch.__version__.split("+")[0] == "2.8.0", torch.__version__
assert torch.version.cuda == "12.9", torch.version.cuda
assert triton.__version__ == "3.4.0", triton.__version__
for absent in (
    "fastvideo",
    "fastvideo_kernel",
    "flash_attn",
    "flash_attn_interface",
    "sageattention",
    "flashinfer",
    "xformers",
):
    assert importlib.util.find_spec(absent) is None, absent
PY

COPY --from=task_runtime attn_qat_runtime.py /opt/attn_qat_worker/attn_qat_runtime.py
COPY run_judge.py verify_attn_qat.py candidate_worker.py source_admission.py protected_hashes.txt /opt/attn_qat_judge/
# The e7 capture is this deployment's own: the recipe, not a byte digest, is its
# identity.  The manifest travels next to the capture and RECIPE.json next to the
# runtime that checks it.
COPY --from=fixtures e7_wan13b_grad_b29_t1_s8192.pt /opt/attn_qat_judge/fixtures/e7_wan13b_grad_b29_t1_s8192.pt
COPY --from=fixtures e7_wan13b_grad_b29_t1_s8192.manifest.json /opt/attn_qat_judge/fixtures/e7_wan13b_grad_b29_t1_s8192.manifest.json
COPY --from=recipe RECIPE.json /opt/attn_qat_worker/RECIPE.json
COPY --from=recipe real_layers/REAL_LAYERS.json /opt/attn_qat_worker/REAL_LAYERS.json
COPY --from=recipe real_layers/REAL_LAYERS_V2.json /opt/attn_qat_worker/REAL_LAYERS_V2.json

RUN chown -R root:root /opt/attn_qat_judge /opt/attn_qat_worker \
    && chmod 0511 /opt/attn_qat_judge \
    && chmod 0400 /opt/attn_qat_judge/run_judge.py \
       /opt/attn_qat_judge/verify_attn_qat.py \
       /opt/attn_qat_judge/source_admission.py \
       /opt/attn_qat_judge/protected_hashes.txt \
    && chmod 0444 /opt/attn_qat_judge/candidate_worker.py \
       /opt/attn_qat_judge/fixtures/e7_wan13b_grad_b29_t1_s8192.pt \
       /opt/attn_qat_judge/fixtures/e7_wan13b_grad_b29_t1_s8192.manifest.json \
    && chmod 0555 /opt/attn_qat_worker \
    && chmod 0444 /opt/attn_qat_worker/attn_qat_runtime.py \
       /opt/attn_qat_worker/RECIPE.json \
       /opt/attn_qat_worker/REAL_LAYERS.json \
       /opt/attn_qat_worker/REAL_LAYERS_V2.json \
    && mkdir -p /app/workdir/attn_qat_lab /logs/verifier /home/agent/.cache \
    && chown -R agent:agent /home/agent/.cache \
    && chmod 0700 /logs/verifier

WORKDIR /app/workdir/attn_qat_lab
USER root
ENTRYPOINT []

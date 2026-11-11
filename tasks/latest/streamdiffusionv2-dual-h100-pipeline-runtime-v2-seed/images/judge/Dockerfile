FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel@sha256:0cf3402e946b7c384ba943ee05c90b4c5a4a05227923921f2b0918c011cfaf56

# ---- shared trusted prefix (byte-identical between both Dockerfiles up to
# ---- the divergence marker so BuildKit shares the heavy layers) -------------
ARG DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    STREAMDIFFUSIONV2_ROOT=/opt/sdv2_weights \
    SDV2_ASSETS_ROOT=/opt/sdv2_weights \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl ffmpeg git jq ninja-build procps ripgrep util-linux xz-utils \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -s /bin/bash agent \
    && git config --system safe.directory '*'

# The official base currently defaults to CPython 3.11, whereas the pinned
# prebuilt FlashAttention wheel is deliberately cp310. Keep the base/CUDA
# libraries, switch the active conda interpreter, and reinstall the exact
# upstream Torch trio for Python 3.10.
RUN conda install -y python=3.10 pip \
    && conda clean -afy \
    && pip install --no-cache-dir --root-user-action=ignore \
      --index-url https://download.pytorch.org/whl/cu124 \
      torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

RUN pip install --no-cache-dir --root-user-action=ignore \
      accelerate==1.10.1 \
      av==13.1.0 \
      diffusers==0.35.1 \
      einops==0.8.1 \
      ftfy==6.3.1 \
      imageio==2.37.0 \
      imageio-ffmpeg==0.6.0 \
      lpips==0.1.4 \
      numpy==1.24.4 \
      omegaconf==2.3.0 \
      Pillow==10.1.0 \
      pydantic==2.10.6 \
      regex==2024.11.6 \
      scikit-image==0.25.2 \
      sentencepiece==0.2.0 \
      transformers==4.54.0

COPY --from=flash_wheel flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl /tmp/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
RUN pip install --no-cache-dir --root-user-action=ignore /tmp/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
    && rm -f /tmp/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

COPY --from=weights . /opt/sdv2_weights/
COPY --from=upstream_smoke . /opt/sdv2_upstream/

RUN cd /opt/sdv2_weights \
    && sha256sum --check --strict checkpoint_manifest.sha256 \
    && python - <<'PY'
import importlib.util
from importlib.metadata import version
import torch
assert torch.__version__.startswith("2.6.0"), torch.__version__
assert version("flash_attn") == "2.7.4.post1", version("flash_attn")
assert importlib.util.find_spec("flash_attn") is not None
assert importlib.util.find_spec("flash_attn_interface") is None, "FA3 must be absent"
PY
# ---- divergence marker: judge-only layers below -----------------------------
# Public Judge: no private/ (final cases, hidden embeddings, clips, goldens); it
# may be published.  Dockerfile.private adds private/ on the host that builds it.

# SDV2_WORKER_TIMEOUT bounds one measured pass; every evaluation runs two of
# them (the in-run seed reference, then the candidate), so the controller
# budget is a little over twice the per-pass budget.
ENV SDV2_WORKER_TIMEOUT=1800 \
    SDV2_TOTAL_TIMEOUT=4200

COPY --from=task_runtime . /opt/sdv2_judge/worker_runtime/
COPY --from=seed_runtime runtime_v001/ /opt/sdv2_judge/seed_runtime/runtime_v001/
# protected_hashes.txt binds the per-hardware goldens, so it comes with them in
# the holdout layer (Dockerfile.private).
COPY run_judge.py verify_sdv2_pipeline.py candidate_worker.py source_admission.py /opt/sdv2_judge/
COPY --from=assets embeddings/public/ /opt/sdv2_judge/fixtures/embeddings/public/

RUN chown -R root:root /opt/sdv2_judge \
    && chmod 0511 /opt/sdv2_judge \
    && chmod -R a-w /opt/sdv2_judge/fixtures \
    && chmod 0400 /opt/sdv2_judge/run_judge.py /opt/sdv2_judge/verify_sdv2_pipeline.py \
       /opt/sdv2_judge/source_admission.py \
    && chmod 0444 /opt/sdv2_judge/candidate_worker.py \
    && chmod 0555 /opt/sdv2_judge/worker_runtime \
    && chmod -R a-w /opt/sdv2_judge/worker_runtime \
    && chmod -R a-w /opt/sdv2_judge/seed_runtime \
    && chmod -R a+rX /opt/sdv2_judge/seed_runtime \
    && mkdir -p /app/workdir/sdv2_pipeline_lab /logs/verifier \
    && chmod 0700 /logs/verifier

WORKDIR /app
USER root
ENTRYPOINT []

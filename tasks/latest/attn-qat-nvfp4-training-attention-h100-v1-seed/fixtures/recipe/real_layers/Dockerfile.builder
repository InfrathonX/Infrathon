FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Pinned dependency set of the real-layer captures.
RUN pip install --no-cache-dir --root-user-action=ignore \
      accelerate==1.14.0 \
      diffusers==0.40.0 \
      ftfy==6.3.1 \
      huggingface_hub==1.33.0 \
      numpy==1.26.4 \
      protobuf==6.32.1 \
      pyarrow==21.0.0 \
      regex==2026.9.10 \
      safetensors==0.8.0 \
      sentencepiece==0.2.1 \
      tokenizers==0.22.2 \
      transformers==5.15.1 \
    && python -c "import torch, diffusers, transformers; assert torch.__version__.split('+')[0] == '2.8.0'"

COPY capture_real_layers.py capture_real_layers_v2.py /opt/real_layers/

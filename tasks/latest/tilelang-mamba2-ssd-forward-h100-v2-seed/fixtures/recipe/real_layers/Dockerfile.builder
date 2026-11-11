FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Pinned dependency set of the real-layer captures.
RUN pip install --no-cache-dir --root-user-action=ignore \
      accelerate==1.14.0 \
      huggingface_hub==1.33.0 \
      numpy==1.26.4 \
      pyarrow==21.0.0 \
      safetensors==0.8.0 \
      tokenizers==0.22.2 \
      transformers==5.15.1 \
    && python -c "import torch, transformers; assert torch.__version__.split('+')[0] == '2.8.0'; assert transformers.__version__ == '5.15.1'"

COPY capture_real_layers.py /opt/real_layers/capture_real_layers.py

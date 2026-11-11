# Clean MInference v3 software parent, rebuildable from this file alone: the official
# PyTorch 2.8.0 / CUDA 12.9 base plus the nine pinned distributions the task runs on.
ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel@sha256:cf5aa3f7045a68c10d80f546746591c5ccae6a33729e5e32625ff76bd2c036fe
FROM ${BASE_IMAGE}
USER root
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential ca-certificates curl git jq ninja-build procps ripgrep xz-utils && rm -rf /var/lib/apt/lists/* && useradd -m -s /bin/bash agent && test "$(id -u agent)" = 1000
ARG FLASH_ATTN_WHEEL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
RUN pip install --no-cache-dir --no-deps \
      transformers==4.57.1 tokenizers==0.22.2 huggingface-hub==0.36.2 hf-xet==1.6.0 safetensors==0.6.2 \
      regex==2026.9.10 accelerate==1.10.1 einops==0.8.1 "${FLASH_ATTN_WHEEL}"
RUN python3 -I - <<'PY'
import importlib.metadata as m,importlib.util,pathlib,torch,triton,transformers,einops
from flash_attn.bert_padding import pad_input, unpad_input
from transformers.models.llama.modeling_llama import LlamaForCausalLM
assert torch.__version__ == '2.8.0+cu129' and torch.version.cuda == '12.9' and triton.__version__=='3.4.0'
expected={'einops': '0.8.1', 'flash-attn': '2.8.3', 'tokenizers': '0.22.2', 'hf-xet': '1.6.0', 'huggingface-hub': '0.36.2', 'safetensors': '0.6.2', 'regex': '2026.9.10', 'accelerate': '1.10.1', 'transformers': '4.57.1'}
assert all(m.version(n)==v for n,v in expected.items())
assert not any(pathlib.Path(p).exists() for p in ('/opt/minference_prefill_judge','/workspace/benchmark','/model','/app/workdir/kv_eviction_lab'))
assert all(importlib.util.find_spec(n) is None for n in ('minference','vllm','sglang','flashinfer','xformers'))
PY
USER root
ENTRYPOINT []

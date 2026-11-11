# Host-only paper-reference image: FlashAttention-3 (hopper) built from the
# task's pinned FlashAttention commit 0251105a2fb19d2957484b7f023cd8c115286ced.
#
# This image is NOT the task's Work or Judge image.  It is built only by
# oracle/paper_reference/run_fa3_reference.sh, carries its own tag, and exists
# so a host operator can time upstream FA3 with the Judge's own runtime.
#
# Only the variants this ABI reaches are compiled (BF16, hdim 128, SM90,
# forward, paged KV, split, pack-GQA, varlen/seqused, causal), which keeps the
# build to tens of minutes.  Build context: a FULL flash-attention checkout at
# the pinned commit with csrc/cutlass initialised (see README.md); the
# harness files come from the named build context `harness` (this directory).
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    MAX_JOBS=8 \
    NVCC_THREADS=1 \
    FLASH_ATTN_CUDA_ARCHS=90 \
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
    FLASH_ATTENTION_DISABLE_BACKWARD=TRUE \
    FLASH_ATTENTION_DISABLE_FP16=TRUE \
    FLASH_ATTENTION_DISABLE_FP8=TRUE \
    FLASH_ATTENTION_DISABLE_HDIM64=TRUE \
    FLASH_ATTENTION_DISABLE_HDIM96=TRUE \
    FLASH_ATTENTION_DISABLE_HDIM192=TRUE \
    FLASH_ATTENTION_DISABLE_HDIM256=TRUE \
    FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE \
    FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE \
    FLASH_ATTENTION_DISABLE_SM80=TRUE \
    FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE \
    FLASH_ATTENTION_DISABLE_LOCAL=TRUE \
    FLASH_ATTENTION_DISABLE_APPENDKV=TRUE

RUN apt-get update && apt-get install -y --no-install-recommends git ninja-build \
    && rm -rf /var/lib/apt/lists/* && pip install --no-cache-dir einops packaging ninja
COPY . /opt/flash-attention
RUN cd /opt/flash-attention/hopper && pip install --no-build-isolation --no-cache-dir . \
    && python -c "import flash_attn_interface, flash_attn_3._C; print('fa3 ok')"
# Fail the build if varlen / paged KV / splits / PackGQA got compiled out
# (FA3 3.0.0b1 wheels built with FLASHATTENTION_DISABLE_VARLEN/PAGEDKV cannot
# run this ABI at all).
COPY --from=harness fa3_preflight.py /opt/fa3_preflight.py
RUN python -I /opt/fa3_preflight.py
WORKDIR /opt/harness
ENTRYPOINT []

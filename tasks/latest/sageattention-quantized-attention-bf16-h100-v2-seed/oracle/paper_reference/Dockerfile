# Paper-reference harness image (host-only, NOT the task's Work or Judge image).
#
# The private Judge image plus upstream SageAttention (thu-ml/SageAttention @
# d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5) compiled for sm_90a.  Built only by
# oracle/paper_reference/run_paper_reference.sh, which supplies the upstream
# checkout as the named build context `sage_src` (upstream is not vendored in
# this repository).  The trusted runtime, the frozen final holdout and the 62
# real layers come from the base Judge image unchanged.
ARG JUDGE_IMAGE=sageattention_quantized_attention.judge.sageattention_quantized_attention_bf16_h100_seed_v2:local
FROM ${JUDGE_IMAGE}
USER root
# Build parallelism only (no effect on the compiled code).  The recorded image
# used the defaults below.
ARG MAX_JOBS=16
ARG EXT_PARALLEL=4
ARG NVCC_THREADS=8
COPY --from=sage_src . /opt/sage_src
COPY check_ext.py /harness_build/check_ext.py
# No GPU needed at build time: setup.py honours TORCH_CUDA_ARCH_LIST.  Builds
# _qattn_sm90 (the wgmma kernel), _qattn_sm89 (the SageAttention2++ mma.sync
# kernel, ablation only) and _fused (per-warp INT8 / per-channel FP8 quant).
RUN cd /opt/sage_src \
 && TORCH_CUDA_ARCH_LIST="9.0" EXT_PARALLEL=${EXT_PARALLEL} NVCC_APPEND_FLAGS="--threads ${NVCC_THREADS}" MAX_JOBS=${MAX_JOBS} \
    pip install --no-build-isolation --no-deps -v . \
 && python /harness_build/check_ext.py
COPY paper_sm90_candidate.py run_paper_sm90.py kernel_only_bench.py /harness/
WORKDIR /harness
ENTRYPOINT []

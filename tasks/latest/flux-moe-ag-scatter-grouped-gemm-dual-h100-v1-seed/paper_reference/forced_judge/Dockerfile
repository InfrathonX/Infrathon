# LOCAL-ONLY forced-oracle Judge image for the Flux MoE AG+scatter task.
#
# Stage 1 builds ByteDance Flux (CUTLASS/CUDA ops, NVSHMEM-enabled because the
# MoE AG-scatter ops are only compiled with NVSHMEM) for sm_90a inside the
# task's own Judge image, so it links the exact torch 2.8.0 / CUDA 12.9 ABI.
# Stage 2 = the private Judge image (it carries the holdout) + the Flux wheel +
# NVSHMEM runtime + an admission override that records the enforced errors but
# admits the oracle. Never pushed, never retagged as :local.
#
# Upstream source is a named build context, e.g.
#   docker build --build-context flux_src=/path/to/flux@19831ca2 ...
ARG JUDGE_IMAGE=flux_moe_ag_scatter_dual_h100.judge.flux_moe_ag_scatter_grouped_gemm_dual_h100_seed_v1:local

FROM ${JUDGE_IMAGE} AS flux_build
USER root
ARG NVSHMEM_PIP=nvidia-nvshmem-cu12==3.3.9
ARG FLUX_JOBS=32
ARG FLUX_SM_CORES=132
RUN python -m pip install --no-cache-dir "${NVSHMEM_PIP}"
COPY --from=flux_src . /workspace/flux
WORKDIR /workspace/flux
RUN test -f 3rdparty/cutlass/include/cutlass/cutlass.h && test -f 3rdparty/nccl/Makefile \
 && sed -i 's/make -j\${nproc} src.staticlib/make -j${JOBS} src.staticlib/' build.sh \
 && OMP_NUM_THREADS=8 ./build.sh --arch 90 --sm-cores "${FLUX_SM_CORES}" --nvshmem --no_test \
      --package --jobs "${FLUX_JOBS}" \
 && ls -l dist/

FROM ${JUDGE_IMAGE}
USER root
ARG NVSHMEM_PIP=nvidia-nvshmem-cu12==3.3.9
RUN python -m pip install --no-cache-dir "${NVSHMEM_PIP}" \
 && ln -sf libnvshmem_host.so.3 "$(python -c 'import nvidia.nvshmem,pathlib;print(pathlib.Path(nvidia.nvshmem.__path__[0]))')/lib/libnvshmem_host.so"
COPY --from=flux_build /workspace/flux/dist/ /tmp/flux-wheel/
RUN python -m pip install --no-cache-dir --no-deps /tmp/flux-wheel/*.whl && rm -rf /tmp/flux-wheel \
 && pkg="$(python -c 'import importlib.util,pathlib;print(pathlib.Path(importlib.util.find_spec("flux").origin).parent)')" \
 && mkdir -p "${pkg}/lib" \
 && for so in libflux_cuda.so libflux_cuda_ths_op.so; do test -f "/opt/conda/lib/${so}" && ln -sf "/opt/conda/lib/${so}" "${pkg}/lib/${so}"; done \
 && ls -l "${pkg}/lib"
RUN mv /opt/flux_moe_ag_judge/source_admission.py /opt/flux_moe_ag_judge/source_admission_enforced.py
COPY source_admission_forced.py /opt/flux_moe_ag_judge/source_admission.py
# verify_moe_ag.py requires exactly the five protected labels; only the
# replaced admission module's digest is rebound.
RUN set -eu; \
    new="$(sha256sum /opt/flux_moe_ag_judge/source_admission.py | cut -d' ' -f1)"; \
    grep -q ' judge:source_admission.py$' /opt/flux_moe_ag_judge/protected_hashes.txt; \
    sed -i "s|^[0-9a-f]\{64\} judge:source_admission.py\$|${new} judge:source_admission.py|" /opt/flux_moe_ag_judge/protected_hashes.txt; \
    chown root:root /opt/flux_moe_ag_judge/source_admission.py /opt/flux_moe_ag_judge/source_admission_enforced.py; \
    chmod 0400 /opt/flux_moe_ag_judge/source_admission.py /opt/flux_moe_ag_judge/source_admission_enforced.py /opt/flux_moe_ag_judge/protected_hashes.txt
LABEL org.infrathon.forced-oracle="1" org.infrathon.flux-commit="19831ca2d820e3e782ed1d15d8b52d0898b78b26"
ENV SFORGE_LLM_JUDGE_MODE=off TRUSTED_CONTROL=1
WORKDIR /app/workdir/moe_ag_lab
USER root
ENTRYPOINT []

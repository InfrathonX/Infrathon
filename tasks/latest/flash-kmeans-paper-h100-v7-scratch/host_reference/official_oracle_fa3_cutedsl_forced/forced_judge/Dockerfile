# LOCAL-ONLY forced-oracle flash-kmeans Judge image.  Never pushed, never
# tagged :local, not the task's Work or Judge image.
#
# Base: the private Judge image (hidden cases included).  Adds
#   * NVIDIA CuTe DSL (nvidia-cutlass-dsl, pinned; the version line upstream's
#     fa3 branch names: "cutlass-dsl 4.4.x / 4.5.x"), and
#   * upstream flash-kmeans at the unmerged fa3 branch head
#     39be82a52e66964aaea816c7740db5526bb9fc90, supplied as the named build
#     context `fk_src` (a checkout or `git archive` of that commit; upstream is
#     not vendored), installed into the system site-packages.
# The CuTe DSL kernels JIT-compile for the visible GPU (sm_90a on H100) at the
# first call, inside the candidate process, like Triton.
#
# The only Judge change: source admission records and then bypasses its
# problems (admission_override.py).  The LLM integrity review is a host-side
# sforge step and is switched off by run_forced_oracle.sh.
ARG JUDGE_IMAGE=flash_kmeans.judge.flash_kmeans_paper_h100_kernel_lab_v7_scratch:local
FROM ${JUDGE_IMAGE}
USER root
ARG CUTLASS_DSL_VERSION=4.5.0
RUN pip install --no-cache-dir "nvidia-cutlass-dsl==${CUTLASS_DSL_VERSION}" \
 && python3 -c "import torch, triton; print(torch.__version__, triton.__version__)"
COPY --from=fk_src flash_kmeans /opt/flash_kmeans_fa3_src/flash_kmeans
COPY --from=fk_src pyproject.toml LICENSE.txt README.md MANIFEST.in /opt/flash_kmeans_fa3_src/
RUN pip install --no-cache-dir --no-deps --no-build-isolation /opt/flash_kmeans_fa3_src \
 && python3 -c "import importlib.util as u; assert u.find_spec('flash_kmeans.cutedsl_impl') and u.find_spec('cutlass.cute')"
COPY admission_override.py /tmp/forced_oracle/admission_override.py
RUN python3 /tmp/forced_oracle/admission_override.py \
 && chmod 400 /opt/flash_kmeans_benchmark/source_admission.py /opt/flash_kmeans_judge/protected_hashes.txt \
 && rm -rf /tmp/forced_oracle
LABEL org.infrathon.forced-oracle="local-only; source admission bypassed; not a published Judge"
WORKDIR /app/workdir/flash_kmeans_lab
ENTRYPOINT []

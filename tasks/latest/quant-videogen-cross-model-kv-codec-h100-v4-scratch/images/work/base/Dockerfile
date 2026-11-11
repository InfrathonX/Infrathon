# Quant-VideoGen Work base image, buildable from this repository alone.
#
# It replaces the historical
# `quant_videogen_h100_v2.work.quant_videogen_cross_model_kv_codec_h100_v2`
# image, which was produced by a task directory that does not live here and
# therefore made the Work image unbuildable from a fresh clone.  The only
# build inputs are a public, digest-pinned upstream image, the pinned
# dependency closure in `requirements.lock`, and task-local source.
#
# Differences from the historical v2 base, all deliberate:
#   * `/opt/mlsysforge` is a real directory.  The v2 base used
#     `/opt/edgebench` plus a hand-built symlink shim.
#   * No Node.js and no Codex CLI.  The agent CLI is a host-provided runtime
#     that SForge bind-mounts read-only at /opt/agent-runtime, and both the
#     Work and Judge Dockerfiles assert the image is agent-free.
#   * No `/opt/qvg-reference`, no v2 starter codec, and no upstream
#     `quant_videogen` package.  The scratch-v3 Work image deletes all three,
#     so building them here would only leave the reference implementation
#     recoverable in a lower layer of the published image.
#   * No `/etc/sudoers.d` entry for the agent (v2 removed it as well).
#   * `/opt/qvg-src/experiments` carries only the public Self-Forcing
#     integration.  The protected LongCat / HY-WorldPlay integrations are
#     added by the Judge image, never by Work.
#
# The build context is the task root, because the image needs `upstream/`,
# `metrics/` and this directory; `.dockerignore` admits exactly those trees.

ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel@sha256:a7103283ea7113e10ae5d014bd2342acebda0bc53164b2f7b1dd6eb7a766bdb6
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

ENV CUDA_HOME=/usr/local/cuda \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    DIFFUSERS_OFFLINE=1 \
    TRITON_CACHE_DIR=/workspace/.cache/triton \
    TORCH_EXTENSIONS_DIR=/workspace/.cache/torch-extensions \
    TZ=Etc/UTC
ENV PATH=/usr/local/cuda/bin:${PATH}

# `setpriv` (util-linux) and `pkill` (procps) are required by the Judge
# candidate sandbox; `curl` is probed through the Work image by the network
# preflight; `ninja-build` backs TORCH_EXTENSIONS_DIR builds.  The agent
# account is created here but is deliberately given no sudoers entry.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl ffmpeg git jq ninja-build procps ripgrep sudo xz-utils \
    && rm -rf /var/lib/apt/lists/* \
    && (id -u agent >/dev/null 2>&1 || useradd -m -s /bin/bash agent) \
    && git config --system safe.directory '*' \
    && command -v setpriv >/dev/null \
    && command -v pkill >/dev/null

COPY images/work/base/requirements.lock /tmp/qvg-requirements.lock
COPY images/work/base/check_environment.py /tmp/qvg-check-environment.py
RUN python -m pip install --no-cache-dir --no-deps -r /tmp/qvg-requirements.lock \
    && python /tmp/qvg-check-environment.py \
    && rm -f /tmp/qvg-requirements.lock /tmp/qvg-check-environment.py

# Task-local source.  `upstream/` is the pinned Quant-VideoGen snapshot
# (commit 0601468f2dbba6a17ac7086faec6d41527cad188, see
# upstream/INTEGRATION_NOTES.md); only its public Self-Forcing integration
# belongs in the Work lineage.
COPY upstream/experiments/Self-Forcing/ /opt/qvg-src/experiments/Self-Forcing/
COPY metrics/ /opt/mlsysforge/metrics/

RUN mkdir -p /opt/qvg-src/ckpts/Self-Forcing /models/public /eval/public \
      /workspace/public-results /workspace/.cache/triton \
      /workspace/.cache/torch-extensions /logs/agent \
      /opt/vbench /opt/vbench-cache \
    && install -d -o root -g root -m 0555 /opt/mlsysforge/assets \
    && install -d -o root -g root -m 0555 /workspace/public_tests \
    && install -d -o agent -g agent -m 0755 /workspace/submission \
    && find /opt/qvg-src /opt/mlsysforge -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete \
    && find /opt/qvg-src /opt/mlsysforge -depth -type d -name __pycache__ -empty -delete \
    && chown -R root:root /opt/qvg-src /opt/mlsysforge /workspace/public_tests \
    && chmod -R a-w /opt/qvg-src /opt/mlsysforge /workspace/public_tests \
    && chown -R agent:agent /workspace/submission /workspace/public-results \
      /workspace/.cache /logs/agent \
    && test -f /opt/mlsysforge/metrics/compare_frames.py \
    && test -f /opt/mlsysforge/metrics/run_vbench.py \
    && test -f /opt/qvg-src/experiments/Self-Forcing/demo_utils/taehv.py \
    && test ! -e /opt/qvg-src/experiments/LongCat \
    && test ! -e /opt/qvg-src/experiments/HY-WorldPlay \
    && test ! -e /opt/qvg-reference \
    && test ! -e /opt/mlsysforge/configs \
    && test ! -e /etc/sudoers.d/mlsysforge-agent \
    && ! command -v codex \
    && ! command -v node \
    && ! command -v npm

USER agent
WORKDIR /workspace
ENTRYPOINT []

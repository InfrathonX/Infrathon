# The Work base supplies CUDA/PyTorch, the pinned Python dependency closure,
# the public Self-Forcing integration and the public metrics.  It is built
# from this repository by images/work/base/Dockerfile (scripts/build_images.sh
# builds it first), so a fresh clone can produce this image without any
# externally supplied intermediate.  Model weights are never baked in; they
# arrive as read-only mounts from the external artifact root.
ARG BASE_WORK_IMAGE=quant_videogen_h100_scratch_v4.base.quant_videogen_cross_model_kv_codec_h100_scratch_v4:local
FROM ${BASE_WORK_IMAGE}

USER root

# Remove every inherited codec implementation/reference and every protected
# evaluator surface before installing the task-owned compatibility runtime.
# The task-owned base carries none of them, so these removals are no-ops
# there; they stay because BASE_WORK_IMAGE may still name an externally
# supplied base (historically the v2 Work image) that did.
#
# This image is agent-free: the agent CLI (Codex, Claude Code, ...) arrives
# as a host-provided runtime that SForge bind-mounts read-only at
# /opt/agent-runtime when SFORGE_AGENT_RUNTIME_DIR is set.  Switching agent
# kind or version never rebuilds this image, so any Node + Codex toolchain
# inherited from a base is stripped below.
RUN rm -rf \
      /opt/qvg-reference \
      /opt/qvg-src/quant_videogen \
      /opt/qvg-src/build \
      /opt/qvg-src/quant_videogen.egg-info \
      /opt/conda/lib/python3.11/site-packages/quant_videogen \
      /opt/conda/lib/python3.11/site-packages/quant_videogen-0.1.0.dist-info \
      /opt/mlsysforge/configs \
      /opt/qvg-judge \
      /opt/qvg-runners \
      /opt/qvg-policy \
      /opt/qvg-judge-assets \
      /models/hidden \
      /eval/hidden \
      /workspace/public_tests \
      /workspace/submission \
      /workspace/starter \
      /opt/qvg-src/experiments/Self-Forcing/pipeline/__pycache__ \
      /usr/local/lib/node_modules \
      /usr/local/include/node \
      /usr/local/share/doc/node \
    && rm -f /etc/sudoers.d/mlsysforge-agent \
      /usr/local/bin/node /usr/local/bin/npm /usr/local/bin/npx \
      /usr/local/bin/corepack /usr/local/bin/codex \
      /usr/local/share/man/man1/node.1 \
      /usr/local/CHANGELOG.md /usr/local/LICENSE /usr/local/README.md \
    && install -d -o root -g root -m 0555 /workspace/public_tests \
    && install -d -o agent -g agent -m 0755 /workspace/submission \
    && install -d -o root -g root -m 0555 /opt/qvg-src/quant_videogen

COPY runtime/quant_videogen/ /opt/qvg-src/quant_videogen/
COPY upstream/experiments/Self-Forcing/pipeline/causal_inference.py \
     /opt/qvg-src/experiments/Self-Forcing/pipeline/causal_inference.py
COPY images/work/public_tests/ /workspace/public_tests/
COPY policy/codec_source_policy.py /workspace/public_tests/codec_source_policy.py
COPY images/work/AGENT_TASK.md /workspace/AGENT_TASK.md
COPY images/work/public_tests/prepare_submit.sh /workspace/.sforge/prepare_submit.sh
COPY assets/public_prompts.txt /opt/mlsysforge/assets/public_prompts.txt
# Refresh the fidelity metric over the base image's copy so the public
# diagnostic runs the same code as the Judge (finite PSNR on identical frames).
COPY metrics/compare_frames.py /opt/mlsysforge/metrics/compare_frames.py
COPY --chown=agent:agent images/work/codex-config.toml /home/agent/.codex/config.toml

RUN sed -i \
      's#experiments/LongCat/longcat_video/utils/metric.py#released fidelity implementation#g' \
      /opt/mlsysforge/metrics/compare_frames.py \
    && sed -i \
      -e 's/Hunyuan Video/generic video/g' \
      -e 's/Hunyuan/generic/g' \
      /opt/qvg-src/experiments/Self-Forcing/demo_utils/taehv.py \
    && find /workspace /opt/qvg-src /opt/mlsysforge \
         -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete \
    && find /workspace /opt/qvg-src /opt/mlsysforge \
         -depth -type d -name __pycache__ -empty -delete \
    && chown -R root:root \
      /workspace/public_tests \
      /workspace/AGENT_TASK.md \
      /opt/qvg-src/quant_videogen \
      /opt/qvg-src/experiments/Self-Forcing/pipeline/causal_inference.py \
      /opt/mlsysforge/assets \
    && chmod -R a-w \
      /workspace/public_tests \
      /workspace/AGENT_TASK.md \
      /opt/qvg-src/quant_videogen \
      /opt/qvg-src/experiments/Self-Forcing/pipeline/causal_inference.py \
      /opt/mlsysforge/assets \
    && chown -R agent:agent /workspace/submission /home/agent/.codex \
    && chmod 600 /home/agent/.codex/config.toml \
    && chown -R root:root /workspace/.sforge \
    && chmod 0555 /workspace/.sforge /workspace/.sforge/prepare_submit.sh \
    && test -z "$(find /workspace/submission -mindepth 1 -print -quit)" \
    && test ! -e /opt/qvg-reference \
    && test ! -e /opt/mlsysforge/configs \
    && test ! -e /opt/qvg-src/experiments/LongCat \
    && test ! -e /opt/qvg-src/experiments/HY-WorldPlay \
    && test ! -e /opt/qvg-judge \
    && test ! -e /eval/hidden \
    && test ! -e /models/hidden \
    && test -f /opt/qvg-src/quant_videogen/payload.py \
    && test ! -e /opt/qvg-src/quant_videogen/functions.py \
    && test ! -e /opt/qvg-src/quant_videogen/real \
    && test ! -e /opt/qvg-src/quant_videogen/kmeans \
    && ! command -v codex \
    && ! command -v node \
    && ! command -v npm

USER agent
WORKDIR /workspace
ENTRYPOINT []

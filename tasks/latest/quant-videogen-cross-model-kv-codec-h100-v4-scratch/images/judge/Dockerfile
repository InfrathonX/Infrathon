ARG WORK_IMAGE=quant_videogen_h100_scratch_v4.work.quant_videogen_cross_model_kv_codec_h100_scratch_v4:local
FROM ${WORK_IMAGE}

USER root

# Only this image restores the private model integrations and protocol.  Root
# owns the controller, scorer, release mount parent, logs, and staged inputs.
# Participant-controlled Python is admitted first and then always runs through
# setpriv as the candidate principal of the GPU slot running the case: one
# account per release judge.slot_uids entry (UID == GID 23001..23008) so
# concurrent cases can neither kill nor read each other.  Slot 0 keeps the v3
# name qvg-candidate; slot i > 0 is qvg-candidate-<i+1>.  The commands stay
# literal so deploy/pyromind/qvg-v4/refresh_files.py can compare them with
# that profile's Dockerfile.judge; the controller checks accounts by UID/GID.
RUN rm -f /etc/sudoers.d/mlsysforge-agent \
    && rm -rf /workspace/submission \
    && install -d -o root -g root -m 0755 /workspace/submission \
    && groupadd --gid 23001 qvg-candidate && useradd --uid 23001 --gid 23001 --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin qvg-candidate \
    && groupadd --gid 23002 qvg-candidate-2 && useradd --uid 23002 --gid 23002 --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin qvg-candidate-2 \
    && groupadd --gid 23003 qvg-candidate-3 && useradd --uid 23003 --gid 23003 --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin qvg-candidate-3 \
    && groupadd --gid 23004 qvg-candidate-4 && useradd --uid 23004 --gid 23004 --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin qvg-candidate-4 \
    && groupadd --gid 23005 qvg-candidate-5 && useradd --uid 23005 --gid 23005 --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin qvg-candidate-5 \
    && groupadd --gid 23006 qvg-candidate-6 && useradd --uid 23006 --gid 23006 --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin qvg-candidate-6 \
    && groupadd --gid 23007 qvg-candidate-7 && useradd --uid 23007 --gid 23007 --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin qvg-candidate-7 \
    && groupadd --gid 23008 qvg-candidate-8 && useradd --uid 23008 --gid 23008 --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin qvg-candidate-8 \
    && install -d -o root -g root -m 0700 /eval \
    && install -d -o root -g root -m 0500 /eval/hidden \
    && install -d -o root -g root -m 0755 /models \
    && install -d -o root -g root -m 0555 /models/hidden \
    && install -d -o root -g root -m 0555 /models/hidden/longcat \
    && install -d -o root -g root -m 0555 /models/hidden/hy-worldplay \
    && install -d -o root -g root -m 0555 /models/hidden/hy-worldplay/base \
    && install -d -o root -g root -m 0555 /models/hidden/hy-worldplay/base/transformer \
    && install -d -o root -g root -m 0555 /models/hidden/hy-worldplay/base/vae \
    && install -d -o root -g root -m 0555 /models/hidden/hy-worldplay/base/scheduler \
    && install -d -o root -g root -m 0555 /models/hidden/hy-worldplay/base/text_encoder \
    && install -d -o root -g root -m 0555 /models/hidden/hy-worldplay/base/text_encoder/llm \
    && install -d -o root -g root -m 0555 /models/hidden/hy-worldplay/base/text_encoder/byt5-small \
    && install -d -o root -g root -m 0555 /models/hidden/hy-worldplay/base/text_encoder/Glyph-SDXL-v2 \
    && install -d -o root -g root -m 0555 /models/hidden/hy-worldplay/base/vision_encoder \
    && install -d -o root -g root -m 0555 /models/hidden/hy-worldplay/base/vision_encoder/siglip \
    && install -d -o root -g root -m 0555 /models/hidden/hy-worldplay/action \
    && install -d -o root -g root -m 0500 /opt/qvg-judge \
    && install -d -o root -g root -m 0555 /opt/qvg-runners \
    && install -d -o root -g root -m 0500 /opt/qvg-policy \
    && command -v setpriv >/dev/null \
    && command -v pkill >/dev/null \
    && command -v taskset >/dev/null

# Candidate-readable trusted model adapters.  They contain no release, gold,
# gate, or scoring data.  All controller/scorer/policy/config assets below are
# root-only and cannot be traversed by any slot account.  Case inputs (prompts,
# init videos, HY images) and goldens come only from the root-only
# /eval/hidden mount.
COPY upstream/experiments/LongCat/ /opt/qvg-src/experiments/LongCat/
COPY upstream/experiments/HY-WorldPlay/ /opt/qvg-src/experiments/HY-WorldPlay/
COPY judge/run_longcat.py judge/run_hy_worldplay_8b.py /opt/qvg-runners/

# Public Judge: everything but the release config (the 17 hidden cases and their
# golden digests); Dockerfile.private adds it on the host.  Hidden inputs and
# goldens are mounted at run time (/eval/hidden) as before.
COPY configs/paper_protocol.json /opt/mlsysforge/configs/paper_protocol.json
COPY metrics/ /opt/mlsysforge/metrics/
COPY policy/codec_source_policy.py /opt/qvg-policy/codec_source_policy.py
COPY judge/evaluate_hidden.py judge/run_judge.py /opt/qvg-judge/
COPY judge/judge_shards.py /opt/qvg-judge/judge_shards.py

RUN chown -R root:root \
      /opt/qvg-src/experiments/LongCat \
      /opt/qvg-src/experiments/HY-WorldPlay \
      /opt/qvg-runners \
      /opt/qvg-judge \
      /opt/qvg-policy \
      /opt/mlsysforge/configs \
      /opt/mlsysforge/metrics \
    && find /opt/qvg-src/experiments/LongCat \
         /opt/qvg-src/experiments/HY-WorldPlay \
         /opt/qvg-runners -type d -exec chmod 555 {} + \
    && find /opt/qvg-src/experiments/LongCat \
         /opt/qvg-src/experiments/HY-WorldPlay \
         /opt/qvg-runners -type f -exec chmod 444 {} + \
    && find /opt/qvg-judge /opt/qvg-policy \
         /opt/mlsysforge/configs /opt/mlsysforge/metrics \
         -type d -exec chmod 500 {} + \
    && find /opt/qvg-judge /opt/qvg-policy \
         /opt/mlsysforge/configs /opt/mlsysforge/metrics \
         -type f -exec chmod 400 {} + \
    && chmod 755 /workspace \
    && test -f /opt/qvg-src/quant_videogen/payload.py \
    && test ! -e /etc/sudoers.d/mlsysforge-agent

WORKDIR /workspace
ENTRYPOINT []

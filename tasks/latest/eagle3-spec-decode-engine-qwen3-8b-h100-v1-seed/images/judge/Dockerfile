ARG WORK_IMAGE=eagle3_spec_decode.work.eagle3_spec_decode_engine_qwen3_8b_h100_v1:local
FROM ${WORK_IMAGE}
# Public Judge: everything but private_data/ (holdout prompts, references,
# commitments).  This image may be published; Dockerfile.private adds
# private_data/ on the host that builds or pulls it (scripts/build_images.sh).
USER root
RUN rm -f /etc/sudoers.d/* 2>/dev/null || true \
 && mkdir -p /opt/eagle3_judge/verifier
COPY run_judge.py /opt/eagle3_judge/run_judge.py
COPY verify_eagle3.py /opt/eagle3_judge/verify_eagle3.py
COPY verifier/ /opt/eagle3_judge/verifier/
COPY protected_hashes.txt /opt/eagle3_judge/protected_hashes.txt
COPY seed_reference/ /opt/eagle3_judge/seed_reference/
# The Judge measures the unmodified seed engine on its own device at the start of
# every evaluation and derives the wall/memory caps and the oracle floor from it.
# This copy is taken from the Work layer before any submission can replace
# /app/workdir/spec_lab, and its identity is bound by protected_hashes.txt.
RUN cp -a /app/workdir/spec_lab/solutions/engine_v001 \
      /opt/eagle3_judge/seed_reference/solutions/engine_v001
RUN chown -R root:root /opt/eagle3_judge /app/workdir/spec_lab/runtime \
 && chmod 0511 /opt/eagle3_judge /opt/eagle3_judge/verifier \
 && chmod 0444 /opt/eagle3_judge/verifier/candidate_worker.py \
 && chmod 0400 /opt/eagle3_judge/run_judge.py /opt/eagle3_judge/verify_eagle3.py \
    /opt/eagle3_judge/protected_hashes.txt \
    /opt/eagle3_judge/verifier/controller.py /opt/eagle3_judge/verifier/source_admission.py \
    /opt/eagle3_judge/verifier/oracle_reference.py /opt/eagle3_judge/verifier/sampling.py /opt/eagle3_judge/verifier/scoring.py \
 && find /opt/eagle3_judge/seed_reference -type d -exec chmod 0555 {} + \
 && find /opt/eagle3_judge/seed_reference -type f -exec chmod 0444 {} +
ENV EAGLE3_INTERACTIVE_JUDGE_TIMEOUT=5400 EAGLE3_FINAL_JUDGE_TIMEOUT=5400 EAGLE3_GPU_ROLE=judge
ENTRYPOINT []
USER root

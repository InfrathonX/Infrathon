ARG WORK_IMAGE=qwen3_gkd_dual_h100.work.qwen3_4b_to_1p7b_gkd_distillation_optimizer_dual_h100_v1:local
FROM ${WORK_IMAGE}

# Public Judge: everything but private_data/ (hidden cases).  This image may be
# published; Dockerfile.private adds private_data/ and runs the protected-file
# check on the host that builds or pulls it (scripts/build_images.sh).

USER root

RUN rm -f /etc/sudoers.d/mlsysforge-agent \
    && sed -i '/^agent ALL=(ALL) NOPASSWD:ALL$/d' /etc/sudoers 2>/dev/null || true

ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV TOKENIZERS_PARALLELISM=false

COPY run_judge.py /opt/gkd_judge/run_judge.py
COPY verifier/ /opt/gkd_judge/verifier/
COPY protected_hashes.txt /opt/gkd_judge/protected_hashes.txt

RUN test "$(id -u agent)" -gt 0 \
    && test "$(id -g agent)" -gt 0 \
    && install -d -o root -g root -m 0555 /opt/gkd_candidate_worker \
    && install -o root -g root -m 0555 \
      /opt/gkd_judge/verifier/candidate_worker.py \
      /opt/gkd_candidate_worker/candidate_worker.py \
    && install -o root -g root -m 0555 \
      /opt/gkd_judge/verifier/e2e_rank_bootstrap.py \
      /opt/gkd_candidate_worker/e2e_rank_bootstrap.py \
    && install -d -o root -g root -m 0700 /logs/verifier \
    && chown -R root:root /opt/gkd_judge /app/workdir/gkd_lab/runtime \
    && chmod 700 /opt/gkd_judge /opt/gkd_judge/verifier \
    && find /opt/gkd_judge -type f -exec chmod 600 {} + \
    && chmod 700 /opt/gkd_judge/run_judge.py /opt/gkd_judge/verifier/verify_gkd.py \
    && find /app/workdir/gkd_lab/runtime -type d -exec chmod 555 {} + \
    && find /app/workdir/gkd_lab/runtime -type f -exec chmod 444 {} +

WORKDIR /app/workdir/gkd_lab
ENTRYPOINT []

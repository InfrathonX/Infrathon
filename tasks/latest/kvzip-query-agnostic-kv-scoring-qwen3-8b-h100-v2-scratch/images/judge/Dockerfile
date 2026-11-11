FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel
# Public Judge: everything but the private holdout (final manifest, materialized
# items, released-weights fingerprint).  This image may be published;
# Dockerfile.private adds them on the host that builds or pulls it.
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 KV_SCORING_MODEL_DIR=/opt/models/qwen3-8b/snapshots/unknown \
    KV_SCORING_CORPUS_DIR=/opt/kvzip_kv_scoring_judge/corpus \
    KV_SCORING_CANDIDATE_TIMEOUT=7200 KV_SCORING_TOTAL_TIMEOUT=10800 KV_SCORING_JUDGE_TIMEOUT=10900
RUN useradd -m -s /bin/bash agent && pip install --no-cache-dir transformers==4.57.1 accelerate==1.10.1 safetensors==0.6.2 rouge==1.0.1 rouge-score==0.1.2 \
    tree-sitter==0.21.3 tree-sitter-languages==1.10.2 nltk==3.9.1
COPY --from=task_runtime kv_scoring_runtime.py kv_scoring_metric.py /opt/kvzip_kv_scoring_worker/
COPY run_judge.py verify_kv_scoring.py candidate_worker.py trusted_replay.py judge_shards.py source_admission.py released_weights_gate.py protected_hashes.txt /opt/kvzip_kv_scoring_judge/
COPY --from=corpus_public items_public.jsonl /opt/kvzip_kv_scoring_judge/corpus/items_public.jsonl
RUN chown -R root:root /opt/kvzip_kv_scoring_judge /opt/kvzip_kv_scoring_worker \
 && chmod 0511 /opt/kvzip_kv_scoring_judge && chmod 0500 /opt/kvzip_kv_scoring_judge/corpus \
 && chmod 0400 /opt/kvzip_kv_scoring_judge/corpus/items_public.jsonl \
    /opt/kvzip_kv_scoring_judge/run_judge.py /opt/kvzip_kv_scoring_judge/verify_kv_scoring.py \
    /opt/kvzip_kv_scoring_judge/trusted_replay.py /opt/kvzip_kv_scoring_judge/judge_shards.py \
    /opt/kvzip_kv_scoring_judge/source_admission.py /opt/kvzip_kv_scoring_judge/protected_hashes.txt \
 && chmod 0444 /opt/kvzip_kv_scoring_judge/candidate_worker.py /opt/kvzip_kv_scoring_judge/released_weights_gate.py \
    /opt/kvzip_kv_scoring_worker/kv_scoring_runtime.py /opt/kvzip_kv_scoring_worker/kv_scoring_metric.py \
 && chmod 0555 /opt/kvzip_kv_scoring_worker && mkdir -p /app/workdir/kv_scoring_lab /logs/verifier /home/agent/.cache \
 && chown -R agent:agent /home/agent/.cache && chmod 0700 /logs/verifier \
 && python -c "from tree_sitter_languages import get_parser; get_parser('python'); import nltk.translate.bleu_score"
WORKDIR /app/workdir/kv_scoring_lab
USER root
ENTRYPOINT []

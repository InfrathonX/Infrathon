FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel
# Public Judge: everything but the private holdout (final manifest, materialized
# items).  This image may be published;
# Dockerfile.private adds them on the host that builds or pulls it.
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 KV_EVICTION_MODEL_DIR=/opt/models/qwen3-8b/snapshots/unknown \
    KV_EVICTION_CORPUS_DIR=/opt/kvpress_kv_eviction_judge/corpus \
    KV_EVICTION_CANDIDATE_TIMEOUT=7200 KV_EVICTION_TOTAL_TIMEOUT=10800 KV_EVICTION_JUDGE_TIMEOUT=10900
RUN useradd -m -s /bin/bash agent && pip install --no-cache-dir transformers==4.57.1 accelerate==1.10.1 safetensors==0.6.2
COPY --from=task_runtime kv_eviction_runtime.py /opt/kvpress_kv_eviction_worker/kv_eviction_runtime.py
COPY run_judge.py verify_kv_eviction.py candidate_worker.py trusted_replay.py judge_shards.py source_admission.py protected_hashes.txt /opt/kvpress_kv_eviction_judge/
COPY --from=corpus_public items_public.jsonl /opt/kvpress_kv_eviction_judge/corpus/items_public.jsonl
RUN chown -R root:root /opt/kvpress_kv_eviction_judge /opt/kvpress_kv_eviction_worker \
 && chmod 0511 /opt/kvpress_kv_eviction_judge && chmod 0500 /opt/kvpress_kv_eviction_judge/corpus \
 && chmod 0400 /opt/kvpress_kv_eviction_judge/corpus/items_public.jsonl \
    /opt/kvpress_kv_eviction_judge/run_judge.py /opt/kvpress_kv_eviction_judge/verify_kv_eviction.py \
    /opt/kvpress_kv_eviction_judge/trusted_replay.py /opt/kvpress_kv_eviction_judge/judge_shards.py \
    /opt/kvpress_kv_eviction_judge/source_admission.py /opt/kvpress_kv_eviction_judge/protected_hashes.txt \
 && chmod 0444 /opt/kvpress_kv_eviction_judge/candidate_worker.py /opt/kvpress_kv_eviction_worker/kv_eviction_runtime.py \
 && chmod 0555 /opt/kvpress_kv_eviction_worker && mkdir -p /app/workdir/kv_eviction_lab /logs/verifier /home/agent/.cache \
 && chown -R agent:agent /home/agent/.cache && chmod 0700 /logs/verifier
WORKDIR /app/workdir/kv_eviction_lab
USER root
ENTRYPOINT []

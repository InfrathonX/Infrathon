# LOCAL-ONLY forced-oracle Judge image for the EAGLE-3 task.
#
# FROM the task's private Judge image (it carries the holdout), this adds:
#   * an isolated SGLang virtualenv at /opt/sglang-venv (the official EAGLE-3
#     serving integration; its own torch/transformers, so the Judge's own
#     torch 2.8.0 / transformers 4.53.1 environment is not touched), and
#   * an admission override that records every enforced-admission error but
#     admits the candidate (oracle-only; agents are still judged by the
#     unmodified :local image).
# The image is never pushed and never retagged as :local.
ARG JUDGE_IMAGE=eagle3_spec_decode.judge.eagle3_spec_decode_engine_qwen3_8b_h100_v1:local
FROM ${JUDGE_IMAGE}
USER root
ARG SGLANG_VERSION=0.5.20
ARG UV_VERSION=0.9.30
RUN /opt/conda/bin/python3 -m pip install --no-cache-dir --target /opt/uv-bootstrap "uv==${UV_VERSION}" \
 && /opt/uv-bootstrap/bin/uv venv --python /opt/conda/bin/python3.11 /opt/sglang-venv \
 && VIRTUAL_ENV=/opt/sglang-venv /opt/uv-bootstrap/bin/uv pip install --no-cache --prerelease=allow "sglang==${SGLANG_VERSION}" \
 && rm -rf /opt/uv-bootstrap /root/.cache \
 && chmod -R a+rX /opt/sglang-venv
# Admission override: keep the enforced module under a new name and wrap it.
RUN mv /opt/eagle3_judge/verifier/source_admission.py /opt/eagle3_judge/verifier/source_admission_enforced.py
COPY source_admission_forced.py /opt/eagle3_judge/verifier/source_admission.py
# Rebind the protected-hash line of the replaced file so verify_protected still
# checks every other Judge/runtime/seed identity unchanged.
RUN set -eu; \
    new="$(sha256sum /opt/eagle3_judge/verifier/source_admission.py | cut -d' ' -f1)"; \
    grep -q '  judge:verifier/source_admission.py$' /opt/eagle3_judge/protected_hashes.txt; \
    sed -i "s|^[0-9a-f]\{64\}  judge:verifier/source_admission.py\$|${new}  judge:verifier/source_admission.py|" /opt/eagle3_judge/protected_hashes.txt; \
    printf '%s  judge:verifier/source_admission_enforced.py\n' \
      "$(sha256sum /opt/eagle3_judge/verifier/source_admission_enforced.py | cut -d' ' -f1)" >> /opt/eagle3_judge/protected_hashes.txt; \
    chown root:root /opt/eagle3_judge/verifier/source_admission.py /opt/eagle3_judge/verifier/source_admission_enforced.py; \
    chmod 0400 /opt/eagle3_judge/verifier/source_admission.py /opt/eagle3_judge/verifier/source_admission_enforced.py /opt/eagle3_judge/protected_hashes.txt
LABEL org.infrathon.forced-oracle="1" org.infrathon.sglang-version="${SGLANG_VERSION}"
ENV SFORGE_LLM_JUDGE_MODE=off TRUSTED_CONTROL=1
USER root

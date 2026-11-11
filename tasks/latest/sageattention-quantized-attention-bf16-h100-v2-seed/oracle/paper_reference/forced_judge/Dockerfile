# LOCAL-ONLY forced-oracle Judge image.  Never pushed, never tagged :local.
#
# Base: the paper-reference harness image built from ../Dockerfile, i.e. the
# private Judge image (frozen final holdout + 62 real layers) plus upstream
# SageAttention (thu-ml/SageAttention @ d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5)
# compiled for sm_90a and installed into the system site-packages.
#
# The only Judge change: source admission records and then bypasses its errors
# (admission_override.py), so the compiled-extension submission reaches the
# unmodified verifier, worker and trusted runtime.  The LLM integrity review is
# a host-side sforge step and is switched off by run_forced_oracle.sh
# (SFORGE_LLM_JUDGE_MODE=off).
ARG PAPER_IMAGE=sageattention-paper-sm90-harness:d1a57a5
FROM ${PAPER_IMAGE}
USER root
COPY admission_override.py /tmp/forced_oracle/admission_override.py
RUN python3 /tmp/forced_oracle/admission_override.py /opt/sageattention_quant_attention_judge \
 && chmod 0400 /opt/sageattention_quant_attention_judge/source_admission.py \
       /opt/sageattention_quant_attention_judge/protected_hashes.txt \
 && rm -rf /tmp/forced_oracle /harness /harness_build \
 && python3 -c "import importlib.util as u; assert u.find_spec('sageattention') is not None"
LABEL org.infrathon.forced-oracle="local-only; source admission bypassed; not a published Judge"
WORKDIR /app/workdir/quant_attention_lab
ENTRYPOINT []

# Sanitize the public runtime before it is flattened into miles-model-dependencies-flat:local.
# Same recipe as the qualified build (flat-runtime.json clean_recipe_sha256 910a8dc5...), with the
# parent passed as a build argument instead of a fixed local tag.
ARG PARENT=miles-model-public-runtime:local
FROM ${PARENT}
USER root
RUN apt-get update && apt-get install -y --no-install-recommends ripgrep jq procps \
    && rm -rf /var/lib/apt/lists/* /root/.cache /root/.ssh /root/.config \
    && test ! -e /root/miles \
    && find /root/Megatron-LM /sgl-workspace/sglang -type d -name .git -prune -exec rm -rf '{}' + \
    && (id agent >/dev/null 2>&1 || useradd -m -s /bin/bash agent) \
    && chmod 755 /root \
    && chmod -R go-w /root/Megatron-LM /sgl-workspace/sglang \
    && git config --system safe.directory '*'
RUN python -c "import importlib.util; assert importlib.util.find_spec('miles') is None; assert importlib.util.find_spec('miles_plugins') is None"
ENTRYPOINT []
CMD ["bash"]

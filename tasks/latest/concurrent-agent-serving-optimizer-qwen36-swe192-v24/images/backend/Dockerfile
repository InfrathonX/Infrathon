# The external vLLM backend this task measures against.
#
# Contract revision v3 binds the backend runtime to this recipe rather than to
# the Docker image ID of one host's build: the official vLLM v0.24.0 release
# image (engine 0.24.0+ee0da84a..., the v0.24.0 tag commit), digest-pinned,
# with nothing layered on top.  Every serving parameter comes from the frozen
# model profile on the command line (see docs/REBUILD.md §3), so the image is
# the runtime and only the runtime.  scripts/preflight.py hashes this file
# against runtime_image_recipe.dockerfile_sha256 in the serving contract.
FROM vllm/vllm-openai:v0.24.0@sha256:f9de5cd9fa907fbf6dbba691eb7db095d48ad58ea283e3eba7142f9a91e186e8

LABEL org.mlsysforge.task="concurrent-agent-serving-optimizer-qwen36-swe192-v24" \
      org.mlsysforge.role="backend-runtime" \
      org.mlsysforge.vllm="0.24.0"

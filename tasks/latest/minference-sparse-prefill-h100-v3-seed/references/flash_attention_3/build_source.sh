#!/bin/bash
# Author-only CPU source build. Work/Judge candidate images must not install FA3.
set -euo pipefail

build_root="$(realpath -m "${1:?usage: build_source.sh OUTPUT_DIRECTORY}")"
source_commit=060c9188beec3a8b62b33a3bfa6d5d2d44975fab
cutlass_commit=dc4817921edda44a549197ff3a9dcf5df0636e7b
base_image=pytorch/pytorch@sha256:cf5aa3f7045a68c10d80f546746591c5ccae6a33729e5e32625ff76bd2c036fe
mkdir -p "$build_root/wheels"
if [[ ! -d "$build_root/source/.git" ]]; then
    git clone --filter=blob:none --no-checkout https://github.com/Dao-AILab/flash-attention.git "$build_root/source"
fi
git -C "$build_root/source" checkout --detach "$source_commit"
git -C "$build_root/source" submodule update --init csrc/cutlass
test "$(git -C "$build_root/source" rev-parse HEAD)" = "$source_commit"
test "$(git -C "$build_root/source/csrc/cutlass" rev-parse HEAD)" = "$cutlass_commit"
test -z "$(git -C "$build_root/source" status --porcelain)"

build_env=(--env NVIDIA_VISIBLE_DEVICES=void --env CUDA_VISIBLE_DEVICES=
    --env MAX_JOBS=4 --env NVCC_THREADS=2 --env TORCH_CUDA_ARCH_LIST=9.0a
    --env FLASH_ATTENTION_FORCE_BUILD=TRUE)
for feature in BACKWARD PAGEDKV APPENDKV LOCAL SOFTCAP FP16 FP8 VARLEN SM80 HDIM64 HDIM96 HDIM192 HDIM256; do
    build_env+=(--env "FLASH_ATTENTION_DISABLE_${feature}=TRUE")
done

docker run --rm --runtime runc --cpus 8 --memory 64g --memory-swap 64g \
    "${build_env[@]}" \
    --mount "type=bind,src=$build_root/source,dst=/source" \
    --mount "type=bind,src=$build_root/wheels,dst=/wheels" \
    --entrypoint /bin/bash "$base_image" -euc '
        python3 -c "import torch,triton; assert torch.__version__ == \"2.8.0+cu129\"; assert triton.__version__ == \"3.4.0\"; assert not torch.cuda.is_initialized()"
        apt-get update
        apt-get install -y --no-install-recommends git=1:2.34.1-1ubuntu1.17
        cd /source/hopper
        python3 -m pip wheel --no-build-isolation --no-deps . --wheel-dir /wheels
    '
sha256sum "$build_root"/wheels/flash_attn_3-*.whl

"""Upstream flash-kmeans fa3 branch (CuTe DSL SM90 assign) behind the task ABI -- FORCED ORACLE, NOT admissible.

Upstream: https://github.com/svg-project/flash-kmeans, unmerged branch `fa3` at
39be82a52e66964aaea816c7740db5526bb9fc90 ("[feat] Remove x^2 compute", 2026-05-13), installed as the
`flash_kmeans` package in the local-only forced-oracle Judge image together with
nvidia-cutlass-dsl 4.5.0 (forced_judge/Dockerfile).  This file is glue only: one Lloyd iteration
exactly as upstream's own `flash_kmeans.cutedsl_impl.cutedsl_kmeans_Euclid` loop body does it.

* assign: `cutedsl_assign_euclid` -- fused TMA + WGMMA CuTe DSL kernel with the in-register
  shifted-distance argmin epilogue for B == 1 fp16/bf16 (D % 16 == 0, D <= 512); upstream itself
  falls back to its Triton `euclid_assign_triton` (H100 heuristic table, no x_sq) for B > 1.
  Tile selection: upstream option `autotune=False`: the static H200-tuned `_pick_tile` heuristic, non-warp-specialized kernel, one compile per shape.
  `x_sq` is not needed (the ||x||^2 term is constant per row).
* update: `triton_lloyd_centroid_step_euclid` -- upstream sort + sorted chunk update + fused
  finalize (empty clusters keep the old centroid); counts come from its `cnts_buf`.

Only glue change: after each assign call, upstream's module-level `_dlpack_cache` (keyed by
data_ptr) is cleared.  Upstream never evicts it, so it would keep every caller tensor alive
(measured: +3 x the input size per 3 fresh-input calls), which is an unbounded leak under the
Judge's fresh-input protocol; clearing it changes no kernel, tile or launch.

It imports `flash_kmeans` and `cutlass`, which the task's clean-room source policy forbids, so it can
only be scored in the local-only forced-oracle Judge image.
"""
import torch
from flash_kmeans import cutedsl_impl as _fa3_cutedsl
from flash_kmeans.centroid_update_triton import triton_lloyd_centroid_step_euclid

_AUTOTUNE = False


def assign(x, centroids, x_sq):
    try:
        return _fa3_cutedsl.cutedsl_assign_euclid(x, centroids, autotune=_AUTOTUNE)
    finally:
        _fa3_cutedsl._dlpack_cache.clear()


def update(x, labels, old_centroids):
    batch = x.shape[0]
    clusters = old_centroids.shape[1]
    counts = torch.zeros((batch, clusters), device=x.device, dtype=torch.int32)
    new_centroids, _, _ = triton_lloyd_centroid_step_euclid(
        x, labels, old_centroids, cnts_buf=counts
    )
    return counts, new_centroids

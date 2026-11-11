# Host-only optimisation ladder

These engines are qualification evidence only. Each began as a direct copy of
`solutions/engine_v001` and retains its `build`/`Engine.generate` ABI, target/head
weights, greedy verification rule, selected 60/7/10 tree policy, canonical
truncation, and KV commit.

- `engine_l1_device_tree`: replaces per-node CPU/Python tree-mask and retrieval-path
  construction with device tensor ancestry operations. Greedy posterior selection
  remains on device. Acceptance length and the stop test over only the newly accepted
  chunk share one device-to-host transfer per cycle; there is no full-history stop
  scan. Fixed invalid rows make retrieval shape static without changing leaf order or
  which verified path can win.
- `engine_l2_cuda_graphs`: L1 plus a CUDA graph for the fixed
  `[top_k, hidden] -> RMSNorm -> lm_head -> log-softmax -> top-k` tail at every draft
  depth. Capturing the decoder-attention body or target verification was not feasible
  within this demonstration: the faithful legacy draft uses newly concatenated,
  variable-length tuple caches whose addresses and shapes change each cycle. Doing so
  legitimately requires the larger static draft-KV rewrite left as participant
  headroom.
- `engine_l3_sdpa_tree`: L1 plus `torch.nn.functional.scaled_dot_product_attention`
  for target prefill and verification. It converts the existing additive causal/tree
  mask to SDPA's boolean allow-mask and uses native GQA, leaving Q/K/V projections,
  RoPE, output projection, KV writes, and tree semantics unchanged.

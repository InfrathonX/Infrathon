# Upstream provenance

All derived logic is from StreamDiffusionV2 commit
`2f8ee1c5b6616b115662d190d5213677882d85d7`, licensed Apache-2.0.

| Local file | Upstream source(s) |
|---|---|
| `dit.py` | `models/wan/causal_model.py`; `models/wan/wan_wrapper.py`; `models/wan/wan_base/modules/model.py` |
| `kv_policy.py` | `models/wan/causal_model.py` (`CausalWanSelfAttention.forward`, KV branch) |
| `loader.py` | `models/wan/wan_wrapper.py`; `streamv2v/inference_common.py` |
| `plan.py` | `streamv2v/inference.py` (`compute_noise_scale_and_step` and single Stream-Batch loop) |
| `reference.py` | `models/wan/causal_stream_inference.py`; `streamv2v/inference.py` |
| `vae.py` | `models/wan/wan_wrapper.py`; `models/wan/wan_base/modules/vae.py` |
| `sink.py` | New task-owned code; no upstream source copied |

The following support files are exact copies from the same pinned commit. Their original
headers are preserved:

- `_upstream/LICENSE` ← `LICENSE`
- `_upstream/models/__init__.py` ← `models/__init__.py`
- `_upstream/models/model_interface.py` ← `models/model_interface.py`
- `_upstream/models/scheduler.py` ← `models/scheduler.py`
- `_upstream/models/wan/__init__.py` ← `models/wan/__init__.py`
- `_upstream/models/wan/causal_model.py` ← `models/wan/causal_model.py`
- `_upstream/models/wan/flow_match.py` ← `models/wan/flow_match.py`
- `_upstream/models/wan/wan_wrapper.py` ← `models/wan/wan_wrapper.py`
- `_upstream/models/wan/wan_base/__init__.py` ← `models/wan/wan_base/__init__.py`
- `_upstream/models/wan/wan_base/modules/__init__.py` ← `models/wan/wan_base/modules/__init__.py`
- `_upstream/models/wan/wan_base/modules/attention.py` ← `models/wan/wan_base/modules/attention.py`
- `_upstream/models/wan/wan_base/modules/model.py` ← `models/wan/wan_base/modules/model.py`
- `_upstream/models/wan/wan_base/modules/t5.py` ← `models/wan/wan_base/modules/t5.py`
- `_upstream/models/wan/wan_base/modules/tokenizers.py` ← `models/wan/wan_base/modules/tokenizers.py`
- `_upstream/models/wan/wan_base/modules/vae.py` ← `models/wan/wan_base/modules/vae.py`

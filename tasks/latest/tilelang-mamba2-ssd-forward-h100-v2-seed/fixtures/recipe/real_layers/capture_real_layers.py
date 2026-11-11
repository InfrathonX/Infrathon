#!/usr/bin/env python3
"""Capture the SSD inputs of real Mamba layers, as REAL_LAYERS.json describes.

For each source the model runs one batch-1 forward pass over real text.  The
module-level ``mamba2_chunk_scan`` of the Transformers modeling file is wrapped:
for a selected layer the wrapper records exactly what the operator of this task
consumes -- ``x``, the preprocessed ``dt = clamp(softplus(dt + dt_bias))``,
``A = -exp(A_log)``, ``B``, ``C`` and ``D`` -- and for every layer it computes
the scan with this task's own blocked FP32 reference, which is exact and, unlike
the library's pure-torch fallback, fits in memory at these lengths.  Each
captured layer is one ``torch.save`` dict; ``real_layers.manifest.json`` lists
every file with its SHA-256 and the recipe's SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELING = {
    "granitemoehybrid": "transformers.models.granitemoehybrid.modeling_granitemoehybrid",
    "bamba": "transformers.models.bamba.modeling_bamba",
}


def load_reference(runtime_path: Path):
    sys.path.insert(0, str(runtime_path.parent))
    return importlib.import_module(runtime_path.stem)


def text_tokens(tokenizer, dataset_dir: Path, entry: dict) -> torch.Tensor:
    table = pq.read_table(dataset_dir / entry["parquet"])
    text = "".join(table.column("text").to_pylist())
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    start, length = int(entry["skip_tokens"]), int(entry["seqlen"])
    if len(ids) < start + length:
        raise SystemExit(f"{entry['parquet']} has {len(ids)} tokens, fewer than {start + length}")
    return torch.tensor([ids[start:start + length]], dtype=torch.long)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True, help="the task's trusted runtime")
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--model-dirs", nargs="+", required=True, help="one snapshot per source")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    recipe_bytes = args.recipe.read_bytes()
    recipe = json.loads(recipe_bytes)
    runtime = load_reference(args.runtime)
    args.out.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict] = {}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    for source, model_dir in zip(args.sources, args.model_dirs, strict=True):
        entry = recipe["sources"][source]
        geometry = runtime.GEOMETRIES[entry["geometry"]] if entry["geometry"] in runtime.GEOMETRIES else None
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).cuda().eval()
        modeling = importlib.import_module(MODELING[model.config.model_type])
        wanted = {int(layer): f"{entry['name_prefix']}{int(layer):02d}" for layer in entry["layers"]}
        current = {"layer": None}

        def pre_hook(module, _args, _kwargs=None):
            current["layer"] = int(module.layer_idx)

        mixers = [m for m in model.modules() if hasattr(m, "A_log") and hasattr(m, "dt_bias")]
        for mixer in mixers:
            mixer.register_forward_pre_hook(pre_hook)
        if set(wanted) - {int(m.layer_idx) for m in mixers}:
            raise SystemExit(f"{source}: layers {sorted(set(wanted) - {int(m.layer_idx) for m in mixers})} are not Mamba layers")

        def chunk_scan(hidden_states, dt, A, B, C, chunk_size, D=None, dt_bias=None,
                       initial_states=None, dt_softplus=False, dt_limit=(0.0, float("inf")),
                       return_final_states=False, **_kwargs):
            if initial_states is not None or return_final_states:
                raise RuntimeError("the capture runs one uncached prefill pass")
            step = dt.float()
            if dt_bias is not None:
                step = step + dt_bias.float()
            if dt_softplus:
                step = F.softplus(step)
            step = torch.clamp(step, min=dt_limit[0], max=dt_limit[1]).contiguous()
            tensors = {
                "x": hidden_states.to(torch.bfloat16).contiguous(),
                "dt": step,
                "A": A.float().contiguous(),
                "B": B.to(torch.bfloat16).contiguous(),
                "C": C.to(torch.bfloat16).contiguous(),
                "D": D.float().contiguous(),
            }
            layer = current["layer"]
            if layer in wanted:
                name = wanted[layer]
                if geometry is not None:
                    shape = (1, int(entry["seqlen"]), geometry["h"], geometry["p"])
                    if tuple(tensors["x"].shape) != shape:
                        raise RuntimeError(f"{name}: x is {tuple(tensors['x'].shape)}, not {shape}")
                path = args.out / f"{name}.pt"
                torch.save({key: value.cpu() for key, value in tensors.items()}, path)
                files[name] = {
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "source": source,
                    "layer": layer,
                    "shape": list(tensors["x"].shape),
                    "groups": int(tensors["B"].shape[2]),
                    "dstate": int(tensors["B"].shape[3]),
                    "dt_range": [float(step.min()), float(step.max())],
                    "a_range": [float(tensors["A"].min()), float(tensors["A"].max())],
                }
                print(f"captured {name} {files[name]['shape']}", flush=True)
            out = runtime.reference_output(
                torch, tensors["x"], tensors["dt"], tensors["A"], tensors["B"], tensors["C"], tensors["D"]
            )
            return out

        modeling.mamba2_chunk_scan = chunk_scan
        ids = text_tokens(tokenizer, args.dataset_dir, entry).cuda()
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
        missing = sorted(set(wanted.values()) - set(files))
        if missing:
            raise SystemExit(f"{source}: layers were not captured: {missing}")
        del model
        torch.cuda.empty_cache()

    manifest = {
        "schema_version": 1,
        "recipe": {"sha256": hashlib.sha256(recipe_bytes).hexdigest()},
        "sources": {source: recipe["sources"][source]["model"] for source in args.sources},
        "files": dict(sorted(files.items())),
    }
    (args.out / "real_layers.manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(files)} layers to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

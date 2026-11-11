"""Trusted author prototype: capture one prompt, never an entire QKV tree.

Load and authenticate the model once in the trusted preparation process. Keep
that process separate from the candidate process. FlashAttention-2, explicit
all-ones mask and stop at L27 produce a newly bound dense trajectory. Historical
SDPA capture file hashes are not claimed to reproduce this trajectory.
GPU execution and bitwise reproduction have not been qualified for this file.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import struct
import time
from importlib.metadata import version
from single_prompt_archive_v3 import (PROTOCOL, LAYERS, canonical_sha, canonical_bytes,
    prompt_specs, validate_bundle, MAX_SCRATCH_BYTES, directory_identity)


def token_sha(ids):
    if not isinstance(ids, list) or any(type(i) is not int or not 0 <= i < 2**31 for i in ids):
        raise ValueError("token IDs must be nonnegative int32")
    return hashlib.sha256(b"".join(struct.pack("<i", i) for i in ids)).hexdigest()


def verify_prompt(plan, prompt):
    specs = prompt_specs(plan, prompt.get("id"))
    f = next(iter(specs.values()))
    if (prompt.get("task"), prompt.get("length"), prompt.get("actual_input_length"), prompt.get("input_token_sha256")) != (
            f["task"], f["context_budget"], f["actual_n"], f["input_token_sha256"]):
        raise ValueError("prompt metadata differs from the fixed plan")
    if len(prompt.get("input_ids", [])) != f["actual_n"] or token_sha(prompt["input_ids"]) != f["input_token_sha256"]:
        raise ValueError("prompt token bytes differ")
    return specs


def write_private_json(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(canonical_bytes(value)); handle.flush(); os.fsync(handle.fileno())


class StopAfterCapture(Exception):
    pass


class TrustedPromptPreparer:
    def __init__(self, model_path, recipe, prompt_manifest, plan, *, image_id, expected_gpu_uuid,
                 capture_helper, torch, transformers, purpose="private-judge"):
        """The caller must attest the actual immutable image and GPU lease."""
        self.torch, self.plan, self.helper = torch, plan, capture_helper
        from single_prompt_archive_v3 import validate_purpose
        validate_purpose(purpose, plan["partition"], os.geteuid())
        self.purpose = purpose
        model_path = Path(model_path)
        if model_path.name != recipe["model"]["revision"] or plan["bindings"]["model"] != recipe["model"]:
            raise ValueError("model revision/file identity differs")
        if plan["bindings"]["image_id"] != image_id:
            raise ValueError("preparation image binding differs")
        if plan["bindings"].get("capture_helper_sha256") != hashlib.sha256(Path(capture_helper.__file__).read_bytes()).hexdigest():
            raise ValueError("capture helper source binding differs")
        if plan["protocol"] != PROTOCOL or plan["bindings"].get("attention_backend") != "flash_attention_2":
            raise ValueError("new on-demand preparation must bind FlashAttention-2")
        if version("flash_attn") != "2.8.3":
            raise ValueError("preparation requires pinned FlashAttention 2.8.3")
        actual = str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix("gpu-")
        if actual != expected_gpu_uuid.lower().removeprefix("gpu-"):
            raise ValueError("actual GPU differs from preparation lease")
        capture_helper.verify_model_files(model_path, recipe)
        for name, digest in prompt_manifest["tokenizer_sha256"].items():
            if capture_helper.sha256_file(model_path / name)[0] != digest:
                raise ValueError("tokenizer differs from fixed prompt inputs")
        expected_software = {"torch": torch.__version__, "cuda": torch.version.cuda,
                             "transformers": transformers.__version__}
        if plan["bindings"]["software"] != expected_software:
            raise ValueError("preparation software differs")
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_path,
            torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map={"": 0},
            low_cpu_mem_usage=True, local_files_only=True, trust_remote_code=False).eval()
        if self.model.config.num_hidden_layers != 32 or self.model.generation_config.eos_token_id != [128001, 128008, 128009]:
            raise ValueError("model layer/EOS convention differs")
        if self.model.config._attn_implementation != "flash_attention_2":
            raise ValueError("model did not select the declared FA2 backend")

    def capture(self, prompt, root):
        specs = verify_prompt(self.plan, prompt)
        root = Path(root)
        if root.exists() or root.is_symlink() or not root.is_absolute():
            raise ValueError("choose a new absolute private scratch directory")
        # No pre-existing ancestor may redirect the selected scratch location.
        if any(p.is_symlink() for p in root.parents):
            raise ValueError("scratch ancestor symlink is forbidden")
        root.mkdir(mode=0o700)
        torch = self.torch
        records, hook_handles, captured = {}, [], []
        started = time.perf_counter()
        try:
            for layer in LAYERS:
                def hook(module, positional, keyword, layer=layer):
                    q, k, v = self.helper._attention_inputs(module, positional, keyword)
                    n = prompt["actual_input_length"]
                    if any(t.dtype != torch.bfloat16 for t in (q, k, v)) or [tuple(t.shape) for t in (q,k,v)] != [(1,32,n,128),(1,8,n,128),(1,8,n,128)]:
                        raise ValueError("captured native-GQA shape/dtype differs")
                    if any(not bool(torch.isfinite(t).all()) for t in (q,k,v)):
                        raise ValueError("captured QKV is nonfinite")
                    fid = prompt["id"] + f"-L{layer}"
                    entry = specs[fid]
                    path = root / entry["file"]
                    self.helper.write_safetensors(path, {"q": q, "k": k, "v": v}, {
                        "case_id": fid, "prompt_id": prompt["id"], "layer": str(layer),
                        "actual_n": str(n), "input_token_sha256": prompt["input_token_sha256"],
                        "task": prompt["task"], "context_budget": str(prompt["length"]),
                        "partition": self.plan["partition"], "trajectory": entry["trajectory"],
                        "protocol": PROTOCOL})
                    os.chmod(path, 0o400)
                    sha, size = self.helper.sha256_file(path)
                    records[fid] = {**entry, "sha256": sha, "size": size}
                    if sum(f["size"] for f in records.values()) > MAX_SCRATCH_BYTES:
                        raise ValueError("capture exceeded scratch limit")
                    captured.append(layer)
                    if layer == 27:
                        raise StopAfterCapture
                hook_handles.append(self.model.model.layers[layer].self_attn.register_forward_pre_hook(hook, with_kwargs=True))
            inputs = torch.tensor([prompt["input_ids"]], device="cuda", dtype=torch.long)
            with torch.inference_mode():
                try:
                    self.model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                               use_cache=False, logits_to_keep=1)
                except StopAfterCapture:
                    pass
                else:
                    raise ValueError("final capture hook did not stop the model")
            torch.cuda.synchronize()
            if captured != list(LAYERS):
                raise ValueError("capture dispatch/order differs")
            bundle = {"schema_version": 1, "protocol": PROTOCOL,
                "purpose": self.purpose, "partition": self.plan["partition"], "plan_sha256": canonical_sha(self.plan),
                "prompt_id": prompt["id"], "bindings": self.plan["bindings"],
                "fixtures": records, "scratch_bytes": sum(f["size"] for f in records.values()),
                "scratch_directory_identity": directory_identity(root.lstat()),
                "preparation_seconds": time.perf_counter() - started,
                "preparation_timed_in_score": False,
                "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
                "bitwise_matches_previous_capture": "unverified; fresh actual SHA authenticated"}
            validate_bundle(bundle, self.plan)
            write_private_json(root / "bundle.json", bundle)
            return bundle
        finally:
            for handle in hook_handles:
                handle.remove()
            # On failure retain the bounded partial directory for investigation;
            # do not reuse it as a complete bundle or capture another prompt.
            torch.cuda.empty_cache()

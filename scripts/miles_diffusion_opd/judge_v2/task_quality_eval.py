"""Judge-owned task-metric evaluation for the SD3.5-M DiffusionOPD task v2: OCR accuracy and PickScore of generated images.

The protocol:
diffusers ``StableDiffusion3Pipeline``, 512x512, 40-step flow-matching Euler ODE, the transformer and the three text
encoders in fp16, the VAE in fp32, the whole call under fp16 autocast, one image per prompt, the LoRA merged into the
transformer before sampling (an explicit ``W0 + (alpha/r) B @ A`` in fp64 over the
fp32 checkpoint weights, then cast to fp16, i.e. the same weights up to fp32 rounding of the merge).
Every image has its own fixed seed (a CPU generator seeded from the case seed and the prompt's index, so a result
does not depend on batching, sharding or GPU), and the classifier-free guidance scale is a parameter.

Scorers, both run in this process and never through candidate code:
- OCR: PaddleOCR (``use_angle_cls=False, lang="en"``, CPU, the image's PP-OCR models) on the rounded RGB uint8
  image; reward ``1 - min(Levenshtein(recognized, target), len(target)) / len(target)`` with spaces removed and lower
  case, 0 distance when the target is a substring (miles_diffusion rm_hub/ocr.py is the same code).
  The target is the text between the prompt's first pair of double quotes.
- PickScore: PickScore_v1 (CLIP-H) with the laion CLIP-ViT-H-14 processor in fp32; the reported value is the raw
  ``logit_scale * cos(text, image)``.

Rows are JSONL ``{input, metadata: {data_source}}``; ``--routes`` maps a data_source to its metric (ocr | pickscore).
``--shard k --num-shards n`` evaluates the rows whose index i has i % n == k; ``merge`` combines shard outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time

PEFT_PREFIX = "base_model.model."
METRICS = ("ocr", "pickscore")


# ------------------------------------------------------------------------------------------------ seeds and rows
def item_seed(case_seed: int, index: int) -> int:
    return int.from_bytes(hashlib.sha256(f"sd35m-mopd-v2:{case_seed}:{index}".encode()).digest()[:8], "little") >> 1


def load_rows(path: str, routes: dict[str, str]) -> list[dict]:
    rows = []
    for i, line in enumerate(open(path)):
        r = json.loads(line)
        source = r["metadata"]["data_source"]
        if routes.get(source) not in METRICS:
            raise ValueError(f"row {i}: data_source {source!r} has no metric in {routes}")
        rows.append({"index": i, "prompt": r["input"], "route": source, "metric": routes[source]})
    return rows


def ocr_target(prompt: str) -> str:
    return prompt.split('"')[1]


# ------------------------------------------------------------------------------------------------ OCR (CPU pool)
_OCR = None


def _ocr_init() -> None:
    global _OCR
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from paddleocr import PaddleOCR

    _OCR = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False, show_log=False)


def ocr_reward(args) -> float:
    """flow_grpo/ocr.py OcrScorer.__call__ for one (image, prompt)."""
    img, prompt = args
    from Levenshtein import distance

    target = ocr_target(prompt).replace(" ", "").lower()
    try:
        result = _OCR.ocr(img, cls=False)
        recognized = "".join([res[1][0] if res[1][1] > 0 else "" for res in result[0]]) if result[0] else ""
        recognized = recognized.replace(" ", "").lower()
        dist = 0 if target in recognized else distance(recognized, target)
        dist = min(dist, len(target))
    except Exception as e:  # noqa: BLE001 - flow_grpo gives the maximum penalty on any OCR failure
        print(f"OCR failed: {e}", file=sys.stderr)
        dist = len(target)
    return 1 - dist / len(target)


# ------------------------------------------------------------------------------------------------ models
class Generator:
    """The SD3.5-M pipeline with explicitly merged LoRAs (swap with ``set_lora``)."""

    def __init__(self, model_dir: str, device: str):
        import torch
        from diffusers import StableDiffusion3Pipeline

        self.torch = torch
        pipe = StableDiffusion3Pipeline.from_pretrained(model_dir, torch_dtype=torch.float32)
        self.modules = dict(pipe.transformer.named_modules())
        self.w0: dict[str, torch.Tensor] = {}  # fp32 checkpoint weights of the modules the current LoRA touched
        # fp32 checkpoint weights of every transformer Linear, kept on the host for merging
        self._fp32_state = {f"{n}.weight": m.weight.detach().clone() for n, m in self.modules.items()
                            if isinstance(m, torch.nn.Linear)}
        pipe.transformer.to(device, dtype=torch.float16)
        pipe.text_encoder.to(device, dtype=torch.float16)
        pipe.text_encoder_2.to(device, dtype=torch.float16)
        pipe.text_encoder_3.to(device, dtype=torch.float16)
        pipe.vae.to(device, dtype=torch.float32)
        pipe.set_progress_bar_config(disable=True)
        self.pipe, self.device = pipe, device

    def set_lora(self, path: str | None) -> dict:
        torch = self.torch
        with torch.no_grad():
            for name in list(self.w0):  # restore the checkpoint weights
                self.modules[name].weight.copy_(self.w0.pop(name).to(self.device, torch.float16))
            if not path:
                return {"lora": None}
            pairs, scale = load_lora(path)
            for name, (a, b) in pairs.items():
                if not isinstance(self.modules.get(name), torch.nn.Linear):
                    raise ValueError(f"LoRA module {name} is not in the SD3 transformer")
                w0 = self._fp32_state[f"{name}.weight"]
                self.w0[name] = w0
                delta = scale * (b.to(self.device, torch.float64) @ a.to(self.device, torch.float64))
                self.modules[name].weight.copy_((w0.to(self.device, torch.float64) + delta).to(torch.float16))
        return {"lora": path, "modules": len(pairs), "scale": scale}

    def generate(self, prompts: list[str], seeds: list[int], cfg: float, steps: int, res: int):
        """Images as uint8 HWC RGB numpy arrays (flow_grpo's rounding of the pipeline's [0, 1] output)."""
        torch = self.torch
        gens = [torch.Generator("cpu").manual_seed(s) for s in seeds]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            images = self.pipe(prompts, num_inference_steps=steps, guidance_scale=cfg, output_type="pt",
                               height=res, width=res, generator=gens)[0]
        images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy().transpose(0, 2, 3, 1)
        return list(images)


def load_lora(path: str):
    """PEFT adapter dir, or a bare PEFT-named safetensors (r 32, alpha 64: the task contract). fp32 (A, B) pairs."""
    import torch
    from safetensors.torch import load_file

    if os.path.isdir(path):
        cfg = json.load(open(os.path.join(path, "adapter_config.json")))
        if cfg.get("use_rslora") or cfg.get("use_dora") or cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
            raise ValueError(f"unsupported adapter config in {path}")
        r, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
        tensors = load_file(os.path.join(path, "adapter_model.safetensors"))
    else:
        r, alpha = 32, 64.0
        tensors = load_file(path)
    pairs: dict[str, list] = {}
    for key, value in tensors.items():
        name = key[len(PEFT_PREFIX):] if key.startswith(PEFT_PREFIX) else key
        for tag, slot in ((".lora_A.weight", 0), (".lora_B.weight", 1)):
            if name.endswith(tag):
                pairs.setdefault(name[: -len(tag)], [None, None])[slot] = value.to(torch.float32)
                break
        else:
            raise ValueError(f"unexpected LoRA key {key}")
    for name, (a, b) in pairs.items():
        if a is None or b is None or a.shape[0] != r or b.shape[1] != r:
            raise ValueError(f"bad LoRA pair for {name}")
    return {k: tuple(v) for k, v in pairs.items()}, alpha / r


class PickScorer:
    def __init__(self, model_dir: str, processor_dir: str, device: str):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch, self.device = torch, device
        self.processor = AutoProcessor.from_pretrained(processor_dir)
        self.model = AutoModel.from_pretrained(model_dir).eval().to(device, dtype=torch.float32)

    def __call__(self, prompts: list[str], images) -> list[float]:
        """flow_grpo/pickscore_scorer.py without the /26."""
        torch = self.torch
        from PIL import Image

        with torch.no_grad():
            pil = [Image.fromarray(im) for im in images]
            ii = self.processor(images=pil, padding=True, truncation=True, max_length=77, return_tensors="pt")
            ii = {k: v.to(self.device) for k, v in ii.items()}
            ti = self.processor(text=prompts, padding=True, truncation=True, max_length=77, return_tensors="pt")
            ti = {k: v.to(self.device) for k, v in ti.items()}
            ie = _features(self.model.get_image_features(**ii))
            ie = ie / ie.norm(p=2, dim=-1, keepdim=True)
            te = _features(self.model.get_text_features(**ti))
            te = te / te.norm(p=2, dim=-1, keepdim=True)
            scores = self.model.logit_scale.exp() * (te * ie).sum(dim=-1)
        return [float(s) for s in scores.cpu()]


def _features(x):
    # transformers >= 5 returns BaseModelOutputWithPooling from get_*_features
    return x if not hasattr(x, "pooler_output") else x.pooler_output


# ------------------------------------------------------------------------------------------------ evaluation
def summarize(items: list[dict]) -> dict:
    out = {}
    for route in sorted({it["route"] for it in items}):
        v = [it["score"] for it in items if it["route"] == route]
        mean = sum(v) / len(v)
        sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
        out[route] = {"metric": next(it["metric"] for it in items if it["route"] == route), "mean": mean,
                      "n": len(v), "sd": sd, "se": sd / math.sqrt(len(v))}
    return out


def run(args) -> dict:
    routes = json.loads(args.routes)
    rows = load_rows(args.prompts, routes)
    rows = [r for r in rows if r["index"] % args.num_shards == args.shard]
    configs = json.loads(args.configs) if args.configs else [
        {"name": "default", "lora": args.lora, "cfg": args.cfg, "case_seed": args.case_seed}]
    device = "cuda"
    gen = Generator(args.model, device)
    pick = PickScorer(args.pickscore_model, args.pickscore_processor, device) \
        if any(r["metric"] == "pickscore" for r in rows) else None
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(args.ocr_workers, initializer=_ocr_init) if any(r["metric"] == "ocr" for r in rows) else None
    results = []
    try:
        for c in configs:
            t0 = time.time()
            info = gen.set_lora(c.get("lora"))
            items, pending = [], []
            # one metric per batch (PickScore runs on the GPU right after generation; OCR goes to the CPU pool)
            for metric in METRICS:
                todo = [r for r in rows if r["metric"] == metric]
                for lo in range(0, len(todo), args.batch):
                    chunk = todo[lo:lo + args.batch]
                    seeds = [item_seed(c["case_seed"], r["index"]) for r in chunk]
                    images = gen.generate([r["prompt"] for r in chunk], seeds, c["cfg"], args.steps, args.res)
                    if args.save_images:
                        from PIL import Image

                        d = os.path.join(args.save_images, c["name"])
                        os.makedirs(d, exist_ok=True)
                        for r, im in zip(chunk, images):
                            Image.fromarray(im).save(os.path.join(d, f"{r['index']:05d}.png"))
                    base = [{"index": r["index"], "route": r["route"], "metric": metric, "seed": s}
                            for r, s in zip(chunk, seeds)]
                    if metric == "pickscore":
                        for b, s in zip(base, pick([r["prompt"] for r in chunk], images)):
                            items.append(dict(b, score=s))
                    else:
                        pending.append((base, pool.map_async(ocr_reward, [(im, r["prompt"]) for r, im in zip(chunk, images)])))
            for base, fut in pending:
                for b, s in zip(base, fut.get()):
                    items.append(dict(b, score=s))
            items.sort(key=lambda it: it["index"])
            results.append({**c, **info, "items": items, "per_route": summarize(items),
                            "seconds": round(time.time() - t0, 1)})
            print(json.dumps({k: v for k, v in results[-1].items() if k != "items"}), flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return {"prompts": args.prompts, "shard": args.shard, "num_shards": args.num_shards, "steps": args.steps,
            "res": args.res, "routes": routes, "results": results}


def merge(paths: list[str]) -> dict:
    """Combine shard outputs (same configs, disjoint indices) into whole-set results."""
    parts = [json.load(open(p)) for p in paths]
    n = parts[0]["num_shards"]
    if sorted(p["shard"] for p in parts) != list(range(n)):
        raise ValueError(f"shards {[p['shard'] for p in parts]} of {n}")
    out = {k: parts[0][k] for k in ("prompts", "steps", "res", "routes")}
    out["results"] = []
    for k, c in enumerate(parts[0]["results"]):
        items = sorted((it for p in parts for it in p["results"][k]["items"]), key=lambda it: it["index"])
        if len({it["index"] for it in items}) != len(items):
            raise ValueError("overlapping shards")
        out["results"].append({**{x: c[x] for x in c if x not in ("items", "per_route", "seconds")},
                               "items": items, "per_route": summarize(items),
                               "seconds_per_shard": [p["results"][k]["seconds"] for p in parts]})
    return out


def _finite_or_none(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


QUALITY_UNIT = "mean per-route closure of the original-to-teacher gap in task metric"


def metric_v3(student: dict, frozen: dict, kl_gate_passed: bool, T, gate_min: float, scale_seconds: float = 3600,
              unit: str = QUALITY_UNIT) -> dict:
    """Score of a candidate whose correctness checks all passed (metric v3; the Judge and public perf use it).

    `student`: route -> the final student's mean task metric on that route's held-out prompts (OCR accuracy, raw
    PickScore). `frozen`: route -> {"base": B, "teacher": T_r}, the original model and the route's teacher under the same
    protocol, measured once and frozen. closure_r = (S_r - B_r) / (T_r - B_r) (0: no better than the original model,
    1: the teacher's level; not capped); quality_score = mean over the routes. The quality gate passes iff the per-route
    KL gate passed and every closure_r >= gate_min; then score = quality_score * scale_seconds / T, else 0 whatever T is.
    `score` is None when the gate passes but T is not a positive finite number of seconds. A student value that is not
    finite fails the gate. A frozen pair with T_r - B_r not positive and finite, or missing routes, is an evaluator error
    (ValueError)."""
    if set(student) != set(frozen) or not frozen:
        raise ValueError(f"routes differ: student {sorted(student)}, frozen {sorted(frozen)}")
    closures = {}
    for r, f in sorted(frozen.items()):
        gap = float(f["teacher"]) - float(f["base"])
        if not (math.isfinite(gap) and gap > 0):
            raise ValueError(f"route {r}: teacher - base = {gap} is not a positive finite gap")
        s = float(student[r]) if isinstance(student[r], (int, float)) and not isinstance(student[r], bool) else math.nan
        closures[r] = (s - float(f["base"])) / gap if math.isfinite(s) else None
    measured = None not in closures.values()
    q = sum(closures.values()) / len(closures) if measured else None
    routes_ok = measured and all(c >= gate_min for c in closures.values())
    passed = bool(kl_gate_passed) and routes_ok
    T = float(T) if isinstance(T, (int, float)) and not isinstance(T, bool) else None
    usable = T is not None and math.isfinite(T) and T > 0
    score = (q * scale_seconds / T if usable else None) if passed else 0.0
    if not kl_gate_passed:
        summary = "quality gate failed: the per-route teacher-student KL gate failed: score 0"
    elif not measured:
        summary = f"quality gate failed: a route's task metric is not finite ({student}): score 0"
    elif not routes_ok:
        low = {r: round(c, 4) for r, c in closures.items() if c < gate_min}
        summary = f"quality gate failed: closure {low} < {gate_min:g}: score 0"
    elif not usable:
        summary = f"no usable timed window T ({T}): not scored"
    else:
        summary = f"score {score:.3f} = quality {q:.4f} x {scale_seconds:g} / T {T:.1f} s"
    return {"closure_per_route": closures, "student_metric_per_route": {r: _finite_or_none(student[r]) for r in sorted(student)},
            "frozen_per_route": {r: dict(frozen[r]) for r in sorted(frozen)}, "quality_score": q,
            "quality_unit": unit, "closure_gate_min": gate_min, "kl_gate_passed": bool(kl_gate_passed),
            "quality_gate_passed": passed, "T_seconds": T if T is not None and math.isfinite(T) else None,
            "score": score,
            "score_formula": f"mean_r closure_r * {scale_seconds:g} / T if correct, the per-route KL gate passes and "
                             f"every closure_r >= {float(gate_min)!r} else 0",
            "summary": summary}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--model", default="/assets/sd35m/stable-diffusion-3.5-medium")
    e.add_argument("--pickscore-model", default="/assets/scorers/PickScore_v1")
    e.add_argument("--pickscore-processor", default="/assets/scorers/CLIP-ViT-H-14-processor")
    e.add_argument("--prompts", required=True)
    e.add_argument("--routes", default='{"ocr": "ocr", "pickscore": "pickscore"}')
    e.add_argument("--lora", help="PEFT adapter dir or PEFT-named safetensors; omit for the original model")
    e.add_argument("--cfg", type=float, default=1.0)
    e.add_argument("--case-seed", type=int, default=0)
    e.add_argument("--configs", help='JSON list [{"name", "lora", "cfg", "case_seed"}] evaluated in one process')
    e.add_argument("--steps", type=int, default=40)
    e.add_argument("--res", type=int, default=512)
    e.add_argument("--batch", type=int, default=16)
    e.add_argument("--shard", type=int, default=0)
    e.add_argument("--num-shards", type=int, default=1)
    e.add_argument("--ocr-workers", type=int, default=8)
    e.add_argument("--save-images")
    e.add_argument("--out", required=True)
    m = sub.add_parser("merge")
    m.add_argument("parts", nargs="+")
    m.add_argument("--out", required=True)
    args = ap.parse_args()
    res = run(args) if args.cmd == "eval" else merge(args.parts)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, args.out)
    for r in res["results"]:
        print(r["name"], json.dumps(r["per_route"]))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CPU tests for the v2 provenance hook, provenance checker and oracle modes.

No GPU and no Miles: fake ``megatron.core`` / ``sglang`` weight-updater / ``miles`` actor
modules are written to a temp dir and imported under the real hook, a short training run
with publications is simulated, and the checker must pass on it and catch tampered
variants, including a final checkpoint that keeps the fingerprint rows but not the rest of
the last step's weights (M6, the hooks' full digests). A tiny Qwen3 (the pinned tokenizer, 2 layers) checks the oracle's conversion of
SGLang's fused served weights and its gradient comparison against an independent float64
gradient, including a data-parallel split of the gradient record.

  python tests/test_v2_provenance.py --student-tokenizer /models/qwen3-1.7b/snapshots/<rev>
"""

import argparse
import importlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROUNDS = 6
LR = 1e-2
FAILS = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAILS.append(msg)


FAKE = {
    "megatron/__init__.py": "",
    "megatron/core/__init__.py": "",
    "megatron/core/models/__init__.py": "",
    "megatron/core/models/gpt/__init__.py": "",
    "megatron/core/models/gpt/gpt_model.py": """
        import torch
        class _Emb(torch.nn.Module):
            def __init__(self, V, H):
                super().__init__()
                self.word_embeddings = torch.nn.Embedding(V, H).to(torch.bfloat16)  # bf16 like Megatron's params
        class _Cfg:
            bf16 = True
            params_dtype = "torch.bfloat16"
            attention_backend = "flash"
        class GPTModel(torch.nn.Module):
            def __init__(self, V, H):
                super().__init__()
                self.embedding = _Emb(V, H)
                self.config = _Cfg()
                self.vocab_size, self.pre_process, self.share_embeddings_and_output_weights = V, True, True
            def forward(self, ids):
                h = self.embedding.word_embeddings(ids)
                return h @ self.embedding.word_embeddings.weight.T
        """,
    "megatron/core/optimizer/__init__.py": "from .optimizer import MegatronOptimizer, MixedPrecisionOptimizer\n",
    "megatron/core/optimizer/optimizer.py": """
        import torch
        class _Cfg:
            optimizer, decoupled_weight_decay, clip_grad = "adam", True, 1.0
            adam_beta1, adam_beta2, adam_eps, lr, weight_decay = 0.9, 0.999, 1e-8, 5e-6, 0.01
        class MegatronOptimizer:
            pass
        class MixedPrecisionOptimizer(MegatronOptimizer):
            def __init__(self, params, lr):
                self.optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
                self.config = _Cfg()
            @property
            def param_groups(self):
                return self.optimizer.param_groups
            def step(self):
                ps = [p for g in self.optimizer.param_groups for p in g["params"]]
                for p in ps:
                    p.grad = p.main_grad.clone()
                gn = float(torch.sqrt(sum((p.grad ** 2).sum() for p in ps)))
                self.optimizer.step()
                return True, gn, 0
        """,
    "miles/__init__.py": "",
    "miles/backends/__init__.py": "",
    "miles/backends/megatron_utils/__init__.py": "",
    "miles/backends/megatron_utils/actor.py": """
        import torch
        class MegatronTrainRayActor:
            def __init__(self, model, opt):
                self.model, self.opt = model, opt
            def train(self, ids):
                w = self.model.embedding.word_embeddings.weight
                loss = self.model(ids).float().logsumexp(-1).mean()
                w.main_grad = torch.autograd.grad(loss, w)[0]
                return self.opt.step()
        """,
    "sglang/__init__.py": "",
    "sglang/srt/__init__.py": "",
    "sglang/srt/managers/__init__.py": "",
    "sglang/srt/managers/scheduler_components/__init__.py": "",
    "sglang/srt/managers/scheduler_components/weight_updater.py": """
        class _Out:
            def __init__(self, ok=True):
                self.success = ok
        class _Args:
            def __init__(self, port):
                self.port, self.model_path = port, "/student"
        class _Sched:
            def __init__(self, port):
                self.server_args = _Args(port)
        class _PS:
            tp_size, tp_rank = 1, 0
        class Runner:
            def __init__(self, model):
                self.model, self.ps = model, _PS()
        class SchedulerWeightUpdaterManager:
            def __init__(self, runner, port):
                self.runner, self.scheduler = runner, _Sched(port)
            def get_model_runners(self, selector="all"):
                return [("", self.runner)]
            def begin_weight_update(self, req=None):
                return _Out()
            def end_weight_update(self, req=None):
                return _Out()
            def update_weights_from_tensor(self, named):
                params = dict(self.runner.model.named_parameters())
                for n, t in named:
                    params[n].data.copy_(t)
                return _Out()
        """,
}


def write_fake(root):
    for rel, src in FAKE.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(src))


def save_embed(path, w):
    import torch
    from safetensors.torch import save_file

    path.mkdir(parents=True, exist_ok=True)
    save_file({"model.embed_tokens.weight": w.detach().to(torch.bfloat16).contiguous()}, str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps({"num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 16,
                                                  "hidden_size": 16}))


def simulate(tmp, audit, *, stale_at=None, edit_at=None):
    """A ROUNDS-step run under the hook: initial push, then per step a forward, a Megatron step
    driven by the fake Miles actor, and one publication to the fake engine. `edit_at`: after that
    step, a weight outside the fingerprint rows is changed before the publication and the next step."""
    import torch

    torch.manual_seed(0)
    gm = importlib.import_module("megatron.core.models.gpt.gpt_model")
    mo = importlib.import_module("megatron.core.optimizer")
    wu = importlib.import_module("sglang.srt.managers.scheduler_components.weight_updater")
    actor_mod = importlib.import_module("miles.backends.megatron_utils.actor")
    model = gm.GPTModel(256, 16)
    save_embed(tmp / "initial_hf", model.embedding.word_embeddings.weight)
    opt = mo.MixedPrecisionOptimizer(model.parameters(), lr=LR)
    actor = actor_mod.MegatronTrainRayActor(model, opt)
    served = torch.nn.Module()
    served.model = torch.nn.Module()
    served.model.embed_tokens = torch.nn.Embedding(256, 16).to(torch.bfloat16)
    mgr = wu.SchedulerWeightUpdaterManager(wu.Runner(served), port=30000)

    def publish(skip=False):
        mgr.begin_weight_update()
        if not skip:
            mgr.update_weights_from_tensor([("model.embed_tokens.weight", model.embedding.word_embeddings.weight.detach().to(torch.bfloat16))])
        mgr.end_weight_update()

    publish()  # initial push
    audit_recs = [{"kind": "hook_installed", "module": "sglang.srt.managers.tokenizer_manager", "pid": os.getpid(),
                   "parent_title": "ray::SGLangEngine.init", "t": time.time()}]
    time.sleep(0.01)
    audit_recs.append({"kind": "dispatch", "rid": "r0", "model_path": "/student", "pid": os.getpid(), "t": time.time(),
                       "sampling": {"max_new_tokens": 8}})
    time.sleep(0.01)
    ids = torch.randint(0, 256, (4, 12))
    for k in range(ROUNDS):
        with torch.no_grad():
            model(ids)  # a log-prob pass: no fingerprint
        actor.train(ids)
        if k == edit_at:
            with torch.no_grad():
                model.embedding.word_embeddings.weight[200] += 1
        publish(skip=(k == stale_at))
    save_embed(tmp / "final_hf", model.embedding.word_embeddings.weight)
    # the same fingerprint rows, another row changed after the last step (untimed work on the weights)
    w = model.embedding.word_embeddings.weight.detach().clone()
    w[200] += 1
    save_embed(tmp / "final_hf_changed", w)
    with open(audit / "audit-test.jsonl", "a") as fh:
        for r in audit_recs:
            fh.write(json.dumps(r) + "\n")
    (tmp / "contract.json").write_text(json.dumps({"optimizer": {"lr": LR, "betas": [0.9, 0.999], "eps": 1e-8,
                                                                "weight_decay": 0.01, "clip_grad": 1.0}}))


def run_checker(tmp, audit, *extra, pinned=None, steps=ROUNDS, final="final_hf"):
    cmd = [sys.executable, str(HERE / "audit/opcd_provenance_check.py"), "--audit-dir", str(audit), "--mode", "run",
           "--steps", str(steps), "--contract", str(tmp / "contract.json"), "--initial-hf", str(tmp / "initial_hf"),
           "--final-hf", str(tmp / final), "--student-model-path", "/student", "--pinned-megatron", str(pinned or tmp), *extra]
    p = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        print(p.stdout[-2000:], p.stderr[-2000:])
        return {"passed": None}


def child(tmp, audit, stale_at, edit_at):
    """The simulation, in a fresh interpreter started with the audit environment (as the
    evaluator starts task_entry.py): the hook installs itself when first imported, from
    site-packages via its .pth where the image has it, else from this checkout."""
    sys.path.insert(1, str(HERE / "audit"))
    sys.path.insert(0, str(tmp))
    import opcd_provenance_hook

    print("hook module:", opcd_provenance_hook.__file__, flush=True)
    simulate(tmp, audit, stale_at=None if stale_at < 0 else stale_at, edit_at=None if edit_at < 0 else edit_at)


def provenance_tests(workdir):
    results = {}
    for name, stale, edit in (("clean", -1, -1), ("stale", 3, -1), ("edited", -1, 2)):
        tmp = workdir / name
        audit = tmp / "audit"
        audit.mkdir(parents=True)
        write_fake(tmp)
        env = dict(os.environ, OPCD_AUDIT_DIR=str(audit), OPCD_AUDIT_ROUNDS=str(ROUNDS))
        p = subprocess.run([sys.executable, __file__, "--child", str(tmp), str(audit), str(stale), str(edit)],
                           capture_output=True, text=True, env=env)
        print("  " + next((x for x in p.stdout.splitlines() if x.startswith("hook module")), "hook module: ?"))
        if p.returncode != 0:
            print(p.stdout[-3000:], p.stderr[-3000:])
        check(p.returncode == 0, f"simulation {name} ran under the hook")
        results[name] = (tmp, audit)

    tmp, audit = results["clean"]
    rep = run_checker(tmp, audit)
    check(rep.get("passed") is True, f"clean run passes the provenance check {rep.get('issues')}")
    check(len(rep.get("snapshot_steps") or []) >= 1, f"clean run has usable snapshots {rep.get('snapshot_steps')}")
    check(len(rep.get("exclude_intervals") or []) == len(rep.get("snapshots") or []) + 3 >= 4,
          "snapshot and digest intervals reported (the trainer's, the engine's after its last two publications)")
    check(all(x["credited_seconds"] == x["seconds"] for x in rep.get("evaluator_intervals") or [{}]),
          f"the simulation's short intervals leave T whole {rep.get('evaluator_intervals')}")
    check(rep.get("trainer_digest_checked") is True and rep.get("final_digest_engines") == 1, "M6 digests checked")
    check(sorted(rep.get("grad_records") or {}) == [str(k) for k in range(ROUNDS)], "a gradient record per step")
    check(rep.get("publications_checked") == ROUNDS, "every publication checked")

    rep = run_checker(tmp, audit, final="initial_hf")
    check("M3" in (rep.get("issues") or {}), "final checkpoint != last step is caught (M3)")
    rep = run_checker(tmp, audit, final="final_hf_changed")
    check(set(rep.get("issues") or {}) == {"M6"},
          f"a final checkpoint changed beyond the fingerprint rows is caught by M6 only {rep.get('issues')}")
    rep = run_checker(tmp, audit, steps=ROUNDS + 1)
    check("M1" in (rep.get("issues") or {}), "wrong step count is caught (M1)")
    rep = run_checker(tmp, audit, pinned="/nonexistent")
    check("M1" in (rep.get("issues") or {}), "Megatron outside the pinned tree is caught (M1)")
    rep = run_checker(*results["stale"])
    check("M3" in (rep.get("issues") or {}), "a publication of stale weights is caught (M3)")
    rep = run_checker(*results["edited"])
    check(set(rep.get("issues") or {}) == {"M3"} and any("changed between optimizer steps 2 and 3" in x
                                                         for x in rep["issues"]["M3"]),
          f"a weight edited between two steps (outside the fingerprint rows) is caught by the state chain {rep.get('issues')}")

    def tampered(mutate, tag):
        t = workdir / f"tamper-{tag}"
        shutil.copytree(tmp, t)
        for f in (t / "audit").glob("*.jsonl"):
            lines = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
            f.write_text("".join(json.dumps(mutate(r)) + "\n" for r in lines))
        return run_checker(t, t / "audit", pinned=tmp)

    def lr(r):
        if r.get("kind") == "megatron_step":
            for g in r["param_groups"]:
                g["lr"] = 1e-5
        return r
    check("M2" in (tampered(lr, "lr").get("issues") or {}), "wrong lr is caught (M2)")

    def no_actor(r):
        if r.get("kind") == "megatron_step":
            r["miles_stack"] = []
        return r
    check("M1" in (tampered(no_actor, "actor").get("issues") or {}), "a step outside Miles' actor is caught (M1)")

    def parent(r):
        if r.get("module"):
            r["parent_title"] = "python3 my_trainer.py"
        return r
    check("M5" in (tampered(parent, "parent").get("issues") or {}), "an engine not started by Ray is caught (M5)")

    def no_megatron(r):
        return {"kind": "noise"} if r.get("kind", "").startswith("megatron") else r
    check("M1" in (tampered(no_megatron, "bypass").get("issues") or {}), "no Megatron evidence at all is caught (M1)")


def oracle_tests(workdir, tok_dir):
    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    sys.path.insert(0, str(HERE))
    import opcd_oracle as orc

    tok = AutoTokenizer.from_pretrained(tok_dir)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=True,
                      max_position_embeddings=512)
    dirs = {}
    for name, seed in (("student", 1), ("teacher", 2)):
        torch.manual_seed(seed)
        m = Qwen3ForCausalLM(cfg).to(torch.bfloat16).to(torch.float32)  # bf16-representable weights
        d = workdir / name
        m.save_pretrained(d, safe_serialization=True)
        for f in Path(tok_dir).iterdir():
            if f.name.startswith(("tokenizer", "vocab", "merges")):
                shutil.copy(f.resolve(), d / f.name)
        dirs[name] = (d, m)
    student, teacher = dirs["student"][1].eval(), dirs["teacher"][1].eval()

    # SGLang's served layout: fused qkv_proj and gate_up_proj, tied head not stored, bf16
    q, kv = 4 * 16, 2 * 16
    sd, fused = student.state_dict(), {}
    for k, v in sd.items():
        if k == "lm_head.weight":
            continue
        if ".q_proj." in k:
            b = k.replace("q_proj.weight", "")
            fused[b + "qkv_proj.weight"] = torch.cat([sd[b + "q_proj.weight"], sd[b + "k_proj.weight"], sd[b + "v_proj.weight"]])
        elif ".gate_proj." in k:
            b = k.replace("gate_proj.weight", "")
            fused[b + "gate_up_proj.weight"] = torch.cat([sd[b + "gate_proj.weight"], sd[b + "up_proj.weight"]])
        elif ".k_proj." in k or ".v_proj." in k or ".up_proj." in k:
            continue
        else:
            fused[k] = v
    snap = workdir / "snap.safetensors"
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in fused.items()}, str(snap))
    fresh = Qwen3ForCausalLM(cfg)
    orc.load_sglang_snapshot(fresh, str(snap))
    same = all(torch.equal(fresh.state_dict()[k], v) for k, v in student.state_dict().items())
    check(same, "SGLang fused snapshot converts back to the exact HF weights")

    # a step dump: 4 rows, K=8, teacher scored with the context prefix
    context = "You are a careful medical assistant."
    (workdir / "context.txt").write_text(context)
    prefix = orc.context_prefix_ids(tok, context, {"enable_thinking": False})
    K, rows = 8, []
    g = torch.Generator().manual_seed(3)
    for i in range(4):
        text = tok.apply_chat_template([{"role": "user", "content": f"question {i}: which nerve?"}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        p_ids = tok.encode(text, add_special_tokens=False)
        r_ids = torch.randint(0, 5000, (5 + i,), generator=g).tolist()
        rows.append((torch.tensor(p_ids + r_ids), len(r_ids)))
    ids_l, old_l, lq_l = [], [], []
    with torch.no_grad():
        for seq, R in rows:
            T, P = seq.numel(), seq.numel() - R
            lp = torch.log_softmax(student(input_ids=seq[None]).logits[0, P - 1:T - 1].float(), -1)
            tseq = torch.cat([torch.tensor(prefix), seq])
            lq = torch.log_softmax(teacher(input_ids=tseq[None]).logits[0, len(prefix) + P - 1:len(prefix) + T - 1].float(), -1)
            top = torch.topk(lp, K, -1)
            ids_l.append(top.indices)
            old_l.append(top.values)
            lq_l.append(lq.gather(-1, top.indices))
    masks = [torch.ones(R) for _, R in rows]
    n_valid = sum(float(m.sum()) for m in masks)

    def reference_grad(surrogate=False):
        """float64 gradient of the specified loss (or a sampled-token surrogate) by separate code."""
        m64 = Qwen3ForCausalLM(cfg).to(torch.float64)
        m64.load_state_dict({k: v.to(torch.float64) for k, v in student.state_dict().items()})
        total = 0.0
        for (seq, R), ids, lq in zip(rows, ids_l, lq_l):
            T, P = seq.numel(), seq.numel() - R
            lp_all = torch.log_softmax(m64(input_ids=seq[None]).logits[0, P - 1:T - 1], -1)
            if surrogate:
                tok_lp = lp_all.gather(-1, seq[P:T, None]).squeeze(-1)
                total = total + (tok_lp * (tok_lp.detach() - lq[:, 0].double())).sum() / n_valid
            else:
                lp = lp_all.gather(-1, ids)
                total = total + (lp.exp() * (lp - lq.double())).sum() / n_valid
        total.backward()
        w = m64.get_input_embeddings().weight.grad
        norm = float(torch.sqrt(sum((p.grad ** 2).sum() for p in m64.parameters() if p.grad is not None)))
        return w[:orc.FP_ROWS].detach().to(torch.float32), norm, float(total)

    grad, norm, loss = reference_grad()
    width = grad.shape[1]
    split = 37 * width + 5  # a data-parallel shard boundary inside a row
    for name, owned in (("r0", [0, split]), ("r1", [split, 10 ** 9])):
        noisy = grad.clone().reshape(-1)
        lo, hi = owned[0], min(owned[1], noisy.numel())
        mask = torch.ones_like(noisy, dtype=torch.bool)
        mask[lo:hi] = False
        noisy[mask] = 1e3  # values outside the owned range are not reduced: must be ignored
        torch.save({"rows": noisy.reshape(grad.shape), "owned": owned, "row_width": width, "vocab_rows": cfg.vocab_size},
                   workdir / f"grad-{name}.pt")
    dump = workdir / "step_001_rank_0.pt"
    torch.save({"step": 1, "loss": loss, "tokens": [s for s, _ in rows], "response_lengths": [R for _, R in rows],
                "loss_masks": masks, "opcd_topk_ids": ids_l, "opcd_old_topk_logprobs": old_l,
                "opcd_teacher_topk_logprobs": lq_l}, dump)

    def oracle(*extra):
        out = workdir / f"oracle-{random.randrange(1 << 30)}.json"
        cmd = [sys.executable, str(HERE / "opcd_oracle.py"), "--dump", str(dump), "--student", str(dirs["student"][0]),
               "--teacher", str(dirs["teacher"][0]), "--context-file", str(workdir / "context.txt"), "--top-k", str(K),
               "--device", "cpu", "--out", str(out), *extra]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            print(p.stdout[-2000:], p.stderr[-3000:])
        return json.loads(out.read_text()) if out.is_file() else {}

    recs = f"{workdir / 'grad-r0.pt'},{workdir / 'grad-r1.pt'}"
    rep = oracle("--student-snapshot", str(snap), "--grad-records", recs, "--grad-norm", repr(norm))
    check((rep.get("grad_slice_cosine") or 0) > 0.99999, f"gradient cosine vs float64 reference {rep.get('grad_slice_cosine')}")
    check(abs((rep.get("grad_slice_norm_ratio") or 0) - 1) < 1e-4, f"gradient slice ratio {rep.get('grad_slice_norm_ratio')}")
    check(abs((rep.get("grad_norm_ratio") or 0) - 1) < 1e-4, f"grad norm ratio {rep.get('grad_norm_ratio')}")
    check(rep.get("grad_valid_elements") == orc.FP_ROWS * width, "stitched records cover every row element")
    check(abs(rep.get("logged_vs_candidate_set_abs_err", 1)) < 1e-4, "logged loss matches the candidate-set loss")
    check((rep.get("candidate_set_overlap_mean") or 0) >= 0.99,
          f"top-k set from the snapshot matches the dump {rep.get('candidate_set_overlap_mean')}")

    sg, snorm, _ = reference_grad(surrogate=True)
    torch.save({"rows": sg, "owned": None, "row_width": width}, workdir / "grad-surrogate.pt")
    rep = oracle("--student-snapshot", str(snap), "--grad-records", str(workdir / "grad-surrogate.pt"), "--grad-norm", repr(snorm))
    ok, why = __import__("opcd_checks").oracle_verdict(rep, grad={})
    check(not ok and any("gradient" in w for w in why), f"a surrogate (sampled-token) gradient is rejected: cos {rep.get('grad_slice_cosine')}")

    rep = oracle("--teacher-only", "--sample-rows", "2", "--row-seed", "5")
    check(rep.get("samples") == 2 and (rep.get("candidate_teacher_logq_p_weighted_abs_err") or 1) < 1e-4,
          f"teacher-only spot check on 2 rows {rep.get('candidate_teacher_logq_p_weighted_abs_err')}")
    (workdir / "context.txt").write_text("A different system prompt entirely.")
    rep = oracle("--teacher-only", "--sample-rows", "4", "--row-seed", "5")
    check((rep.get("candidate_teacher_logq_p_weighted_abs_err") or 0) > 1e-3,
          f"teacher scored without the right context is visible {rep.get('candidate_teacher_logq_p_weighted_abs_err')}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-tokenizer", required=True)
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as t:
        work = Path(t)
        provenance_tests(work / "prov")
        (work / "oracle").mkdir()
        oracle_tests(work / "oracle", a.student_tokenizer)
    print(f"\n{len(FAILS)} failure(s)")
    raise SystemExit(1 if FAILS else 0)


if __name__ == "__main__":
    main()

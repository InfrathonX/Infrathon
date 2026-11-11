#!/usr/bin/env python3
"""Planted-bug submissions for Judge calibration (G7): copies of the reference repo with one targeted change each.

R6-class plants keep every self-reported artifact consistent (per-pair logs, π_old, R4 self-consistency, the 1-step
probe) and only corrupt the cumulative update, so only the trajectory gate can catch them:
  skip_backward   steps 2-4: forward (logged) but no backward / optimizer step
  extrapolate     steps 2-4 skipped; θ := θ0 + 4·(θ1 − θ0) at the end
  noise           steps 2-4 skipped; θ += N(0, 2e-5) at the end
  sub50           backward only for pairs of even sample_index (half the samples)
  one_window      backward only for pairs with odd sde_step (exactly one of each sample's two window steps)
  signsgd         θ -= lr·(sign(g) + wd·θ) instead of AdamW (identical to Adam at step 1)
  lr2             lr × 2 (variant_md_args) — J1 magnitude should catch it at the probe
  eng_*           Dance transition in the engine overlay: η instead of η², score-term sign, no score term, one
                  SDE-noise seed for every sample (J4 should catch; run J0,RUN)
  tr_*            the same math errors in the trainer's DanceSdeStepBackend (J1/R3 should catch; run SERVE)
Plants act only in calls with >1 optimizer step (replays), so the 1-step probe (J1) sees honest behaviour: this
measures what R6 alone catches (a cheater can tell the probe from a replay by its step count). signsgd equals Adam at
step 1 anyway; lr2 is the exception and is meant for J1.

  make_plants.py --src <reference repo> --out <dir> [--only name,...]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ACTOR = "miles/backends/fsdp_utils/actor.py"

LOOP_HEAD = """            for optim_step_idx, microbatch_ranges in enumerate(microbatch_schedule):
                self.optimizer.zero_grad(set_to_none=True)
"""
BACKWARD = """                    if not self.args.debug_skip_optimizer_step:
                        # ShardedGradScaler keeps fp16 policy grads from underflowing
                        # (required for SD3.5 fp16 forward); no-op for bf16/fp32.
                        self.scaler.scale(loss_sum / float(num_local_pairs)).backward()
"""
STEP_GUARD = """                if not self.args.debug_skip_optimizer_step:
                    self.scaler.unscale_(self.optimizer)
"""
OPT_STEP = """                    self.scaler.step(self.optimizer)
"""
LOOP_TAIL = """        from miles.utils import pair_log

        pair_log.flush(self.args, rollout_id, dist.get_rank())
"""


def only_first_step(s: str) -> str:
    s = s.replace(BACKWARD, BACKWARD.replace("if not self.args.debug_skip_optimizer_step:",
                                             "if not self.args.debug_skip_optimizer_step and optim_step_idx == 0:"))
    return s.replace(STEP_GUARD, STEP_GUARD.replace("if not self.args.debug_skip_optimizer_step:",
                                                    "if not self.args.debug_skip_optimizer_step and optim_step_idx == 0:"))


def snapshot_theta0(s: str) -> str:
    return s.replace(LOOP_HEAD, LOOP_HEAD + """                if optim_step_idx == 0:  # PLANT
                    _plant_theta0 = [p.detach().clone() for p in self.model.parameters()]
""")


def at_end(s: str, body: str) -> str:
    # only when the call has >1 optimizer step: the 1-step probe (J1) sees the honest update
    return s.replace(LOOP_TAIL, "        if len(microbatch_schedule) > 1:  # PLANT\n            with torch.no_grad():\n"
                     + body + "\n" + LOOP_TAIL)


def skip_backward(s):
    return only_first_step(s)


def extrapolate(s):
    s = snapshot_theta0(only_first_step(s))
    return at_end(s, """                for _p, _p0 in zip(self.model.parameters(), _plant_theta0):
                    _p.add_(3.0 * (_p - _p0))""")


def noise(s):
    s = only_first_step(s)
    return at_end(s, """                for _p in self.model.parameters():
                    _p.add_(torch.randn_like(_p) * 2e-5)""")


def _backward_if(cond: str):
    # zero-weight the excluded pairs instead of skipping their backward: skipping makes the FSDP collectives differ
    # between ranks and deadlocks (one_window 2026-09-24: NCCL allgather timeout); the gradient is identical
    def f(s):
        return s.replace(BACKWARD, BACKWARD.replace(
            "self.scaler.scale(loss_sum / float(num_local_pairs)).backward()",
            "self.scaler.scale(loss_sum * (1.0 if (len(microbatch_schedule) == 1 or "
            f"all({cond} for pair in chunk)) else 0.0) / float(num_local_pairs)).backward()  # PLANT"))
    return f


sub50 = _backward_if('int(pair["sample_index"]) % 2 == 0')
one_window = _backward_if('int(pair.get("sde_step", 1)) % 2 == 1')


def signsgd(s):
    return s.replace(OPT_STEP, """                    with torch.no_grad():  # PLANT: SignSGD with decoupled weight decay
                        for _g in self.optimizer.param_groups:
                            for _p in _g["params"]:
                                if _p.grad is not None:
                                    _p.mul_(1 - _g["lr"] * _g["weight_decay"]).add_(torch.sign(_p.grad), alpha=-_g["lr"])
""")


PLANTS = {"skip_backward": skip_backward, "extrapolate": extrapolate, "noise": noise, "sub50": sub50,
          "one_window": one_window, "signsgd": signsgd, "lr2": None}

# Transition-math plants (other files). Engine plants (J4 should catch; run J0,RUN): the Dance branch of the engine
# overlay. Trainer plants (J1/R3 should catch; run SERVE): MD's DanceSdeStepBackend.
ENGINE = "engine_overlay/runtime/post_training/scheduler_rl_mixin.py"
TRAINER = "miles/backends/fsdp_utils/sde_step_backend.py"
FILE_PLANTS = {
    "eng_eta_lin": (ENGINE, "                * noise_level**2\n", "                * noise_level\n"),
    "eng_score_sign": (ENGINE, "prev_sample_mean = sample + dt * model_output + score_term * dt",
                       "prev_sample_mean = sample + dt * model_output - score_term * dt"),
    "eng_no_score": (ENGINE, "prev_sample_mean = sample + dt * model_output + score_term * dt",
                     "prev_sample_mean = sample + dt * model_output"),
    "eng_shared_noise": (ENGINE, ").manual_seed(int(sde_noise_seed))", ").manual_seed(0)"),
    "tr_eta_lin": (TRAINER, "score_term = 0.5 * noise_level**2 * (sample", "score_term = 0.5 * noise_level * (sample"),
    "tr_score_sign": (TRAINER, "prev_mean = sample + dt * model_output + score_term * dt",
                      "prev_mean = sample + dt * model_output - score_term * dt"),
    "tr_no_score": (TRAINER, "prev_mean = sample + dt * model_output + score_term * dt",
                    "prev_mean = sample + dt * model_output"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    names = a.only.split(",") if a.only else list(PLANTS) + list(FILE_PLANTS)
    src_actor = (a.src / ACTOR).read_text()
    for anchor in (LOOP_HEAD, BACKWARD, STEP_GUARD, OPT_STEP, LOOP_TAIL):
        assert src_actor.count(anchor) == 1, f"anchor not unique in {ACTOR}: {anchor[:60]!r}"
    for name in names:
        dst = a.out / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(a.src, dst, symlinks=True, ignore=shutil.ignore_patterns(".git", "__pycache__"))
        if name in FILE_PLANTS:
            rel, before, after = FILE_PLANTS[name]
            text = (a.src / rel).read_text()
            assert text.count(before) == 1, (name, before)
            (dst / rel).write_text(text.replace(before, after))
        elif name == "lr2":
            contract = json.loads((Path(__file__).resolve().parents[1] / "judge" / "contract_template.json").read_text())
            (dst / "variant_md_args.json").write_text(json.dumps(["--lr", str(2 * contract["optim"]["lr"])]))
        else:
            patched = PLANTS[name](src_actor)
            assert patched != src_actor, name
            (dst / ACTOR).write_text(patched)
        (dst / "PLANT").write_text(name + "\n")
        print("planted", name, dst)


if __name__ == "__main__":
    main()

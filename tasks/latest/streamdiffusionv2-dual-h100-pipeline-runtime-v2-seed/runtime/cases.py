#!/usr/bin/env python3
"""Canonical, task-owned case definitions for all SDV2 pipeline tiers."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable


TIERS = ("public", "final")
# The StreamDiffusionV2 paper's FPS grid (Fig. 8): two resolutions x 1-4 steps.
# Every point must appear in the final tier so results line up with the paper's
# one-GPU and two-GPU numbers.
PAPER_RESOLUTIONS = ((480, 832), (512, 512))
PAPER_STEPS = (1, 2, 3, 4)


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    clip: str
    height: int
    width: int
    steps: int
    frames: int
    prompt_id: str
    case_seed: int
    tier: str

    def canonical_identity(self) -> tuple[object, ...]:
        """Identity used to prove that case tiers are disjoint."""
        return (
            self.clip,
            self.height,
            self.width,
            self.steps,
            self.frames,
            self.prompt_id,
            self.case_seed,
        )

    @property
    def chunks_expected(self) -> int:
        return (self.frames - 1) // 4

    def to_manifest_row(self) -> dict[str, object]:
        row = asdict(self)
        row["id"] = row.pop("case_id")
        row["chunks_expected"] = self.chunks_expected
        if self.clip.startswith("asset:"):
            row["input"] = {
                "kind": "asset",
                "asset_id": self.clip.removeprefix("asset:"),
            }
        elif self.clip.startswith("procedural:"):
            _, seed, size, frame_count, cut = self.clip.split(":")
            if size != f"{self.height}x{self.width}" or int(frame_count) != self.frames:
                raise ValueError(f"procedural clip descriptor disagrees with {self.case_id}")
            payload: dict[str, object] = {
                "kind": "procedural",
                "seed": int(seed),
                "num_frames": self.frames,
                "height": self.height,
                "width": self.width,
            }
            if cut != "none":
                payload["cut_at"] = int(cut.removeprefix("cut"))
            row["input"] = payload
        else:
            raise ValueError(f"unsupported clip descriptor: {self.clip}")
        return row


PUBLIC_CASES = (
    # Public uses only steps 2 and 4 and stream lengths 129 and 225, so no public case
    # has the (height, width, steps, frames) of a final case; final also adds steps 1
    # and 3, three other resolutions and longer streams.
    CaseSpec("public_bbb_480p_129_s2", "asset:bbb_480p_129", 480, 832, 2, 129, "public_001", 2101, "public"),
    CaseSpec("public_sintel_480p_129_s4", "asset:sintel_480p_129", 480, 832, 4, 129, "public_002", 3102, "public"),
    CaseSpec("public_proc_square_225_cut_s2", "procedural:63:512x512:225:cut129", 512, 512, 2, 225, "public_003", 2203, "public"),
    CaseSpec("public_proc_480p_225_cut_s4", "procedural:61:480x832:225:cut129", 480, 832, 4, 225, "public_004", 4404, "public"),
    CaseSpec("public_proc_square_129_s2", "procedural:42:512x512:129:none", 512, 512, 2, 129, "public_005", 4205, "public"),
    CaseSpec("public_proc_square_225_s4", "procedural:67:512x512:225:none", 512, 512, 4, 225, "public_006", 4506, "public"),
)


FINAL_CASES = (
    # The paper grid: 832x480 and 512x512 at 1, 2, 3 and 4 denoising steps.
    CaseSpec("final_proc_480p_257_cut_s1", "procedural:811:480x832:257:cut129", 480, 832, 1, 257, "hidden_001", 9101, "final"),
    CaseSpec("final_proc_480p_257_s2", "procedural:823:480x832:257:none", 480, 832, 2, 257, "hidden_002", 9102, "final"),
    CaseSpec("final_proc_480p_257_cut_s3", "procedural:829:480x832:257:cut97", 480, 832, 3, 257, "hidden_003", 9103, "final"),
    CaseSpec("final_proc_480p_257_s4", "procedural:839:480x832:257:none", 480, 832, 4, 257, "hidden_004", 9104, "final"),
    CaseSpec("final_sintel_square_513_cut_s1", "asset:sintel_square_513_cut", 512, 512, 1, 513, "hidden_005", 9105, "final"),
    CaseSpec("final_proc_square_257_cut_s2", "procedural:853:512x512:257:cut161", 512, 512, 2, 257, "hidden_006", 9106, "final"),
    CaseSpec("final_proc_square_257_s3", "procedural:857:512x512:257:none", 512, 512, 3, 257, "hidden_007", 9107, "final"),
    CaseSpec("final_proc_square_257_cut_s4", "procedural:863:512x512:257:cut129", 512, 512, 4, 257, "hidden_008", 9108, "final"),
    # Generalization: three resolutions the public suite never shows.
    CaseSpec("final_tears_square576_257_s2", "asset:tears_square576_257", 576, 576, 2, 257, "hidden_009", 9109, "final"),
    CaseSpec("final_tears_portrait_513_cut_s3", "asset:tears_portrait_513_cut", 768, 448, 3, 513, "hidden_010", 9110, "final"),
    CaseSpec("final_proc_640x384_257_s1", "procedural:877:384x640:257:none", 384, 640, 1, 257, "hidden_011", 9111, "final"),
    CaseSpec("final_proc_portrait_257_cut_s4", "procedural:881:768x448:257:cut65", 768, 448, 4, 257, "hidden_012", 9112, "final"),
)


CASES_BY_TIER = {
    "public": PUBLIC_CASES,
    "final": FINAL_CASES,
}


def validate_cases(cases: Iterable[CaseSpec] | None = None) -> None:
    selected = tuple(cases) if cases is not None else sum(CASES_BY_TIER.values(), ())
    ids: set[str] = set()
    identities: dict[tuple[object, ...], str] = {}
    for case in selected:
        if case.case_id in ids:
            raise ValueError(f"duplicate case id: {case.case_id}")
        ids.add(case.case_id)
        if case.canonical_identity() in identities:
            raise ValueError(
                f"case identity collision: {case.case_id} and "
                f"{identities[case.canonical_identity()]}"
            )
        identities[case.canonical_identity()] = case.case_id
        if case.tier not in TIERS or not 1 <= case.steps <= 4:
            raise ValueError(f"invalid tier/steps for {case.case_id}")
        if case.frames < 5 or case.frames % 4 != 1:
            raise ValueError(f"frames must be 4n+1 for {case.case_id}")
        if case.height % 16 or case.width % 16:
            raise ValueError(f"resolution must be divisible by 16 for {case.case_id}")
        expected_prompt_prefix = "hidden_" if case.tier == "final" else "public_"
        if not case.prompt_id.startswith(expected_prompt_prefix):
            raise ValueError(f"prompt visibility mismatch for {case.case_id}")
        case.to_manifest_row()

    public = CASES_BY_TIER["public"]
    final = CASES_BY_TIER["final"]
    public_resolutions = {(case.height, case.width) for case in public}
    public_lengths = {case.frames for case in public}
    public_steps = {case.steps for case in public}
    if not ({(case.height, case.width) for case in final} - public_resolutions):
        raise ValueError("final tier must contain an unseen public resolution")
    if not ({case.frames for case in final} - public_lengths):
        raise ValueError("final tier must contain an unseen public length")
    if not ({case.steps for case in final} - public_steps):
        raise ValueError("final tier must contain an unseen public step count")
    final_grid = {((case.height, case.width), case.steps) for case in final}
    missing = [
        (resolution, steps)
        for resolution in PAPER_RESOLUTIONS
        for steps in PAPER_STEPS
        if (resolution, steps) not in final_grid
    ]
    if missing:
        raise ValueError(f"final tier must cover the paper grid; missing {missing}")
    if len({case.prompt_id for case in final}) != len(final):
        raise ValueError("final cases must use distinct hidden prompts")


def manifest(tier: str) -> dict[str, object]:
    if tier not in CASES_BY_TIER:
        raise ValueError(f"unknown tier: {tier}")
    validate_cases()
    return {
        "schema_version": 1,
        "suite": f"sdv2-pipeline-{tier}-v2",
        "tier": tier,
        "cases": [case.to_manifest_row() for case in CASES_BY_TIER[tier]],
    }


def write_manifests(public_dir: Path, private_dir: Path) -> None:
    public_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    target = public_dir / "public_cases.json"
    target.write_text(json.dumps(manifest("public"), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    final_target = private_dir / "final_cases.json"
    final_target.write_text(json.dumps(manifest("final"), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    final_target.chmod(0o600)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-dir", type=Path, default=Path(__file__).with_name("manifests"))
    parser.add_argument("--private-dir", type=Path, required=True)
    args = parser.parse_args()
    write_manifests(args.public_dir, args.private_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

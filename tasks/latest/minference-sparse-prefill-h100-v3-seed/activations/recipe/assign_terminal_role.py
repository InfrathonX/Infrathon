#!/usr/bin/env python3
"""Bind the already-generated independent holdout to private terminal QA.

This changes only manifest role metadata before the quality-policy freeze; the
input bytes, references, seed domains and generator identity remain unchanged.
"""
import argparse
import json
from pathlib import Path
from quality_v3 import file_sha256, validate_matrix
from ruler_v3 import load_prompts


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    args=parser.parse_args(argv)
    prompts,manifest=load_prompts(args.root,expected_partition="author-holdout")
    validate_matrix(prompts,samples_per_cell=25)
    if len(prompts)!=300:
        raise ValueError("terminal role requires exactly 300 independent QA prompts")
    path=args.root/"manifest.json"
    if manifest.get("role") is not None:
        if manifest["role"]!="private-terminal-quality":
            raise ValueError("cannot overwrite a different role")
        return 0
    manifest["role"]="private-terminal-quality"
    manifest["visibility"]="private Judge only; no interactive/public probing"
    manifest["role_assignment"]={"protocol":"v3-terminal-quality-role-assignment-v1",
        "script_sha256":file_sha256(Path(__file__)),"previous_manifest_sha256":file_sha256(path),
        "freeze_order":"assign role before policy freeze and before first terminal-model prediction"}
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
    temporary.replace(path)
    print(json.dumps({"role":manifest["role"],"samples":300,"prompts_sha256":manifest["prompts_sha256"],
                      "manifest_sha256":file_sha256(path)}),flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())

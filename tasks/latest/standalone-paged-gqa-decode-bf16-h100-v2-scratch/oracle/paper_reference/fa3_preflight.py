"""CPU-only preflight: refuse an FA3 build that cannot run the paged-GQA ABI.

Run inside the candidate FA3 image WITHOUT GPUs (audit addition, 2026-10-29).
FA3 compiles each disabled feature into a TORCH_CHECK message
"This flash attention build does not support <feature>." (hopper/flash_api.cpp
:850-852 and :1148-1164 at 0251105a), so the presence of the message in the
extension binary means the feature is compiled out.  This ABI needs:
  varlen   -- cache_seqlens is passed as seqused_k, which sets is_varlen (:849)
  paged KV -- the non-TMA paged path, taken whenever page_size % kBlockN != 0
              (page 1/16/64 cells) or S_q*G <= kBlockM (every q1 cell) (:415-424)
  splits / pack_gqa / combine kernels -- num_splits_heuristic and PackGQA.
The default image of the original harness, minference-owned-clean-fa3-software
:20261005 (flash_attn_3 3.0.0b1), fails this check: varlen and paged KV are
both compiled out, so every scored call would raise.
"""

import importlib.util
import pathlib
import sys

REQUIRED = (b"varlen", b"paged KV", b"splits", b"pack_gqa", b"combine kernels")
PACKAGES = ("flash_attn_3", "sgl_kernel")


def main() -> int:
    found = False
    bad = []
    for pkg in PACKAGES:
        spec = importlib.util.find_spec(pkg)
        if spec is None or not spec.submodule_search_locations:
            continue
        for root in spec.submodule_search_locations:
            for so in sorted(pathlib.Path(root).glob("*.so")):
                data = so.read_bytes()
                if b"flash attention build does not support" not in data and b"flash_attn" not in data:
                    continue
                found = True
                for feat in REQUIRED:
                    if b"This flash attention build does not support " + feat + b"." in data:
                        bad.append(f"{so}: {feat.decode()} compiled out")
                print(f"checked {so}")
    if not found:
        print("no FA3 extension found (flash_attn_3 / sgl_kernel)", file=sys.stderr)
        return 1
    if bad:
        print("FA3 build cannot run this ABI:\n  " + "\n  ".join(bad), file=sys.stderr)
        return 2
    print("FA3 build has varlen, paged KV (non-TMA), splits and PackGQA")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

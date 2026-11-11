"""Author-only: record frozen replay workloads from a real reference run.

Use with ``--custom-generate-function-path opd_author.record_generate.generate`` and env
``OPD_RECORD_DIR``. Delegates to MD's own ``generate_microgroup`` (non-parser-fetch path) but wraps the
module's ``post`` so every ``POST /rollout/generate`` payload and its raw msgpack response body are written
to ``$OPD_RECORD_DIR/row_{row_id:06d}.{json,msgpack}`` before parsing. The replay server serves exactly
these bytes back.
"""

from __future__ import annotations

import hashlib
import json
import os

import miles.rollout.sglang_diffusion_rollout as sdr

_state = {"installed": False, "row": None}


def _install() -> None:
    if _state["installed"]:
        return
    orig_post = sdr.post

    async def recording_post(url, payload, *a, **kw):
        row = payload.pop("_opd_row_id", None) if isinstance(payload, dict) else None
        out = await orig_post(url, payload, *a, **kw)
        if row is not None and url.endswith("/rollout/generate") and kw.get("raw"):
            d = os.environ["OPD_RECORD_DIR"]
            os.makedirs(d, exist_ok=True)
            clean = payload
            stem = os.path.join(d, f"row_{int(row):06d}")
            with open(stem + ".msgpack", "wb") as f:
                f.write(out)
            json.dump({"payload": clean, "sha256": hashlib.sha256(out).hexdigest(), "bytes": len(out)},
                      open(stem + ".json", "w"), indent=1)
        return out

    sdr.post = recording_post
    _state["installed"] = True


async def generate(args, microgroup, sampling_params, evaluation: bool = False):
    if evaluation:
        return await sdr.generate_microgroup(args, microgroup, sampling_params, evaluation=True)
    _install()
    if args.rollout_fetch_in_parser:
        raise RuntimeError("recording needs the non-fetch path; drop --rollout-fetch-in-parser")
    sampling_params = dict(sampling_params)
    sampling_params["_opd_row_id"] = microgroup[0].metadata["row_id"]
    return await sdr.generate_microgroup(args, microgroup, sampling_params, evaluation=False)

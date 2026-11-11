#!/usr/bin/env python3
"""Cut the Work image's small public data root from the full one assemble_public.py wrote (host tool).

The public suites only get a pipeline working. The cut keeps the probe recordings as they are, reuses the recorded
engine responses of the first 36 rows for a replay of 9 rounds of 4 rows (1 untimed, 2 segments of 4), and keeps the
first 32 contract rows (8 rounds of 4). No GPU is needed: the replay server serves recordings by row.

  python trim_public.py --source OLD_ROOT --out NEW_ROOT
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROWS_PER_ROUND, REPLAY_ROUNDS, CONTRACT_ROWS = 4, 9, 32
SEGMENTS = [[1, 4], [5, 8]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(a.source / "probe", a.out / "probe")
    for name in ("student_init.safetensors", "heldout.jsonl"):
        shutil.copyfile(a.source / name, a.out / name)
    contract = (a.source / "rows.jsonl").read_text().splitlines(keepends=True)
    (a.out / "rows.jsonl").write_text("".join(contract[:CONTRACT_ROWS]))

    replay = a.out / "replay"
    replay.mkdir()
    rows = (a.source / "replay/rows.jsonl").read_text().splitlines(keepends=True)[: ROWS_PER_ROUND * REPLAY_ROUNDS]
    ids = [json.loads(r)["metadata"]["row_id"] for r in rows]
    assert len(ids) == ROWS_PER_ROUND * REPLAY_ROUNDS
    (replay / "rows.jsonl").write_text("".join(rows))
    for row_id in ids:
        for ext in ("json", "msgpack"):
            shutil.copyfile(a.source / f"replay/row_{row_id:06d}.{ext}", replay / f"row_{row_id:06d}.{ext}")
    shutil.copyfile(a.source / "replay/student_init.safetensors", replay / "student_init.safetensors")
    rounds = [ids[i: i + ROWS_PER_ROUND] for i in range(0, len(ids), ROWS_PER_ROUND)]
    (replay / "manifest.json").write_text(json.dumps({"rounds": rounds, "segments": SEGMENTS}, indent=1))
    for path in [a.out, *a.out.rglob("*")]:  # read by the agent user in Work
        path.chmod(0o755 if path.is_dir() else 0o644)
    print(json.dumps({"out": str(a.out), "contract_rows": CONTRACT_ROWS, "replay_rows": len(ids), "rounds": len(rounds)}))


if __name__ == "__main__":
    main()

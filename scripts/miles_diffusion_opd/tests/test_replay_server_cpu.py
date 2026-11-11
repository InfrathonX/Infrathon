"""CPU flow test of the replay server: gating, exact payload match, duplicate/unknown rejection, timing."""

import asyncio
import json
import os
import sys
import tempfile

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "judge"))
from replay_server import ReplayState, build_app, segment_times  # noqa: E402


def make_workload(d: str) -> None:
    rounds = [[0, 1], [2, 3]]
    for r, rows in enumerate(rounds):
        for row in rows:
            payload = {"prompt": f"p{row}", "seed": 42 + 8 * row, "guidance_scale": 1.0, "rollout_sde_type": "cps",
                       "rollout_sde_step_indices": [row % 3, row % 3 + 1, row % 3 + 2], "num_outputs_per_prompt": 8}
            json.dump({"payload": payload, "sha256": "x", "bytes": 5}, open(f"{d}/row_{row:06d}.json", "w"))
            open(f"{d}/row_{row:06d}.msgpack", "wb").write(bytes([row]) * 5)
    json.dump({"rounds": rounds, "segments": [[0, 0], [1, 1]]}, open(f"{d}/manifest.json", "w"))


async def run(d: str, log: str) -> None:
    state = ReplayState(d, log)
    app = build_app(state)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            gold = {row: state.golden[row] for row in range(4)}
            # round 1 row blocks until round 0 is published
            pending = asyncio.create_task(c.post("/rollout/generate", json=gold[2]))
            await asyncio.sleep(0.2)
            assert not pending.done()
            r0 = await c.post("/rollout/generate", json=gold[0])
            assert r0.status_code == 200 and r0.content == bytes([0]) * 5
            bad = dict(gold[1], guidance_scale=4.5)
            assert (await c.post("/rollout/generate", json=bad)).status_code == 400
            assert (await c.post("/replay/published", params={"round": 0})).status_code == 409  # row 1 not served
            assert (await c.post("/rollout/generate", json=gold[1])).status_code == 200
            assert (await c.post("/rollout/generate", json=gold[1])).status_code == 409  # duplicate
            assert (await c.post("/replay/published", params={"round": 0})).status_code == 200
            r2 = await asyncio.wait_for(pending, 5)
            assert r2.status_code == 200 and r2.content == bytes([2]) * 5
            assert (await c.post("/rollout/generate", json={"prompt": "zzz", "seed": 1})).status_code == 400
            assert (await c.post("/rollout/generate", json=gold[3])).status_code == 200
            assert (await c.post("/replay/published", params={"round": 1})).status_code == 200
            st = (await c.get("/replay/status")).json()
            assert st["published"] == 1 and st["violations"] == 4, st


def test_replay_flow():
    with tempfile.TemporaryDirectory() as d:
        make_workload(d)
        log = os.path.join(d, "log.jsonl")
        asyncio.run(run(d, log))
        t = segment_times(log, [[0, 0], [1, 1]])
        assert len(t) == 2 and all(x >= 0 for x in t), t


if __name__ == "__main__":
    test_replay_flow()
    print("ok")

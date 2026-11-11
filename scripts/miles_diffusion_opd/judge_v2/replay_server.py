"""Frozen-response replay / fixture server for the SD3.5-M DiffusionOPD task (Judge side; public copy for perf).

Serves ``POST /rollout/generate`` with the exact msgpack bytes a real engine returned for the same request in
a reference run, so the candidate's parse + teacher scoring + training + publication can be timed without the
(fixed, unscored) engine generation.

Workload dir:
  manifest.json  {"rounds": [[row_id, ...], ...], "segments": [[first_round, last_round], ...]}
  row_{id:06d}.json      {"payload": <request body the reference sent>, "sha256": ..., "bytes": ...}
  row_{id:06d}.msgpack   the engine's response body

Protocol (candidate side):
  POST /rollout/generate                   request must equal the recorded payload exactly (prompt + seed select
                                           the row); each row is served once; a row of a not-yet-released round
                                           blocks until its round is released
  POST /replay/published?round=r           after round r's update was applied to every engine
Release rule: round 0 is released at start; round r+1 only after ``published(r)``. A segment's time is
  t(published(last round)) - t(start), where start is the release of its first round, except for a segment that
  begins at round 0: there it is the arrival of the candidate's first request (process start-up, model loading and
  warm-up before that are not timed; the Judge caps them). All times come from this server's monotonic clock.
Every event is appended to the JSONL log; the Judge reconciles it with the in-engine audit hook afterwards.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response


class ReplayState:
    def __init__(self, workload: str, log_path: str):
        self.dir = workload
        self.manifest = json.load(open(os.path.join(workload, "manifest.json")))
        self.rounds: list[list[int]] = self.manifest["rounds"]
        self.round_of: dict[int, int] = {row: r for r, rows in enumerate(self.rounds) for row in rows}
        self.golden: dict[int, dict] = {}
        self.key: dict[tuple[str, int], int] = {}
        for row in self.round_of:
            rec = json.load(open(os.path.join(workload, f"row_{row:06d}.json")))
            self.golden[row] = rec["payload"]
            self.key[(rec["payload"]["prompt"], int(rec["payload"]["seed"]))] = row
        self.served: set[int] = set()
        self.released = -1
        self.published = -1
        self.events: dict[int, asyncio.Event] = {r: asyncio.Event() for r in range(len(self.rounds))}
        self.log = open(log_path, "a", buffering=1)
        self.violations = 0

    def emit(self, **rec) -> None:
        rec["t_mono"] = time.monotonic()
        rec["t_wall"] = time.time()
        self.log.write(json.dumps(rec, separators=(",", ":")) + "\n")

    def release(self, r: int) -> None:
        if r < len(self.rounds) and r > self.released:
            self.released = r
            self.emit(kind="release", round=r)
            self.events[r].set()


def build_app(state: ReplayState) -> FastAPI:
    app = FastAPI()

    @app.post("/rollout/generate")
    async def generate(request: Request):
        body = await request.json()
        row = state.key.get((body.get("prompt"), int(body.get("seed", -1))))
        state.emit(kind="request", row=row, round=None if row is None else state.round_of.get(row))
        if row is None:
            state.violations += 1
            state.emit(kind="violation", reason="unknown_request", prompt=str(body.get("prompt"))[:80], seed=body.get("seed"))
            raise HTTPException(status_code=400, detail="request does not match any contract row")
        if body != state.golden[row]:
            diff = sorted(k for k in set(body) | set(state.golden[row]) if body.get(k) != state.golden[row].get(k))
            state.violations += 1
            state.emit(kind="violation", reason="payload_mismatch", row=row, fields=diff)
            raise HTTPException(status_code=400, detail=f"payload differs from the contract in {diff}")
        if row in state.served:
            state.violations += 1
            state.emit(kind="violation", reason="duplicate", row=row)
            raise HTTPException(status_code=409, detail="row already served")
        r = state.round_of[row]
        t_wait = time.monotonic()
        await state.events[r].wait()
        state.served.add(row)
        state.emit(kind="serve", row=row, round=r, waited=time.monotonic() - t_wait)
        with open(os.path.join(state.dir, f"row_{row:06d}.msgpack"), "rb") as f:
            data = f.read()
        return Response(content=data, media_type="application/msgpack")

    @app.post("/replay/published")
    async def published(round: int):
        if round != state.published + 1 or round > state.released:
            state.violations += 1
            state.emit(kind="violation", reason="bad_published_order", round=round)
            raise HTTPException(status_code=409, detail=f"expected published round {state.published + 1}")
        missing = [row for row in state.rounds[round] if row not in state.served]
        if missing:
            state.violations += 1
            state.emit(kind="violation", reason="published_before_all_rows_served", round=round, missing=missing)
            raise HTTPException(status_code=409, detail=f"round {round} rows not all served: {missing}")
        state.published = round
        state.emit(kind="published", round=round)
        state.release(round + 1)
        return {"ok": True, "released": state.released}

    @app.get("/replay/status")
    async def status():
        return {"released": state.released, "published": state.published, "served": len(state.served),
                "violations": state.violations}

    @app.on_event("startup")
    async def _start():
        state.emit(kind="start", rounds=len(state.rounds))
        state.release(0)

    return app


def segment_times(log_path: str, segments: list[list[int]]) -> list[float]:
    ev = [json.loads(line) for line in open(log_path)]
    rel = {e["round"]: e["t_mono"] for e in ev if e["kind"] == "release"}
    pub = {e["round"]: e["t_mono"] for e in ev if e["kind"] == "published"}
    first_req = min((e["t_mono"] for e in ev if e["kind"] == "request" and e.get("round") == 0), default=None)
    out = []
    for first, last in segments:
        start = first_req if first == 0 else rel[first]
        out.append(pub[last] - start)
    return out


def warmup_seconds(log_path: str) -> float | None:
    """Server start -> the candidate's first generate request (untimed; reported)."""
    ev = [json.loads(line) for line in open(log_path)]
    start = next(e["t_mono"] for e in ev if e["kind"] == "start")
    first_req = min((e["t_mono"] for e in ev if e["kind"] == "request"), default=None)
    return None if first_req is None else first_req - start


def setup_seconds(log_path: str, segments: list[list[int]]) -> float | None:
    """Untimed setup, capped by the Judge: server start -> release of the first timed round, i.e. process start-up,
    loading, engine start, compilation and every untimed round before the first segment (the probe round). With a
    segment starting at round 0 it is the warm-up up to the first request."""
    first = min((s[0] for s in segments), default=0)
    if first == 0:
        return warmup_seconds(log_path)
    ev = [json.loads(line) for line in open(log_path)]
    start = next(e["t_mono"] for e in ev if e["kind"] == "start")
    rel = {e["round"]: e["t_mono"] for e in ev if e["kind"] == "release"}
    return rel[first] - start if first in rel else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--log", required=True)
    args = ap.parse_args()
    state = ReplayState(args.workload, args.log)
    uvicorn.run(build_app(state), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

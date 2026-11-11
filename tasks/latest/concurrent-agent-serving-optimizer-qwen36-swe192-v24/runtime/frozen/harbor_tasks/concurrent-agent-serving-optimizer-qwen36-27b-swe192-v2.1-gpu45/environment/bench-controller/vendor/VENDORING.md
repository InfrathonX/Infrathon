# Vendored mini-SWE-agent core

The following files originate from this repository's
`examples/inference/mini-swe-agent` tree at ThunderAgent commit
`a989f24c2d53096e243aae39775637b8ddd0e380`:

- `minisweagent/agents/default.py`
- `minisweagent/environments/extra/bubblewrap.py`

That tree reports mini-SWE-agent version `1.14.4`. The small package
`__init__.py` here only supplies the two typing protocols imported by the
vendored agent without loading user-level config. Upstream's MIT license is in
`mini-swe-agent.LICENSE.md`.

`agents/default.py` has task-local behavioral adaptations: it requires the
Qwen3.6 system-role text protocol, rejects function-tool calls and empty/multiple
Bash actions, records query/tool timing, accepts the benchmark's direct submit
marker, and preserves bounded job-timeout handling. These differences are
covered by the adapter, tool-agent, warm-up, and watchdog unit suites; this is
not a byte-for-byte upstream vendoring claim.

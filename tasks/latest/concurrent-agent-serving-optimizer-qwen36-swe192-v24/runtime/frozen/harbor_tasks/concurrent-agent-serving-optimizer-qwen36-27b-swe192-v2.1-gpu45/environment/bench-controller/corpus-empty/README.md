This directory is an intentional fail-closed placeholder.

Development profiles use the synthetic CPU fixture. A release run must mount
the pinned, staged SWE-bench Lite corpus at `/opt/kvbench-corpus`; the controller
will not report real-workload readiness without its validated `manifest.json`
and clean workspaces.

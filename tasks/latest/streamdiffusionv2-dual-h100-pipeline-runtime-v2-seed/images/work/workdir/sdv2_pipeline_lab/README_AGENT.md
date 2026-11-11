# Participant workspace

Keep experiments under numbered candidates/runtime_vNNN packages and never edit a prior
version in place. The selected version is the integer in FINAL_VERSION.

Commands:

    tools/run_one.sh --version 1 --case public_bbb_480p_129_s2
    tools/run_public_tests.sh --candidate-version 1 --quick
    tools/run_public_tests.sh --candidate-version 1

The controller uses the same constructor/start/push/flush/close lifecycle, the same
push stamps (taken by the controller just before each start/push) and the same exact
ledger comparator as the Judge. Read AGENT_TASK.md completely for the per-chunk
protocol, metric definitions, migration state, hard gates, and promotion policy.

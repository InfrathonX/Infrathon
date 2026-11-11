"""Moved to sforge.harness.agent.claude_local; kept for existing callers."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("sforge.harness.agent.claude_local", run_name="__main__", alter_sys=True)
else:
    from sforge.harness.agent.claude_local import *  # noqa: F401,F403

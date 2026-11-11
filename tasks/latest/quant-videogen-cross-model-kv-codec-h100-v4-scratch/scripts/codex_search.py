"""Small command adapter for the task's native Codex web-search contract."""


def enable_native_web_search(command: str) -> str:
    marker = "codex exec"
    if marker not in command:
        raise ValueError("Codex command does not contain a non-interactive exec invocation")
    if 'web_search="disabled"' in command:
        raise ValueError("Codex command explicitly disables web search")
    if "codex --search exec" in command:
        return command
    return command.replace(marker, "codex --search exec", 1)

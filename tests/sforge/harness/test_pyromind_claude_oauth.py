"""Native Claude OAuth stays refreshable and private for the whole run.

Every cache/token here is synthetic. The actual generated supervisor and remote
probe run against isolated file/process stand-ins; no host login cache is read.
"""
import ast
import io
import json
import os
from pathlib import Path
import tarfile
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sforge.harness import agent_config as AUTH
from sforge.harness.agent import claude_agent as CLAUDE
from sforge.pyromind.runner import supervisor_source, validate_no_credentials


def cache(access="oauth-access-original", refresh="oauth-refresh-original", expiry=2_000_000_000_000):
    return {"claudeAiOauth": {"accessToken": access, "refreshToken": refresh,
                             "expiresAt": expiry, "scopes": ["user:inference"]}}


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    import sforge.harness.agent_runtime as runtime_module
    runtime = SimpleNamespace(agent="claude-code", cli="claude", version="2.1.287",
                              root=tmp_path / "runtime")
    monkeypatch.setattr(runtime_module, "load_agent_runtime", lambda _: runtime)
    native = tmp_path / "credentials.json"
    native.write_text(json.dumps(cache()))
    native.chmod(0o600)
    key = tmp_path / "model.key"
    key.write_text("fake-anthropic-key")
    key.chmod(0o600)
    base = {"AGENT": "claude-code", "MODEL": "claude-opus-5-5",
            "REASONING_EFFORT": "max", "SFORGE_AGENT_RUNTIME_DIR": str(runtime.root)}
    def config(**values):
        path = tmp_path / "model.env"
        path.write_text("".join(f"{name}={value}\n" for name, value in {**base, **values}.items()))
        return path
    return SimpleNamespace(native=native, key=key, runtime=runtime, config=config)


def test_common_loader_keeps_native_cache_and_uses_official_oauth(fixture):
    before = fixture.native.read_bytes()
    model = AUTH.load_model_config(fixture.config(CLAUDE_AUTH_JSON_PATH=fixture.native))
    credentials = AUTH.load_credentials(model)
    assert model["AUTH_MODE"] == credentials.mode == "claude-oauth"
    assert model["runtime_version"] == "2.1.287" and model["REASONING_EFFORT"] == "max"
    assert credentials.payload == before == fixture.native.read_bytes()
    assert credentials.secrets == {"oauth-access-original", "oauth-refresh-original"}
    assert all(secret not in repr(credentials) for secret in credentials.secrets)
    settings = CLAUDE.claude_settings(model)
    assert settings["env"] == {"CLAUDE_CODE_EFFORT_LEVEL": "max"}
    assert "ANTHROPIC" not in json.dumps(settings)
    assert all(secret not in json.dumps(settings) for secret in credentials.secrets)
    assert "--effort" in CLAUDE.claude_command(model, "continue", resume="continue")


@pytest.mark.parametrize("extra", [
    {}, {"MODEL_API_KEY_FILE": "key"}, {"MODEL_API_BASE_URL": "https://api.invalid"},
    {"MODEL_API_KEY_FILE": "key", "MODEL_API_BASE_URL": "https://api.invalid"},
    {"CODEX_AUTH_JSON_PATH": "native"}, {"AUTH_MODE": "anthropic-api-key"},
])
def test_oauth_modes_are_mutually_exclusive(fixture, extra):
    values = {name: str(getattr(fixture, value)) if value in {"key", "native"} else value
              for name, value in extra.items()}
    if not extra:
        # No cache and no key/base must also fail.
        path = fixture.config()
    else:
        path = fixture.config(CLAUDE_AUTH_JSON_PATH=fixture.native, **values)
    with pytest.raises(ValueError):
        AUTH.load_model_config(path)


@pytest.mark.parametrize("change", [
    {"accessToken": None}, {"accessToken": ""}, {"accessToken": " token "},
    {"refreshToken": None}, {"refreshToken": ""}, {"refreshToken": "line\nbreak"},
    {"expiresAt": None}, {"expiresAt": "2000000000000"}, {"expiresAt": True},
    {"expiresAt": -1}, {"expiresAt": float("inf")}, {"expiresAt": float("nan")},
])
def test_native_cache_rejects_missing_nonrefreshable_or_malformed_credentials(change):
    document = cache()
    document["claudeAiOauth"].update(change)
    with pytest.raises(ValueError):
        CLAUDE.parse_claude_oauth_cache(json.dumps(document).encode())


@pytest.mark.parametrize("data", [b"not json", b"[]", b"{}", b'{"claudeAiOauth":[]}', b"\xff"])
def test_native_cache_rejects_non_native_documents(data):
    with pytest.raises(ValueError):
        CLAUDE.parse_claude_oauth_cache(data)


def test_expired_access_token_with_native_refresh_token_remains_refreshable():
    document = cache(expiry=1)
    assert CLAUDE.parse_claude_oauth_cache(json.dumps(document).encode()) == document


def test_claude_api_key_mode_keeps_existing_transport(fixture):
    model = AUTH.load_model_config(fixture.config(MODEL_API_KEY_FILE=fixture.key,
        MODEL_API_BASE_URL="https://model.example.test"))
    credentials = AUTH.load_credentials(model)
    assert credentials.mode == "anthropic-api-key"
    assert credentials.payload == b"fake-anthropic-key"
    assert CLAUDE.claude_settings(model)["env"]["ANTHROPIC_BASE_URL"] == "https://model.example.test"


@pytest.mark.parametrize("mode", ["claude-oauth", "anthropic-api-key"])
def test_worker_install_private_native_cache_and_actual_cli_probe(fixture, mode):
    values = ({"CLAUDE_AUTH_JSON_PATH": fixture.native} if mode == "claude-oauth"
              else {"MODEL_API_KEY_FILE": fixture.key, "MODEL_API_BASE_URL": "https://model.example.test"})
    model = AUTH.load_model_config(fixture.config(**values))
    credentials = AUTH.load_credentials(model)
    backend, checks = Mock(), []
    def check(_backend, _worker, command, **kwargs):
        checks.append((command, kwargs))
        return "OK"
    receipt = AUTH.install_and_validate_agent(backend, object(), model, fixture.runtime,
                                             credentials, check)
    written = {str(call.args[2]): call.args[1] for call in backend.write_to_container.call_args_list}
    destination = AUTH.CLAUDE_AUTH_CACHE_PATH if mode == "claude-oauth" else CLAUDE.CLAUDE_KEY_PATH
    assert written[destination].encode() == credentials.payload
    assert (AUTH.CLAUDE_AUTH_CACHE_PATH in written) == (mode == "claude-oauth")
    assert (CLAUDE.CLAUDE_KEY_PATH in written) == (mode == "anthropic-api-key")
    assert (["install", "-o", "agent", "-g", "agent", "-m", "600", "/dev/null", destination],
            {"timeout": 15}) in checks
    assert receipt["auth_mode"] == mode and receipt["claude_cli"] is True
    assert all(secret not in repr(checks) for secret in credentials.secrets)
    probe = next(command for command, _ in checks if isinstance(command, list)
                 and command[:2] == ["python3", "-c"])
    reads, invocations = [], []
    def read_path(path):
        reads.append(path)
        return SimpleNamespace(read_text=lambda: "fake-anthropic-key")
    def run(command, **kwargs):
        invocations.append((command, kwargs))
        record = {"type": "result", "subtype": "success", "is_error": False,
                  "result": "READY", "modelUsage": {model["MODEL"]: {}}}
        return SimpleNamespace(returncode=0, stdout=json.dumps(record))
    namespace = {"json": json, "os": SimpleNamespace(environ={"PATH": "/bin",
        "ANTHROPIC_API_KEY": "inherited-wrong", "ANTHROPIC_BASE_URL": "https://wrong.invalid",
        "CLAUDE_CODE_OAUTH_TOKEN": "inherited-short-token"}),
        "pathlib": SimpleNamespace(Path=read_path), "subprocess": SimpleNamespace(run=run),
        "sys": SimpleNamespace(argv=["-c", *probe[3:]])}
    tree = ast.parse(probe[2])
    tree.body = [node for node in tree.body if not isinstance(node, ast.Import)]
    exec(compile(tree, "<isolated-native-cli-probe>", "exec"), namespace)
    environment = invocations[0][1]["env"]
    assert reads == ([CLAUDE.CLAUDE_KEY_PATH] if mode == "anthropic-api-key" else [])
    assert ("ANTHROPIC_API_KEY" in environment) == (mode == "anthropic-api-key")
    assert ("ANTHROPIC_BASE_URL" in environment) == (mode == "anthropic-api-key")
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in environment
    assert environment["HOME"] == "/home/agent" and environment["CLAUDE_CODE_EFFORT_LEVEL"] == "max"


def test_supervisor_native_oauth_never_reads_key_or_routes_custom_endpoint():
    tree, nodes, started = ast.parse(supervisor_source()), [], False
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "env" for target in node.targets):
            started = True
        if started and isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "fast_failures" for target in node.targets):
            break
        if started:
            nodes.append(node)
    reads = []
    namespace = {"os": SimpleNamespace(environ={"PATH": "/bin", "OPENAI_API_KEY": "wrong",
        "ANTHROPIC_API_KEY": "wrong", "ANTHROPIC_BASE_URL": "https://wrong.invalid",
        "CLAUDE_CODE_OAUTH_TOKEN": "wrong", "PYROMIND_API_KEY": "platform"}),
        "cfg": {"environment": {}, "auth_mode": "claude-oauth", "reasoning_effort": "max"},
        "pathlib": SimpleNamespace(Path=lambda value: reads.append(value)), "deadline": 43_200}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<isolated-supervisor-env>", "exec"), namespace)
    environment = namespace["env"]
    assert not reads
    assert all(name not in environment for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL", "CLAUDE_CODE_OAUTH_TOKEN", "PYROMIND_API_KEY"))
    assert environment["CLAUDE_CODE_EFFORT_LEVEL"] == "max"


def test_native_refresh_journal_and_export_protect_initial_intermediate_current_tokens(tmp_path):
    native = tmp_path / "credentials.json"
    native.write_text(json.dumps(cache()))
    def path(value):
        return native if str(value) == AUTH.CLAUDE_AUTH_CACHE_PATH else Path(value)
    namespace = {"cfg": {"auth_mode": "claude-oauth"}, "json": json,
                 "pathlib": SimpleNamespace(Path=path), "ROOT": tmp_path, "os": os}
    exec(AUTH.SUPERVISOR_AUTH_SOURCE, namespace)
    native.chmod(0o600)
    original = AUTH.load_credentials({"AUTH_MODE": "claude-oauth",
        "CLAUDE_AUTH_JSON_PATH": str(native)})
    namespace["remember_auth"]()
    for label in ("between", "current"):
        native.write_text(json.dumps(cache("access-" + label, "refresh-" + label)))
        namespace["remember_auth"]()
    journal = tmp_path / "auth-secrets.json"
    assert journal.stat().st_mode & 0o777 == 0o600
    responses = {AUTH.CLAUDE_AUTH_CACHE_PATH: native.read_bytes(), AUTH.AUTH_JOURNAL_PATH: journal.read_bytes()}
    backend = SimpleNamespace(client=SimpleNamespace(read_file=lambda _id, name: responses[name]))
    known = set(original.secrets)
    AUTH.collect_worker_auth_secrets(backend, SimpleNamespace(id="isolated-worker"), original,
                                    known, required=True, journal_required=True)
    assert known == {"oauth-access-original", "oauth-refresh-original", "access-between",
                     "refresh-between", "access-current", "refresh-current"}
    leaked = b"candidate contains refresh-between and access-current"
    assert AUTH.redact(leaked, known) == b"candidate contains [REDACTED] and [REDACTED]"
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as bundle:
        member = tarfile.TarInfo("candidate.py")
        member.size = len(leaked)
        bundle.addfile(member, io.BytesIO(leaked))
    with pytest.raises(ValueError, match="Credential material"):
        validate_no_credentials(payload.getvalue(), known)
    del responses[AUTH.AUTH_JOURNAL_PATH]
    with pytest.raises(RuntimeError, match="before artifact export"):
        AUTH.collect_worker_auth_secrets(backend, SimpleNamespace(id="isolated-worker"), original,
                                        known, required=True, journal_required=True)


def test_generic_secret_collection_handles_both_oauth_field_spellings():
    document = {"nested": [cache(), {"idToken": "camel-id", "access_token": "snake-access",
                                    "API_KEY": "api-secret", "accountId": "public-account"}]}
    assert AUTH.auth_secrets(document) == {"oauth-access-original", "oauth-refresh-original",
                                          "camel-id", "snake-access", "api-secret"}

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


TASK_ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = (
    TASK_ROOT
    / "runtime/frozen/harbor_tasks"
    / "concurrent-agent-serving-optimizer-qwen36-27b-swe192-v2.1-gpu45"
    / "environment"
)
TRANSPORT_PATH = ENVIRONMENT / "model-api/transport.py"

SPEC = importlib.util.spec_from_file_location("kvbench_model_api_transport", TRANSPORT_PATH)
assert SPEC is not None and SPEC.loader is not None
transport = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transport)


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        ("http://example.com", "http://example.com"),
        ("HTTP://Example.COM/", "http://example.com"),
        ("http://127.0.0.1:80/", "http://127.0.0.1"),
        ("https://host.docker.internal:443", "https://host.docker.internal"),
        ("https://[2001:db8::1]:8443/", "https://[2001:db8::1]:8443"),
    ],
)
def test_configured_base_url_accepts_only_an_origin(raw: str, normalized: str) -> None:
    assert transport.configured_base_url(raw) == normalized


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "example.com:8000",
        "ftp://example.com",
        "http://",
        "http://example.com:",
        "http://example.com:65536",
        "http://user@example.com",
        "http://user:secret@example.com",
        "http://example.com/v1",
        "http://example.com//",
        "http://example.com?",
        "http://example.com?x=1",
        "http://example.com#",
        "http://example.com#fragment",
        "http://example.com/%2e",
        "http://example.com\\evil.invalid",
        "http://-invalid.example",
        "http://example_com",
        "http://éxample.com",
        "http://[fe80::1%25eth0]",
        " http://example.com",
        "http://example.com\n",
    ],
)
def test_configured_base_url_rejects_non_origins(raw: str) -> None:
    with pytest.raises(transport.UpstreamConfigurationError):
        transport.configured_base_url(raw)


def test_configured_base_url_is_required_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("UPSTREAM_BASE_URL", raising=False)
    with pytest.raises(transport.UpstreamConfigurationError):
        transport.configured_base_url()

    monkeypatch.setenv("UPSTREAM_BASE_URL", "https://Serving.Example:8443/")
    assert transport.configured_base_url() == "https://serving.example:8443"


def test_upstream_url_joins_a_trusted_request_target() -> None:
    assert (
        transport.upstream_url(
            "/v1/models?limit=1", base_url="http://host.docker.internal:8000"
        )
        == "http://host.docker.internal:8000/v1/models?limit=1"
    )


@pytest.mark.parametrize(
    "target",
    [
        "",
        "v1/models",
        "//evil.invalid/models",
        "/\\evil",
        "/models#fragment",
        "/x\nInjected: yes",
    ],
)
def test_upstream_url_rejects_ambiguous_request_targets(target: str) -> None:
    with pytest.raises(transport.UpstreamConfigurationError):
        transport.upstream_url(target, base_url="http://example.com")

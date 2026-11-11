#!/usr/bin/env python3
"""Fail-closed HTTP(S) upstream configuration for the trusted model proxy."""

from __future__ import annotations

import ipaddress
import os
import re
from urllib.parse import urlsplit, urlunsplit


class UpstreamConfigurationError(RuntimeError):
    """The trusted proxy was not given a safe OpenAI-compatible origin."""


DNS_LABEL = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")


def _valid_hostname(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        pass
    try:
        hostname.encode("ascii")
    except UnicodeEncodeError:
        return False
    absolute_name = hostname[:-1] if hostname.endswith(".") else hostname
    return bool(
        absolute_name
        and len(absolute_name) <= 253
        and all(DNS_LABEL.fullmatch(label) for label in absolute_name.split("."))
    )


def configured_base_url(value: str | None = None) -> str:
    """Return one normalized HTTP(S) origin with no embedded credentials.

    ``UPSTREAM_BASE_URL`` names the server origin, not its OpenAI ``/v1``
    prefix.  Keeping the path empty lets the proxy address both standard
    OpenAI routes and the vLLM health/metrics/cache-control routes without
    accepting an ambiguous URL prefix.
    """

    raw = os.environ.get("UPSTREAM_BASE_URL") if value is None else value
    if not raw:
        raise UpstreamConfigurationError("UPSTREAM_BASE_URL must be configured")
    if (
        raw != raw.strip()
        or "\x00" in raw
        or any(character.isspace() for character in raw)
    ):
        raise UpstreamConfigurationError(
            "UPSTREAM_BASE_URL must not contain whitespace or NUL bytes"
        )
    if any(character in raw for character in ("\\", "%", "?", "#")):
        raise UpstreamConfigurationError(
            "UPSTREAM_BASE_URL must be a literal HTTP(S) origin"
        )

    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError as exc:
        raise UpstreamConfigurationError("UPSTREAM_BASE_URL is malformed") from exc

    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise UpstreamConfigurationError("UPSTREAM_BASE_URL must use http or https")
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise UpstreamConfigurationError(
            "UPSTREAM_BASE_URL must be an origin without credentials, path, query, or fragment"
        )
    if parsed.netloc.endswith(":") or not _valid_hostname(parsed.hostname):
        raise UpstreamConfigurationError("UPSTREAM_BASE_URL has an invalid host")
    if port is not None and not 1 <= port <= 65535:
        raise UpstreamConfigurationError("UPSTREAM_BASE_URL port is out of range")

    hostname = parsed.hostname.lower()
    rendered_host = f"[{hostname}]" if ":" in hostname else hostname
    default_port = 80 if scheme == "http" else 443
    netloc = rendered_host if port in {None, default_port} else f"{rendered_host}:{port}"
    return urlunsplit((scheme, netloc, "", "", ""))


def upstream_url(path_qs: str, *, base_url: str | None = None) -> str:
    """Join a trusted absolute-path request target to the configured origin."""

    if (
        not isinstance(path_qs, str)
        or not path_qs.startswith("/")
        or path_qs.startswith("//")
        or "\\" in path_qs
        or "#" in path_qs
        or any(
            ord(character) < 0x20 or ord(character) == 0x7F
            for character in path_qs
        )
    ):
        raise UpstreamConfigurationError("upstream request target must be an absolute path")
    origin = configured_base_url() if base_url is None else configured_base_url(base_url)
    return f"{origin}{path_qs}"

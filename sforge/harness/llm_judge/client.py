"""Minimal OpenAI-compatible chat-completions client for the review.

Any endpoint that speaks ``POST {base}/chat/completions`` works: a
self-hosted sglang/vLLM server, OpenAI, OpenRouter, DashScope's compatible
mode.  The key is read from the configured file or variable per request and
never appears in an exception, log line or persisted artifact.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests

from sforge.harness.llm_judge.config import LLMJudgeConfig

REASONING_CAPTURE_MAX_CHARS = 200_000


class LLMRequestError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool, status: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


@dataclass
class ChatResult:
    content: str
    reasoning: str | None
    finish_reason: str | None
    usage: dict[str, Any] | None
    model: str | None
    latency_seconds: float

    def record(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "reasoning": (
                self.reasoning[:REASONING_CAPTURE_MAX_CHARS]
                if self.reasoning
                else None
            ),
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "model": self.model,
            "latency_seconds": round(self.latency_seconds, 3),
        }


def _excerpt(response: requests.Response) -> str:
    try:
        text = response.text
    except Exception:
        return ""
    return " ".join(text.split())[:300]


class ChatCompletionsClient:
    def __init__(self, config: LLMJudgeConfig) -> None:
        self.config = config
        self._json_mode = config.json_mode

    @property
    def endpoint(self) -> str:
        assert self.config.api_base_url is not None
        return self.config.api_base_url.rstrip("/") + "/chat/completions"

    def _body(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        config = self.config
        body: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
        }
        if "max_completion_tokens" not in config.extra_body:
            body["max_tokens"] = config.max_output_tokens
        if config.reasoning_effort:
            body["reasoning_effort"] = config.reasoning_effort
        if self._json_mode:
            body["response_format"] = {"type": "json_object"}
        body.update(dict(config.extra_body))
        return body

    def complete(
        self, messages: list[dict[str, str]], *, timeout: float
    ) -> ChatResult:
        key = self.config.read_api_key()
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        started = time.monotonic()
        for _json_retry in range(2):
            try:
                response = requests.post(
                    self.endpoint,
                    json=self._body(messages),
                    headers=headers,
                    timeout=(min(30.0, timeout), timeout),
                )
            except requests.Timeout:
                raise LLMRequestError(
                    f"request timed out after {timeout:.0f}s", retryable=True
                ) from None
            except requests.RequestException as exc:
                raise LLMRequestError(
                    f"request failed: {type(exc).__name__}", retryable=True
                ) from None
            if (
                response.status_code == 400
                and self._json_mode
                and "response_format" in response.text
            ):
                # Some servers reject JSON mode; the prompt already demands a
                # bare JSON object, so retry once without it.
                self._json_mode = False
                continue
            break
        latency = time.monotonic() - started
        status = response.status_code
        if status != 200:
            retryable = status in {408, 409, 425, 429} or status >= 500
            raise LLMRequestError(
                f"HTTP {status}: {_excerpt(response)}",
                retryable=retryable,
                status=status,
            )
        try:
            payload = response.json()
            choice = payload["choices"][0]
            message = choice.get("message") or {}
        except (ValueError, KeyError, IndexError, TypeError):
            raise LLMRequestError(
                f"response is not a chat completion: {_excerpt(response)}",
                retryable=True,
                status=status,
            ) from None
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if not isinstance(reasoning, str):
            reasoning = None
        return ChatResult(
            content=content if isinstance(content, str) else "",
            reasoning=reasoning,
            finish_reason=choice.get("finish_reason"),
            usage=payload.get("usage") if isinstance(payload.get("usage"), dict) else None,
            model=payload.get("model") if isinstance(payload.get("model"), str) else None,
            latency_seconds=latency,
        )

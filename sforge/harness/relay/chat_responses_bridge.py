"""Stateless text/tool Responses compatibility for a Chat-only model provider.

The caller owns authentication, HTTP timeouts, admission and safe usage logs.
This module performs no I/O. Custom Responses tools are represented upstream
as functions with an ``input`` string; their original raw input and call IDs
are restored in Responses output. Thinking history is preserved in an opaque
base64 marker in ``encrypted_content``. That marker is encoding, NOT encryption,
and may contain provider reasoning text; it must never enter usage logs.

Schema references: https://developers.openai.com/api/docs/guides/function-calling
and https://developers.openai.com/api/reference/resources/responses/streaming-events
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import time
from collections.abc import Iterator


REASONING_PREFIX = "sforge-chat-reasoning-v1:"
# Hosted search runs on the Responses provider. A Chat Completions provider
# cannot execute it, so the bridge omits it instead of failing the request
# (Codex sends it whenever the task allows internet).
HOSTED_SEARCH_TOOLS = ("web_search", "web_search_preview")


def _text(value, *, label):
    if not isinstance(value, str):
        raise ValueError(label + " must be a string")
    return value


def _content(value, *, label):
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise ValueError(label + " must be text or a text content list")
    pieces = []
    for part in value:
        if not isinstance(part, dict):
            raise ValueError(label + " content part is invalid")
        kind = part.get("type")
        if kind in ("input_text", "output_text", "text"):
            pieces.append(_text(part.get("text"), label=label + " text"))
        elif kind == "refusal":
            pieces.append(_text(part.get("refusal"), label=label + " refusal"))
        else:
            raise ValueError("Chat bridge does not support content type " + str(kind))
    return "\n".join(pieces)


def _tool_kinds(document):
    tools = document.get("tools", [])
    if not isinstance(tools, list):
        raise ValueError("Responses tools must be a list")
    result = {}
    for tool in tools:
        if isinstance(tool, dict) and tool.get("type") in HOSTED_SEARCH_TOOLS:
            continue
        if not isinstance(tool, dict) or tool.get("type") not in ("function", "custom"):
            raise ValueError("Chat bridge supports only function and custom tools")
        name = _text(tool.get("name"), label="Tool name")
        if not name or name in result:
            raise ValueError("Tool names must be nonempty and unique")
        result[name] = tool["type"]
    return result


def _reasoning(item):
    opaque = item.get("encrypted_content")
    if opaque is not None:
        if not isinstance(opaque, str) or not opaque.startswith(REASONING_PREFIX):
            raise ValueError("Chat bridge cannot replay another provider's encrypted reasoning")
        try:
            return base64.b64decode(opaque[len(REASONING_PREFIX):], altchars=b"-_", validate=True).decode()
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError("Chat bridge reasoning marker is malformed") from exc
    parts = item.get("summary", item.get("content", []))
    if not isinstance(parts, list):
        raise ValueError("Reasoning summary must be a list")
    texts = []
    for part in parts:
        if not isinstance(part, dict) or part.get("type") not in ("summary_text", "reasoning_text"):
            raise ValueError("Chat bridge cannot replay unknown reasoning content")
        texts.append(_text(part.get("text"), label="Reasoning text"))
    return "\n".join(texts)


def request_to_chat(document: dict, model: str, effort: str = "max") -> dict:
    """Convert complete stateless Responses history without dropping unknown items."""
    if not isinstance(document, dict) or not isinstance(model, str) or not model:
        raise ValueError("Chat bridge requires a Responses object and pinned model")
    if document.get("previous_response_id") is not None or document.get("conversation") is not None:
        raise ValueError("Chat bridge requires complete stateless input history")
    kinds = _tool_kinds(document)
    messages = []
    instructions = document.get("instructions")
    if instructions is not None:
        messages.append({"role": "system", "content": _text(instructions, label="Instructions")})
    source = document.get("input", [])
    if isinstance(source, str):
        source = [{"role": "user", "content": source}]
    if not isinstance(source, list):
        raise ValueError("Responses input must be a string or item list")
    assistant = None
    def flush():
        nonlocal assistant
        if assistant is not None:
            messages.append(assistant)
            assistant = None
    def current_assistant():
        nonlocal assistant
        if assistant is None:
            assistant = {"role": "assistant", "content": ""}
        return assistant
    # Providers sometimes reuse a call ID already in the history (GLM on MaaS did, and every
    # later turn of that run then failed).  IDs only pair calls with outputs here, so a reused
    # ID is sent under a fresh wire ID and outputs pair with that ID's oldest unanswered call.
    wire_ids, unanswered = set(), {}
    for item in source:
        if not isinstance(item, dict):
            raise ValueError("Responses input items must be objects")
        kind = item.get("type", "message" if "role" in item else None)
        if kind == "message":
            role = item.get("role")
            if role not in ("user", "assistant", "system", "developer"):
                raise ValueError("Unsupported Responses message role")
            content = _content(item.get("content", ""), label="Message")
            if role == "assistant":
                target = current_assistant()
                target["content"] += ("\n" if target["content"] and content else "") + content
            else:
                flush()
                messages.append({"role": "system" if role == "developer" else role, "content": content})
        elif kind == "reasoning":
            target = current_assistant()
            text = _reasoning(item)
            previous = target.get("reasoning_content", "")
            target["reasoning_content"] = previous + ("\n" if previous and text else "") + text
        elif kind in ("function_call", "custom_tool_call"):
            call_id = _text(item.get("call_id"), label="Tool call ID")
            name = _text(item.get("name"), label="Tool call name")
            # A past tool may no longer be advertised after compaction; preserve
            # its exact historical wire type instead of losing completed history.
            expected = "custom" if kind == "custom_tool_call" else "function"
            if name in kinds and kinds[name] != expected:
                raise ValueError("Historical tool call type differs from advertised tool")
            if not call_id:
                raise ValueError("Tool call IDs must be nonempty")
            wire_id, copy_number = call_id, 1
            while wire_id in wire_ids:
                copy_number += 1
                wire_id = f"{call_id}#{copy_number}"
            wire_ids.add(wire_id)
            unanswered.setdefault(call_id, []).append(wire_id)
            if kind == "custom_tool_call":
                arguments = json.dumps({"input": _text(item.get("input"), label="Custom tool input")},
                                       ensure_ascii=False, separators=(",", ":"))
            else:
                arguments = _text(item.get("arguments"), label="Function arguments")
                try:
                    parsed = json.loads(arguments)
                except ValueError as exc:
                    raise ValueError("Function arguments must be valid JSON") from exc
                if not isinstance(parsed, dict):
                    raise ValueError("Function arguments must be a JSON object")
            current_assistant().setdefault("tool_calls", []).append({"id": wire_id,
                "type": "function", "function": {"name": name, "arguments": arguments}})
        elif kind in ("function_call_output", "custom_tool_call_output"):
            flush()
            call_id = _text(item.get("call_id"), label="Tool output call ID")
            if not unanswered.get(call_id):
                raise ValueError("Tool output has no unique preceding tool call")
            messages.append({"role": "tool", "tool_call_id": unanswered[call_id].pop(0),
                             "content": _content(item.get("output"), label="Tool output")})
        else:
            raise ValueError("Chat bridge does not support input item " + str(kind))
    flush()
    if any(unanswered.values()):
        raise ValueError("Chat bridge requires completed tool outputs before the next model turn")
    payload = {"model": model, "messages": messages, "stream": False,
               "enable_thinking": True, "reasoning_effort": effort}
    translated_tools = []
    for tool in document.get("tools", []):
        if tool["type"] in HOSTED_SEARCH_TOOLS:
            continue
        function = {"name": tool["name"], "description": tool.get("description", "")}
        if tool["type"] == "custom":
            function["parameters"] = {"type": "object", "properties": {"input": {"type": "string",
                "description": "The exact raw input for this custom tool, including newlines."}},
                "required": ["input"], "additionalProperties": False}
            function["description"] += " Send the complete raw tool input in the input string."
        else:
            parameters = tool.get("parameters", {"type": "object", "properties": {}})
            if not isinstance(parameters, dict):
                raise ValueError("Function tool parameters must be an object")
            function["parameters"] = copy.deepcopy(parameters)
            if "strict" in tool:
                function["strict"] = tool["strict"]
        translated_tools.append({"type": "function", "function": function})
    if translated_tools:
        payload["tools"] = translated_tools
    choice = document.get("tool_choice")
    if choice is not None:
        if isinstance(choice, str) and choice in ("auto", "required", "none"):
            payload["tool_choice"] = choice
        elif isinstance(choice, dict) and choice.get("type") in ("function", "custom") and choice.get("name") in kinds:
            payload["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
        else:
            raise ValueError("Unsupported Responses tool choice")
    for name in ("temperature", "top_p", "parallel_tool_calls"):
        if name in document:
            payload[name] = document[name]
    if "max_output_tokens" in document:
        payload["max_tokens"] = document["max_output_tokens"]
    text_configuration = document.get("text") or {}
    if not isinstance(text_configuration, dict) or not isinstance(text_configuration.get("format") or {}, dict):
        raise ValueError("Responses text configuration must be an object")
    if (text_configuration.get("format") or {}).get("type", "text") != "text":
        raise ValueError("Chat bridge does not silently replace structured output formats")
    return payload


def _count(value, name):
    if not isinstance(value, dict):
        raise ValueError("Chat usage must be an object")
    result = value.get(name, 0)
    if type(result) is not int or result < 0:
        raise ValueError("Chat usage counts must be nonnegative integers")
    return result


def chat_to_response(chat_document: dict, original_request: dict, model: str) -> dict:
    """Convert one complete Chat response into the original advertised tool ABI."""
    if not isinstance(chat_document, dict) or not isinstance(chat_document.get("choices"), list) or len(chat_document["choices"]) != 1:
        raise ValueError("Chat bridge requires exactly one model choice")
    choice = chat_document["choices"][0]
    if not isinstance(choice, dict):
        raise ValueError("Chat model choice must be an object")
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("role", "assistant") != "assistant":
        raise ValueError("Chat bridge received an invalid assistant message")
    finish = choice.get("finish_reason")
    if finish not in ("stop", "tool_calls", "length", "content_filter"):
        raise ValueError("Unsupported Chat finish reason")
    kinds = _tool_kinds(original_request)
    identity = hashlib.sha256(json.dumps(chat_document, sort_keys=True, separators=(",", ":"),
                                        ensure_ascii=False).encode()).hexdigest()[:24]
    output = []
    reasoning = message.get("reasoning_content")
    if reasoning is not None:
        reasoning = _text(reasoning, label="Provider reasoning")
        if reasoning:
            output.append({"type": "reasoning", "id": "rs_bridge_" + identity,
                "summary": [{"type": "summary_text", "text": reasoning}],
                "encrypted_content": REASONING_PREFIX + base64.urlsafe_b64encode(reasoning.encode()).decode()})
    content = message.get("content")
    if content is not None and content != "":
        text = _content(content, label="Chat assistant content")
        output.append({"type": "message", "id": "msg_bridge_" + identity, "status": "completed",
                       "role": "assistant", "content": [{"type": "output_text", "text": text, "annotations": []}]})
    if message.get("refusal") is not None:
        output.append({"type": "message", "id": "msg_refusal_" + identity, "status": "completed",
            "role": "assistant", "content": [{"type": "refusal", "refusal": _text(message["refusal"], label="Refusal")}]})
    seen = set()
    calls = message.get("tool_calls", [])
    if not isinstance(calls, list):
        raise ValueError("Chat tool calls must be a list")
    for index, call in enumerate(calls):
        if not isinstance(call, dict) or call.get("type") != "function" or not isinstance(call.get("function"), dict):
            raise ValueError("Unknown Chat tool call type")
        function = call["function"]
        name = _text(function.get("name"), label="Chat tool name")
        call_id = _text(call.get("id"), label="Chat tool call ID")
        if name not in kinds or not call_id or call_id in seen:
            raise ValueError("Chat tool call is unknown or has a duplicate ID")
        seen.add(call_id)
        arguments = _text(function.get("arguments"), label="Chat tool arguments")
        try:
            parsed = json.loads(arguments)
        except ValueError as exc:
            raise ValueError("Chat tool arguments must be valid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Chat tool arguments must be a JSON object")
        item = {"id": "call_bridge_" + identity + "_" + str(index), "call_id": call_id,
                "name": name, "status": "completed"}
        if kinds[name] == "custom":
            if set(parsed) != {"input"}:
                raise ValueError("Custom Chat tool arguments must contain only input")
            item.update(type="custom_tool_call", input=_text(parsed["input"], label="Raw custom input"))
        else:
            item.update(type="function_call", arguments=arguments)
        output.append(item)
    if finish == "tool_calls" and not calls:
        raise ValueError("Chat tool finish has no tool calls")
    raw_usage = chat_document.get("usage") or {}
    usage = {"input_tokens": _count(raw_usage, "prompt_tokens"),
             "output_tokens": _count(raw_usage, "completion_tokens"),
             "total_tokens": _count(raw_usage, "total_tokens"),
             "input_tokens_details": {"cached_tokens": _count(raw_usage.get("prompt_tokens_details") or {}, "cached_tokens")},
             "output_tokens_details": {"reasoning_tokens": _count(raw_usage.get("completion_tokens_details") or {}, "reasoning_tokens")}}
    if "total_tokens" not in raw_usage:
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    incomplete = finish in ("length", "content_filter")
    return {"id": "resp_bridge_" + identity, "object": "response",
            "created_at": chat_document.get("created", int(time.time())), "model": model,
            "status": "incomplete" if incomplete else "completed", "error": None,
            "incomplete_details": {"reason": "max_output_tokens" if finish == "length" else "content_filter"} if incomplete else None,
            "output": output, "usage": usage, "parallel_tool_calls": original_request.get("parallel_tool_calls", True),
            "store": False}


def response_sse_events(response: dict) -> Iterator[dict]:
    """Emit official Responses event shapes after a bounded complete Chat call."""
    if response.get("status") not in ("completed", "incomplete"):
        raise ValueError("Bridge SSE requires a terminal Response")
    number = 0
    def event(kind, **fields):
        nonlocal number
        item = {"type": kind, "sequence_number": number, **fields}
        number += 1
        return item
    opening = dict(response, status="in_progress", output=[], usage=None, incomplete_details=None)
    yield event("response.created", response=opening)
    yield event("response.in_progress", response=copy.deepcopy(opening))
    for output_index, item in enumerate(response["output"]):
        added = copy.deepcopy(item)
        added["status"] = "in_progress"
        kind = item["type"]
        if kind == "message":
            added["content"] = []
        elif kind == "reasoning":
            added["summary"] = []
            added.pop("encrypted_content", None)
        elif kind == "function_call":
            added["arguments"] = ""
        elif kind == "custom_tool_call":
            added["input"] = ""
        else:
            raise ValueError("Unknown bridge output item")
        yield event("response.output_item.added", output_index=output_index, item=added)
        common = {"item_id": item["id"], "output_index": output_index}
        if kind == "reasoning":
            for summary_index, part in enumerate(item["summary"]):
                fields = dict(common, summary_index=summary_index)
                yield event("response.reasoning_summary_part.added", **fields, part={"type": "summary_text", "text": ""})
                yield event("response.reasoning_summary_text.delta", **fields, delta=part["text"])
                yield event("response.reasoning_summary_text.done", **fields, text=part["text"])
                yield event("response.reasoning_summary_part.done", **fields, part=copy.deepcopy(part))
        elif kind == "message":
            for content_index, part in enumerate(item["content"]):
                fields = dict(common, content_index=content_index)
                field = "text" if part["type"] == "output_text" else "refusal"
                empty = dict(part, **{field: ""})
                yield event("response.content_part.added", **fields, part=empty)
                yield event("response." + ("output_text" if field == "text" else "refusal") + ".delta", **fields, delta=part[field])
                yield event("response." + ("output_text" if field == "text" else "refusal") + ".done", **fields, **{field: part[field]})
                yield event("response.content_part.done", **fields, part=copy.deepcopy(part))
        elif kind == "function_call":
            yield event("response.function_call_arguments.delta", **common, delta=item["arguments"])
            yield event("response.function_call_arguments.done", **common, name=item["name"], arguments=item["arguments"])
        else:
            yield event("response.custom_tool_call_input.delta", **common, delta=item["input"])
            yield event("response.custom_tool_call_input.done", **common, input=item["input"])
        yield event("response.output_item.done", output_index=output_index, item=copy.deepcopy(item))
    yield event("response." + response["status"], response=copy.deepcopy(response))

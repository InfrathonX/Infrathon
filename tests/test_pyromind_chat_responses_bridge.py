import copy
import json

import pytest

from sforge.harness.relay.chat_responses_bridge import (
    REASONING_PREFIX, chat_to_response, request_to_chat, response_sse_events,
)


TOOLS = [
    {"type": "function", "name": "exec_command", "description": "Run a command",
     "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"], "additionalProperties": False}, "strict": True},
    {"type": "custom", "name": "apply_patch", "description": "Apply a patch",
     "format": {"type": "grammar", "syntax": "lark", "definition": "start: /[\\s\\S]+/"}},
]
PATCH = "*** Begin Patch\n*** Add File: ready.txt\n+READY\n*** End Patch\n"


def chat(message, finish="tool_calls"):
    return {"id": "chat-example", "created": 1, "choices": [
        {"index": 0, "message": {"role": "assistant", **message}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                  "completion_tokens_details": {"reasoning_tokens": 12}}}


def tool_response():
    original = {"input": "Make ready.txt and verify it.", "tools": TOOLS}
    result = chat_to_response(chat({"reasoning_content": "First create; then inspect.",
        "tool_calls": [{"id": "patch-id", "type": "function", "function": {
            "name": "apply_patch", "arguments": json.dumps({"input": PATCH})}},
                       {"id": "exec-id", "type": "function", "function": {
            "name": "exec_command", "arguments": '{"cmd":"cat ready.txt"}'}}]}), original, "glm")
    return original, result


def test_custom_and_function_roundtrip_preserves_ids_input_and_reasoning():
    original, response = tool_response()
    reasoning, patch, execute = response["output"]
    assert reasoning["encrypted_content"].startswith(REASONING_PREFIX)
    assert patch["type"] == "custom_tool_call" and patch["input"] == PATCH
    assert patch["call_id"] == "patch-id"
    assert execute["type"] == "function_call" and execute["call_id"] == "exec-id"
    next_request = {**original, "input": [{"role": "user", "content": original["input"]},
        *response["output"], {"type": "custom_tool_call_output", "call_id": "patch-id", "output": "Success"},
        {"type": "function_call_output", "call_id": "exec-id", "output": "READY"}]}
    translated = request_to_chat(next_request, "ZHIPU/GLM-5.3-Flash")
    assert translated["model"] == "ZHIPU/GLM-5.3-Flash"
    assert translated["reasoning_effort"] == "max" and translated["enable_thinking"] is True
    assert translated["stream"] is False
    assistant = translated["messages"][1]
    assert assistant["reasoning_content"] == "First create; then inspect."
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"])["input"] == PATCH
    assert [entry["tool_call_id"] for entry in translated["messages"][2:]] == ["patch-id", "exec-id"]
    assert translated["messages"][-1]["content"] == "READY"
    assert translated["tools"][1]["function"]["parameters"]["required"] == ["input"]
    assert "format" not in translated["tools"][1]
    assert response["usage"]["output_tokens_details"]["reasoning_tokens"] == 12


def test_sse_reassembles_tools_and_exact_terminal_response():
    _, response = tool_response()
    events = list(response_sse_events(response))
    assert [entry["sequence_number"] for entry in events] == list(range(len(events)))
    assert events[0]["type"] == "response.created"
    assert events[0]["response"]["output"] == []
    assert events[-1] == {"type": "response.completed", "sequence_number": len(events) - 1,
                          "response": response}
    reconstructed = [entry["item"] for entry in events if entry["type"] == "response.output_item.done"]
    assert reconstructed == response["output"]
    assert next(entry for entry in events if entry["type"] == "response.custom_tool_call_input.done")["input"] == PATCH
    added = [entry["item"] for entry in events if entry["type"] == "response.output_item.added"]
    assert added[1]["input"] == "" and added[2]["arguments"] == ""
    assert "encrypted_content" not in added[0]


def test_message_text_and_incomplete_response():
    response = chat_to_response(chat({"content": "partial answer", "reasoning_content": "working"},
                                    finish="length"), {"input": "question"}, "glm")
    events = list(response_sse_events(response))
    assert response["incomplete_details"] == {"reason": "max_output_tokens"}
    assert events[-1]["type"] == "response.incomplete"
    assert next(entry for entry in events if entry["type"] == "response.output_text.delta")["delta"] == "partial answer"


def test_roles_system_instructions_and_newline_content_are_preserved():
    result = request_to_chat({"instructions": "system instruction", "input": [
        {"role": "developer", "content": [{"type": "input_text", "text": "policy"}]},
        {"role": "user", "content": [{"type": "input_text", "text": "a"}, {"type": "input_text", "text": "b"}]}],
        "tools": TOOLS, "tool_choice": {"type": "custom", "name": "apply_patch"}, "max_output_tokens": 512}, "glm")
    assert result["messages"] == [{"role": "system", "content": "system instruction"},
        {"role": "system", "content": "policy"}, {"role": "user", "content": "a\nb"}]
    assert result["tool_choice"] == {"type": "function", "function": {"name": "apply_patch"}}
    assert result["max_tokens"] == 512


@pytest.mark.parametrize("document", [
    {"input": [{"type": "web_search_call"}]},
    {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "https://example.test/x"}]}]},
    {"input": "x", "tools": [{"type": "namespace", "name": "multi_agent_v1", "tools": []}]},
    {"input": "x", "previous_response_id": "old"},
    {"input": [{"type": "reasoning", "encrypted_content": "another-provider"}]},
    {"input": [{"type": "function_call_output", "call_id": "unknown", "output": "x"}]},
    {"input": [{"type": "function_call", "call_id": "x", "name": "exec_command", "arguments": "{}"}]},
    {"input": "x", "text": {"format": {"type": "json_schema"}}},
])
def test_unsupported_or_unbound_requests_fail_closed(document):
    with pytest.raises(ValueError):
        request_to_chat(document, "glm")


@pytest.mark.parametrize("name,arguments", [
    ("unknown", "{}"), ("apply_patch", "not-json"), ("apply_patch", '{"other":"x"}'),
    ("apply_patch", '{"input":123}'), ("exec_command", "[]"),
])
def test_malformed_provider_tool_calls_do_not_reach_codex(name, arguments):
    with pytest.raises(ValueError):
        chat_to_response(chat({"tool_calls": [{"id": "x", "type": "function",
             "function": {"name": name, "arguments": arguments}}]}), {"tools": TOOLS}, "glm")


@pytest.mark.parametrize("hosted", [{"type": "web_search", "external_web_access": True},
                                    {"type": "web_search_preview"}])
def test_hosted_web_search_is_omitted_for_chat_providers(hosted):
    # Codex sends hosted search whenever the task allows internet; a Chat
    # provider cannot run it, so the request must still go through.
    body = request_to_chat({"input": "x", "tools": [*TOOLS, hosted]}, "glm")
    assert [tool["function"]["name"] for tool in body["tools"]] == ["exec_command", "apply_patch"]
    response = chat_to_response(chat({"content": "READY"}, finish="stop"),
                                {"input": "x", "tools": [*TOOLS, hosted]}, "glm")
    assert response["output"][-1]["content"][0]["text"] == "READY"


def test_conversion_does_not_mutate_either_protocol_document():
    original, response = tool_response()
    initial = copy.deepcopy(original)
    request_to_chat(original, "glm")
    assert original == initial
    before = copy.deepcopy(response)
    list(response_sse_events(response))
    assert response == before


def test_reasoning_summary_fallback_survives_complete_tool_turn():
    request = {"input": [{"type": "reasoning", "summary": [{"type": "summary_text", "text": "plan"}]},
        {"type": "function_call", "call_id": "x", "name": "exec_command", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "x", "output": "ok"}], "tools": TOOLS}
    assert request_to_chat(request, "glm")["messages"][0]["reasoning_content"] == "plan"


def test_a_provider_reused_call_id_gets_a_fresh_wire_id():
    # GLM on MaaS once returned a call ID it had used 15 minutes earlier; refusing the whole
    # history stalled the run for good, because every resumed turn carried the duplicate.
    call = {"type": "function_call", "name": "exec_command", "arguments": '{"cmd":"ls"}'}
    document = {"tools": TOOLS, "input": [
        {"role": "user", "content": "go"},
        {**call, "call_id": "call_-7163"}, {"type": "function_call_output", "call_id": "call_-7163", "output": "a"},
        {**call, "call_id": "call_-7163"}, {"type": "function_call_output", "call_id": "call_-7163", "output": "b"}]}
    translated = request_to_chat(document, "glm")
    calls = [m["tool_calls"][0]["id"] for m in translated["messages"] if m.get("tool_calls")]
    outputs = [(m["tool_call_id"], m["content"]) for m in translated["messages"] if m["role"] == "tool"]
    assert calls == ["call_-7163", "call_-7163#2"]
    assert outputs == [("call_-7163", "a"), ("call_-7163#2", "b")]
    unanswered = {**document, "input": document["input"][:4]}
    with pytest.raises(ValueError, match="completed tool outputs"):
        request_to_chat(unanswered, "glm")
    extra = {**document, "input": [*document["input"], document["input"][-1]]}
    with pytest.raises(ValueError, match="no unique preceding"):
        request_to_chat(extra, "glm")

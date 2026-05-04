"""Direct Ollama API client — bypasses litellm for proper tool call support.

Litellm's Ollama chat transformation corrupts streaming tool calls by
placing ``index`` inside ``function.index`` instead of at the top level,
which prevents the agent loop from accumulating tool calls correctly.
We hit Ollama's ``/api/chat`` endpoint directly to avoid this.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator

import httpx


# Sentinel for unset usage

class _UsageInfo:
    __slots__ = ("prompt_tokens", "completion_tokens", "total_tokens")
    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


# ---------------------------------------------------------------------------
# Streaming response wrapper — mimics litellm's API
# ---------------------------------------------------------------------------

class _DeltaToolCall:
    """Mirrors litellm's delta tool call shape."""
    __slots__ = ("index", "id", "type", "function_name", "function_args")

    def __init__(self, index: int, id: str, function_name: str, function_args: str):
        self.index = index
        self.id = id
        self.type = "function"
        self.function_name = function_name
        self.function_args = function_args

    @property
    def function(self):
        return _FunctionRef(self.function_name, self.function_args)


class _FunctionRef:
    __slots__ = ("name", "arguments")
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _Delta:
    """Mimics litellm delta object with content + tool_calls."""
    __slots__ = ("content", "reasoning_content", "tool_calls")

    def __init__(
        self,
        content: str | None = None,
        reasoning_content: str | None = None,
        tool_calls: list[_DeltaToolCall] | None = None,
    ):
        self.content = content
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls or []


class _Choice:
    __slots__ = ("delta", "finish_reason", "index")

    def __init__(self, delta: _Delta, finish_reason: str | None = None, index: int = 0):
        self.delta = delta
        self.finish_reason = finish_reason
        self.index = index


class _Chunk:
    """Streaming chunk that mimics litellm's ModelResponseStream shape."""
    __slots__ = ("choices", "usage", "model")

    def __init__(self, delta: _Delta, finish_reason: str | None = None):
        self.choices = [_Choice(delta, finish_reason)]
        self.usage = None
        self.model = ""

    @staticmethod
    def make_usage(prompt_tokens: int, completion_tokens: int) -> _Chunk:
        c = _Chunk(_Delta())
        c.choices = []
        c.usage = _Usage(prompt_tokens, completion_tokens)
        return c


class _Usage:
    __slots__ = ("prompt_tokens", "completion_tokens", "total_tokens")
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class _OllamaNonStreamingResult:
    """Mimics litellm's non-streaming response."""
    def __init__(
        self,
        content: str | None,
        tool_calls: list[dict] | None,
        finish_reason: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ):
        self._content = content
        self._tool_calls = tool_calls
        self.finish_reason = finish_reason
        self.usage = _UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    @property
    def choices(self):
        return [_NonStreamingChoice(self._content, self._tool_calls, self.finish_reason)]


class _NonStreamingChoice:
    def __init__(self, content, tool_calls, finish_reason):
        self.message = _NonStreamingMessage(content, tool_calls)
        self.finish_reason = finish_reason
        self.index = 0


class _NonStreamingMessage:
    def __init__(self, content, tool_calls):
        self.content = content
        self.tool_calls = tool_calls


# ---------------------------------------------------------------------------
# Core client
# ---------------------------------------------------------------------------

def _ollama_base_url() -> str:
    import os
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def _extract_ollama_model(raw_name: str) -> str:
    """Strip the ``ollama/`` prefix if present."""
    if raw_name.startswith("ollama/"):
        return raw_name[len("ollama/"):]
    return raw_name


def _build_ollama_request(
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    llm_params: dict,
    *,
    stream: bool,
) -> dict:
    """Build the Ollama /api/chat request body."""
    model_name = _extract_ollama_model(model)

    # Map litellm-style params to Ollama options
    options: dict[str, Any] = {}
    for param in ("temperature", "top_p", "seed", "num_predict", "repeat_penalty"):
        if param in llm_params:
            options[param] = llm_params[param]
    if "max_tokens" in llm_params:
        options["num_predict"] = llm_params["max_tokens"]

    body: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "stream": stream,
    }

    if tools:
        # Ollama uses the same OpenAI-style tool format
        body["tools"] = tools

    if options:
        body["options"] = options

    return body


async def _ollama_stream(
    api_base: str,
    request_body: dict,
    timeout: float = 600.0,
) -> AsyncIterator[_Chunk]:
    """Stream chat completions directly from Ollama."""
    url = f"{api_base}/api/chat"
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        async with client.stream("POST", url, json=request_body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk_data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Extract thinking / content
                message = chunk_data.get("message") or {}
                thinking = message.get("thinking")
                content = message.get("content")

                # Extract tool calls
                raw_tool_calls = message.get("tool_calls")
                tool_calls: list[_DeltaToolCall] = []
                if raw_tool_calls:
                    for tc in raw_tool_calls:
                        fn = tc.get("function") or {}
                        fn_name = fn.get("name") or ""
                        fn_args = fn.get("arguments") or {}
                        if isinstance(fn_args, dict):
                            fn_args_str = json.dumps(fn_args)
                        else:
                            fn_args_str = str(fn_args)
                        # Use function.index from Ollama, fallback to 0
                        idx = fn.get("index", 0)
                        tc_id = tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"
                        tool_calls.append(_DeltaToolCall(
                            index=idx,
                            id=tc_id,
                            function_name=fn_name,
                            function_args=fn_args_str,
                        ))

                delta = _Delta(
                    content=content if content else None,
                    reasoning_content=thinking if thinking else None,
                    tool_calls=tool_calls if tool_calls else None,
                )

                done = chunk_data.get("done", False)
                finish_reason = chunk_data.get("done_reason", "stop") if done else None
                if done and tool_calls:
                    finish_reason = "tool_calls"

                yield _Chunk(delta, finish_reason)

                if done:
                    # Emit usage on final chunk
                    prompt_tokens = chunk_data.get("prompt_eval_count", 0)
                    completion_tokens = chunk_data.get("eval_count", 0)
                    yield _Chunk.make_usage(prompt_tokens, completion_tokens)


async def ollama_chat_streaming(
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    llm_params: dict,
    *,
    timeout: float = 600.0,
) -> AsyncIterator[_Chunk]:
    """Stream chat completions from Ollama — returns chunks mimicking litellm."""
    api_base = _ollama_base_url()
    request_body = _build_ollama_request(model, messages, tools, llm_params, stream=True)
    async for chunk in _ollama_stream(api_base, request_body, timeout):
        yield chunk


async def ollama_chat_non_streaming(
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    llm_params: dict,
    *,
    timeout: float = 600.0,
) -> _OllamaNonStreamingResult:
    """Non-streaming chat completion from Ollama."""
    api_base = _ollama_base_url()
    request_body = _build_ollama_request(model, messages, tools, llm_params, stream=False)
    url = f"{api_base}/api/chat"

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        resp = await client.post(url, json=request_body)
        resp.raise_for_status()
        data = resp.json()

    message = data.get("message") or {}
    content = message.get("content")
    raw_tool_calls = message.get("tool_calls")

    tool_calls: list[dict] | None = None
    if raw_tool_calls:
        tool_calls = []
        for tc in raw_tool_calls:
            fn = tc.get("function") or {}
            tc_id = tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"
            tool_calls.append({
                "id": tc_id,
                "type": "function",
                "function": {
                    "name": fn.get("name", ""),
                    "arguments": json.dumps(fn.get("arguments", {})) if isinstance(fn.get("arguments", {}), dict) else str(fn.get("arguments", "")),
                },
                "index": fn.get("index", 0),
            })

    prompt_tokens = data.get("prompt_eval_count", 0)
    completion_tokens = data.get("eval_count", 0)
    done_reason = data.get("done_reason", "stop")
    if tool_calls:
        done_reason = "tool_calls"

    return _OllamaNonStreamingResult(
        content=content if content else None,
        tool_calls=tool_calls,
        finish_reason=done_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


async def ollama_chat_compact(
    model: str,
    messages: list[dict],
    *,
    max_tokens: int = 2000,
    temperature: float = 0.3,
    timeout: float = 300.0,
) -> tuple[str, int]:
    """Simple chat for compaction/restore — returns (content, completion_tokens).

    Does NOT use litellm or tool calls. Pure ollama /api/chat.
    """
    model_name = _extract_ollama_model(model)
    api_base = _ollama_base_url()
    url = f"{api_base}/api/chat"
    body: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
        },
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        resp = await client.post(url, json=body)
        resp.raise_for_status()
        data = resp.json()

    message = data.get("message") or {}
    content = message.get("content") or ""
    completion_tokens = data.get("eval_count", 0)
    return content, completion_tokens


def _messages_to_dict_compact(messages) -> list[dict]:
    """Convert litellm Message objects to plain dicts for compaction.

    Strips litellm-only fields. No tool call conversion needed —
    compaction summaries don't use tools.
    """
    result: list[dict] = []
    for m in messages:
        if hasattr(m, 'model_dump'):
            d = m.model_dump(exclude_none=True)
        elif isinstance(m, dict):
            d = dict(m)
        else:
            d = {"role": getattr(m, "role", "user"), "content": str(m)}
        # Strip fields Ollama doesn't accept
        for key in ("function_call", "provider_specific_fields", "tool_calls", "tool_call_id", "name"):
            d.pop(key, None)
        result.append(d)
    return result

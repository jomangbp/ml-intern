"""loop
Main agent implementation with integrated tool system and MCP support
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field

from litellm import ChatCompletionMessageToolCall, Message, acompletion
from litellm.exceptions import ContextWindowExceededError

from agent.config import Config
from agent.core import telemetry
from agent.core.doom_loop import check_for_doom_loop
from agent.core.codex_responses import codex_responses_completion, is_codex_responses_params
from agent.core.llm_params import _resolve_llm_params
from agent.core.ollama_client import (
    ollama_chat_non_streaming,
    ollama_chat_streaming,
)
from agent.core.overflow import is_context_overflow, is_silent_overflow
from agent.core.prompt_caching import with_prompt_caching
from agent.core.session import Event, OpType, Session
from agent.core.tools import ToolRouter
from agent.tools.jobs_tool import CPU_FLAVORS

logger = logging.getLogger(__name__)

ToolCall = ChatCompletionMessageToolCall

_MALFORMED_TOOL_PREFIX = "ERROR: Tool call to '"
_MALFORMED_TOOL_SUFFIX = "' had malformed JSON arguments"


def _malformed_tool_name(message: Message) -> str | None:
    """Return the tool name for malformed-json tool-result messages."""
    if getattr(message, "role", None) != "tool":
        return None
    content = getattr(message, "content", None)
    if not isinstance(content, str):
        return None
    if not content.startswith(_MALFORMED_TOOL_PREFIX):
        return None
    end = content.find(_MALFORMED_TOOL_SUFFIX, len(_MALFORMED_TOOL_PREFIX))
    if end == -1:
        return None
    return content[len(_MALFORMED_TOOL_PREFIX):end]


def _detect_repeated_malformed(
    items: list[Message], threshold: int = 2,
) -> str | None:
    """Return the repeated malformed tool name if the tail contains a streak.

    Walk backward over the current conversation tail. A streak counts only
    consecutive malformed tool-result messages for the same tool; any other
    tool result breaks it.
    """
    if threshold <= 0:
        return None

    streak_tool: str | None = None
    streak = 0

    for item in reversed(items):
        if getattr(item, "role", None) != "tool":
            continue

        malformed_tool = _malformed_tool_name(item)
        if malformed_tool is None:
            break

        if streak_tool is None:
            streak_tool = malformed_tool
            streak = 1
        elif malformed_tool == streak_tool:
            streak += 1
        else:
            break

        if streak >= threshold:
            return streak_tool

    return None


def _validate_tool_args(tool_args: dict) -> tuple[bool, str | None]:
    """
    Validate tool arguments structure.

    Returns:
        (is_valid, error_message)
    """
    args = tool_args.get("args", {})
    # Sometimes LLM passes args as string instead of dict
    if isinstance(args, str):
        return (
            False,
            f"Tool call error: 'args' must be a JSON object, not a string. You passed: {repr(args)}",
        )
    if not isinstance(args, dict) and args is not None:
        return (
            False,
            f"Tool call error: 'args' must be a JSON object. You passed type: {type(args).__name__}",
        )
    return True, None


def _needs_approval(
    tool_name: str, tool_args: dict, config: Config | None = None
) -> bool:
    """Check if a tool call requires user approval before execution."""
    # Yolo mode: skip all approvals
    if config and config.yolo_mode:
        return False

    # If args are malformed, skip approval (validation error will be shown later)
    args_valid, _ = _validate_tool_args(tool_args)
    if not args_valid:
        return False

    if tool_name == "sandbox_create":
        return True

    if tool_name == "hf_jobs":
        operation = tool_args.get("operation", "")
        if operation not in ["run", "uv", "scheduled run", "scheduled uv"]:
            return False

        # Check if this is a CPU-only job
        # hardware_flavor is at top level of tool_args, not nested in args
        hardware_flavor = (
            tool_args.get("hardware_flavor")
            or tool_args.get("flavor")
            or tool_args.get("hardware")
            or "cpu-basic"
        )
        is_cpu_job = hardware_flavor in CPU_FLAVORS

        if is_cpu_job:
            if config and not config.confirm_cpu_jobs:
                return False
            return True

        return True

    # Check for file upload operations (hf_private_repos or other tools)
    if tool_name == "hf_private_repos":
        operation = tool_args.get("operation", "")
        if operation == "upload_file":
            if config and config.auto_file_upload:
                return False
            return True
        # Other operations (create_repo, etc.) always require approval
        if operation in ["create_repo"]:
            return True

    # hf_repo_files: upload (can overwrite) and delete require approval
    if tool_name == "hf_repo_files":
        operation = tool_args.get("operation", "")
        if operation in ["upload", "delete"]:
            return True

    # hf_repo_git: destructive operations require approval
    if tool_name == "hf_repo_git":
        operation = tool_args.get("operation", "")
        if operation in [
            "delete_branch",
            "delete_tag",
            "merge_pr",
            "create_repo",
            "update_repo",
        ]:
            return True

    return False


# -- LLM retry constants --------------------------------------------------


def _is_rate_limit_error(error: Exception) -> bool:
    """Return True for rate-limit / quota-bucket style provider errors."""
    err_str = str(error).lower()
    rate_limit_patterns = [
        "429",
        "rate limit",
        "rate_limit",
        "too many requests",
        "too many tokens",
        "request limit",
        "throttl",
    ]
    return any(pattern in err_str for pattern in rate_limit_patterns)


def _is_cloud_overloaded(error: Exception) -> bool:
    """Detect Ollama cloud overload specifically — needs persistent retry."""
    msg = str(error).lower()
    return any(p in msg for p in ["overloaded", "server overloaded", "capacity"])


def _persistent_retry_delay(error: Exception, attempt: int) -> int:
    """Exponential backoff for persistent retries, capped at 5 minutes."""
    if _is_rate_limit_error(error):
        base = 30
    elif _is_cloud_overloaded(error):
        base = 15
    else:
        base = 5
    return min(base * (2 ** attempt), 300)


def _is_context_overflow_error(error: Exception) -> bool:
    """Return True when the prompt exceeded the model's context window."""
    if isinstance(error, ContextWindowExceededError):
        return True

    err_str = str(error).lower()
    overflow_patterns = [
        "context window exceeded",
        "maximum context length",
        "max context length",
        "prompt is too long",
        "context length exceeded",
        "too many input tokens",
        "input is too long",
    ]
    return any(pattern in err_str for pattern in overflow_patterns)


def _is_transient_error(error: Exception) -> bool:
    """Return True for errors that are likely transient and worth retrying."""
    err_str = str(error).lower()
    transient_patterns = [
        "timeout", "timed out",
        "503", "service unavailable",
        "502", "bad gateway",
        "500", "internal server error",
        "overloaded", "capacity",
        "connection reset", "connection refused", "connection error",
        "eof", "broken pipe", "empty response",
    ]
    return _is_rate_limit_error(error) or any(pattern in err_str for pattern in transient_patterns)


def _is_effort_config_error(error: Exception) -> bool:
    """Catch the two 400s the effort probe also handles — thinking
    unsupported for this model, or the specific effort level invalid.

    This is our safety net for the case where ``/effort`` was changed
    mid-conversation (which clears the probe cache) and the new level
    doesn't work for the current model. We heal the cache and retry once.
    """
    from agent.core.effort_probe import _is_invalid_effort, _is_thinking_unsupported
    return _is_thinking_unsupported(error) or _is_invalid_effort(error)


async def _heal_effort_and_rebuild_params(
    session: Session, error: Exception, llm_params: dict,
) -> dict:
    """Update the session's effort cache based on ``error`` and return new
    llm_params. Called only when ``_is_effort_config_error(error)`` is True.

    Two branches:
      • thinking-unsupported → cache ``None`` for this model, next call
        strips thinking entirely
      • invalid-effort → re-run the full cascade probe; the result lands
        in the cache
    """
    from agent.core.effort_probe import ProbeInconclusive, _is_thinking_unsupported, probe_effort

    model = session.config.model_name
    if _is_thinking_unsupported(error):
        session.model_effective_effort[model] = None
        logger.info("healed: %s doesn't support thinking — stripped", model)
    else:
        try:
            outcome = await probe_effort(
                model, session.config.reasoning_effort, session.hf_token,
            )
            session.model_effective_effort[model] = outcome.effective_effort
            logger.info(
                "healed: %s effort cascade → %s", model, outcome.effective_effort,
            )
        except ProbeInconclusive:
            # Transient during healing — strip thinking for safety, next
            # call will either succeed or surface the real error.
            session.model_effective_effort[model] = None
            logger.info("healed: %s probe inconclusive — stripped", model)

    return _resolve_llm_params(
        model,
        session.hf_token,
        reasoning_effort=session.effective_effort_for(model),
    )


def _friendly_error_message(error: Exception) -> str | None:
    """Return a user-friendly message for known error types, or None to fall back to traceback."""
    err_str = str(error).lower()

    if "authentication" in err_str or "unauthorized" in err_str or "invalid x-api-key" in err_str:
        return (
            "Authentication failed — your API key is missing or invalid.\n\n"
            "To fix this, set the API key for your model provider:\n"
            "  • Anthropic:   export ANTHROPIC_API_KEY=sk-...\n"
            "  • OpenAI:      export OPENAI_API_KEY=sk-...\n"
            "  • HF Router:   export HF_TOKEN=hf_...\n\n"
            "You can also add it to a .env file in the project root.\n"
            "To switch models, use the /model command."
        )

    if "insufficient" in err_str and "credit" in err_str:
        return (
            "Insufficient API credits. Please check your account balance "
            "at your model provider's dashboard."
        )

    if "not supported by provider" in err_str or "no provider supports" in err_str:
        return (
            "The model isn't served by the provider you pinned.\n\n"
            "Drop the ':<provider>' suffix to let the HF router auto-pick a "
            "provider, or use '/model' (no arg) to see which providers host "
            "which models."
        )

    if "model_not_found" in err_str or (
        "model" in err_str
        and ("not found" in err_str or "does not exist" in err_str)
    ):
        return (
            "Model not found. Use '/model' to list suggestions, or paste an "
            "HF model id like 'MiniMaxAI/MiniMax-M2.7'. Availability is shown "
            "when you switch."
        )

    return None


async def _compact_and_notify(session: Session) -> None:
    """Run compaction and send event if context was reduced."""
    cm = session.context_manager
    old_usage = cm.running_context_usage
    logger.debug(
        "Compaction check: usage=%d, max=%d, threshold=%d, needs_compact=%s",
        old_usage, cm.model_max_tokens, cm.compaction_threshold, cm.needs_compaction,
    )
    if cm.needs_compaction:
        await session.send_event(
            Event(
                event_type="tool_log",
                data={"tool": "system", "log": "Compacting context..."},
            )
        )
    stats = await cm.compact(
        model_name=session.config.model_name,
        tool_specs=session.tool_router.get_tool_specs_for_llm(),
        hf_token=session.hf_token,
        provider_keys=getattr(session, "provider_keys", None),
    )
    if stats is None:
        return
    new_usage = stats["tokens_after"]
    saved = stats["tokens_saved"]
    logger.warning(
        "Context compacted: %d -> %d tokens (saved %d, max=%d, %d -> %d messages)",
        old_usage, new_usage, saved, cm.model_max_tokens,
        stats["messages_before"], stats["messages_after"],
    )
    await session.send_event(
        Event(
            event_type="tool_log",
            data={"tool": "system", "log": f"Context compacted: {old_usage} -> {new_usage} tokens ({saved} saved)"},
        )
    )
    await session.send_event(
        Event(
            event_type="compacted",
            data={
                "old_tokens": old_usage,
                "new_tokens": new_usage,
                "tokens_saved": saved,
                "messages_before": stats["messages_before"],
                "messages_after": stats["messages_after"],
                "messages_removed": stats["messages_removed"],
                "summary": stats["summary"],
            },
        )
    )


async def _cleanup_on_cancel(session: Session) -> None:
    """Kill sandbox processes and cancel HF jobs when the user interrupts."""
    # Kill active sandbox processes
    sandbox = getattr(session, "sandbox", None)
    if sandbox:
        try:
            await asyncio.to_thread(sandbox.kill_all)
            logger.info("Killed sandbox processes on cancel")
        except Exception as e:
            logger.warning("Failed to kill sandbox processes: %s", e)

    # Cancel running HF jobs
    job_ids = list(session._running_job_ids)
    if job_ids:
        from huggingface_hub import HfApi

        api = HfApi(token=session.hf_token)
        for job_id in job_ids:
            try:
                await asyncio.to_thread(api.cancel_job, job_id=job_id)
                logger.info("Cancelled HF job %s on interrupt", job_id)
            except Exception as e:
                logger.warning("Failed to cancel HF job %s: %s", job_id, e)
        session._running_job_ids.clear()


import hashlib


def _find_json_span(text: str, start: int) -> tuple[int, int] | None:
    """Find a balanced JSON object or array starting at or after `start`.

    Returns (obj_start, obj_end) inclusive indices, or None if nothing found.
    Skips non-JSON characters to find the next `{` or `[`.
    """
    import re as _re
    # Look for the next `{` or `[` starting from `start`
    m = _re.search(r'[\[{]', text[start:])
    if not m:
        return None
    obj_start = start + m.start()
    open_char = text[obj_start]
    close_char = '}' if open_char == '{' else ']'
    depth = 0
    for i in range(obj_start, len(text)):
        if text[i] == open_char:
            depth += 1
        elif text[i] == close_char:
            depth -= 1
            if depth <= 0:
                return (obj_start, i)
    return None


def _extract_tool_calls_from_content(content: str | None) -> tuple[dict[int, dict], str | None]:
    """Detect tool calls emitted as raw JSON text and strip them from content.

    Some models (especially smaller Ollama models) do not support the API-level
    ``tools`` parameter and instead output the tool call as a JSON object in the
    text response.  This helper tries to parse that JSON and returns a
    ``tool_calls_acc`` dict in the same format the streaming/non-streaming
    paths produce, **plus** the content with the JSON tool call(s) removed so the
    assistant message doesn't show raw JSON.

    Handles:
    • Bare JSON objects: {"name": "bash", "arguments": {...}}
    • JSON with a ``function`` wrapper: {"function": {"name": ..., "arguments": ...}}
    • Embedded JSON inside markdown fences or surrounding text
    • Arrays of tool calls
    • Multiple concatenated JSON objects (iterated until none left)
    • "Tool Calls: [...]" or "tool_call: {...}" prefixed patterns

    Returns (tool_calls_dict, cleaned_content).  cleaned_content may be None
    if the entire text consisted of tool call JSON.
    """
    if not content or not content.strip():
        return {}, content

    import re

    detected: dict[int, dict] = {}
    text = content
    # Track ranges to remove: list of (start, end) indices (inclusive end)
    remove_ranges: list[tuple[int, int]] = []

    def _try_parse(obj: dict | list) -> None:
        """Recursively extract tool calls from a parsed JSON object."""
        nonlocal detected
        if isinstance(obj, list):
            for item in obj:
                _try_parse(item)
            return
        if not isinstance(obj, dict):
            return

        # Determine the tool name from various formats
        name = None
        args_raw = None
        if "name" in obj and "arguments" in obj:
            name = obj.get("name")
            args_raw = obj.get("arguments")
        elif "function" in obj and isinstance(obj["function"], dict):
            name = obj["function"].get("name")
            args_raw = obj["function"].get("arguments")

        if not name or args_raw is None:
            # Not a direct tool call — try recursing into dict values
            # This handles {"tool_calls": [...]} wrapper format
            for val in obj.values():
                _try_parse(val)
            return
        if not isinstance(name, str):
            return

        # Normalize arguments to a JSON string
        if isinstance(args_raw, str):
            args_str = args_raw
        else:
            try:
                args_str = json.dumps(args_raw, ensure_ascii=False)
            except Exception:
                return

        tid = "tc_text_" + hashlib.md5(f"{name}:{args_str}".encode()).hexdigest()[:12]
        idx = len(detected)
        detected[idx] = {
            "id": tid,
            "type": "function",
            "function": {
                "name": name,
                "arguments": args_str,
            },
        }

    # --- Strategy 1: try parsing the whole text as JSON ---
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        _try_parse(parsed)
        if detected:
            return detected, None
    except (json.JSONDecodeError, ValueError):
        pass

    # --- Strategy 2: extract JSON from markdown fences ---
    fence_pattern = re.compile(r"```(?:json)?\s*\n?(\{.*?\}|\[.*?\])\s*\n?```", re.DOTALL)
    for m in fence_pattern.finditer(text):
        try:
            parsed = json.loads(m.group(1))
            _try_parse(parsed)
        except (json.JSONDecodeError, ValueError):
            continue
        remove_ranges.append((m.start(), m.end() - 1))

    # --- Strategy 3: iteratively find all bare JSON objects/arrays ---
    # Also captures common prefixes like "Tool Calls:" or "tool_call:"
    prefix_pattern = re.compile(
        r'(?:^|\n)\s*'
        r'(?:Tool\s+Calls?\s*[:=]|tool_calls?\s*[:=]|json\s*[:=]|output\s*[:=])\s*',
        re.IGNORECASE,
    )
    scan_pos = 0
    while scan_pos < len(text):
        span = _find_json_span(text, scan_pos)
        if span is None:
            break
        obj_start, obj_end = span
        candidate = text[obj_start:obj_end + 1]
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            # Not valid JSON, skip past this opening bracket
            scan_pos = obj_end + 1
            continue

        # Check if there's a prefix label before this JSON (on the same or prev line)
        prefix_start = obj_start
        line_start = text.rfind('\n', 0, obj_start)
        line_start = 0 if line_start < 0 else line_start + 1
        preceding = text[line_start:obj_start]
        pm = prefix_pattern.match(preceding)
        if pm:
            prefix_start = line_start + pm.start()

        _try_parse(parsed)
        remove_ranges.append((prefix_start, obj_end))
        scan_pos = obj_end + 1

    if not detected:
        return {}, content

    return detected, _remove_ranges(text, remove_ranges)


def _remove_ranges(text: str, ranges: list[tuple[int, int]]) -> str | None:
    """Remove character ranges from text and return the remainder, or None if empty.

    Each range is (start, end_inclusive) — both start and end are removed.
    """
    if not ranges:
        return text
    ranges = sorted(ranges, key=lambda r: r[0])
    parts: list[str] = []
    prev_end = 0
    for start, end in ranges:
        if start > prev_end:
            parts.append(text[prev_end:start])
        prev_end = end + 1  # inclusive end → exclusive
    if prev_end < len(text):
        parts.append(text[prev_end:])
    result = "".join(parts).strip()
    return result if result else None


@dataclass
class LLMResult:
    """Result from an LLM call (streaming or non-streaming)."""
    content: str | None
    tool_calls_acc: dict[int, dict]
    token_count: int
    finish_reason: str | None
    usage: dict = field(default_factory=dict)


async def _make_cancelled_result(session: Session):
    """Notify the frontend and return a cancelled LLMResult."""
    logger.info("LLM call cancelled by user")
    await session.send_event(Event(
        event_type="tool_log",
        data={"tool": "system", "log": "⏹️ Stopped"},
    ))
    return LLMResult(
        content=None, tool_calls_acc={}, token_count=0,
        finish_reason="cancelled", usage={},
    )


def _maybe_enable_ollama_think(llm_params: dict) -> dict:
    """Enable Ollama's `think` parameter for reasoning-capable models.

    Ollama supports `think=True` for models with reasoning capabilities
    (e.g. qwen3, deepseek-v4).  litellm forwards this as a top-level
    param in the Ollama /api/chat request.  We only set it when the
    model is ollama/ and the user hasn't already configured `think` or
    `reasoning_effort`.
    """
    model = llm_params.get("model", "")
    if not model.startswith("ollama/"):
        return llm_params
    if "think" in llm_params or "reasoning_effort" in llm_params:
        return llm_params
    llm_params = {**llm_params, "think": True}
    return llm_params

async def _call_llm_streaming(session: Session, messages, tools, llm_params) -> LLMResult:
    """Call the LLM with streaming, emitting assistant_chunk events."""

    llm_params = _maybe_enable_ollama_think(llm_params)

    # ── Ollama direct streaming ────────────────────────────────────────
    # litellm's ollama/ provider corrupts streaming tool calls by placing
    # ``index`` inside ``function.index`` instead of at the top level.
    # We hit Ollama's /api/chat directly so tool calls work correctly.
    if (llm_params.get("model") or "").startswith("ollama/"):
        return await _call_llm_ollama_direct(session, messages, tools, llm_params)

    if is_codex_responses_params(llm_params):
        t_start = time.monotonic()
        result = await codex_responses_completion(
            messages=messages,
            tools=tools,
            params=llm_params,
            stream=True,
            on_delta=lambda delta: session.send_event(
                Event(event_type="assistant_chunk", data={"content": delta})
            ),
            timeout=600,
        )
        usage = await telemetry.record_llm_call(
            session,
            model=llm_params.get("model", session.config.model_name),
            response=None,
            latency_ms=int((time.monotonic() - t_start) * 1000),
            finish_reason=result.finish_reason,
        )
        if not usage:
            usage = result.usage
        return LLMResult(
            content=result.content,
            tool_calls_acc=result.tool_calls_acc,
            token_count=result.token_count,
            finish_reason=result.finish_reason,
            usage=usage,
        )

    response = None
    _healed_effort = False  # one-shot safety net per call
    messages, tools = with_prompt_caching(messages, tools, llm_params.get("model"))
    t_start = time.monotonic()
    llm_attempt = 0
    while True:
        try:
            response = await acompletion(
                messages=messages,
                tools=tools,
                tool_choice="auto",
                stream=True,
                stream_options={"include_usage": True},
                timeout=600,
                **llm_params,
            )
            break
        except ContextWindowExceededError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if _is_context_overflow_error(e):
                raise ContextWindowExceededError(str(e)) from e
            if not _healed_effort and _is_effort_config_error(e):
                _healed_effort = True
                llm_params = await _heal_effort_and_rebuild_params(session, e, llm_params)
                await session.send_event(Event(
                    event_type="tool_log",
                    data={"tool": "system", "log": "Reasoning effort not supported for this model — adjusting and retrying."},
                ))
                continue
            if not _is_transient_error(e):
                raise
            # Allow user interruption during retries
            if session.is_cancelled:
                logger.info("LLM call cancelled by user during retry")
                return await _make_cancelled_result(session)
            # Persistent retry with exponential backoff
            delay = _persistent_retry_delay(e, llm_attempt)
            mins = delay // 60
            secs = delay % 60
            wait_msg = f"{mins}m{secs}s" if mins else f"{secs}s"
            logger.warning(
                "Transient LLM error (attempt %d): %s — retrying in %s",
                llm_attempt + 1, e, wait_msg,
            )
            state = "overloaded" if _is_cloud_overloaded(e) else "connection_error"
            await session.send_event(Event(
                event_type="tool_log",
                data={"tool": "system", "log": f"{'☁️ Cloud overloaded' if state == 'overloaded' else '🔌 Connection error'} — retrying in {wait_msg} (attempt {llm_attempt + 1})…"},
            ))
            llm_attempt += 1
            await asyncio.sleep(delay)
            continue

    full_content = ""
    tool_calls_acc: dict[int, dict] = {}
    token_count = 0
    finish_reason = None
    final_usage_chunk = None

    async for chunk in response:
        if session.is_cancelled:
            return await _make_cancelled_result(session)

        choice = chunk.choices[0] if chunk.choices else None
        if not choice:
            if hasattr(chunk, "usage") and chunk.usage:
                token_count = chunk.usage.total_tokens
                final_usage_chunk = chunk
            continue

        delta = choice.delta
        if choice.finish_reason:
            finish_reason = choice.finish_reason

        if delta.content:
            full_content += delta.content
            await session.send_event(
                Event(event_type="assistant_chunk", data={"content": delta.content})
            )

        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in tool_calls_acc:
                    tool_calls_acc[idx] = {
                        "id": "", "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                if tc_delta.id:
                    tool_calls_acc[idx]["id"] = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        tool_calls_acc[idx]["function"]["name"] += tc_delta.function.name
                    if tc_delta.function.arguments:
                        tool_calls_acc[idx]["function"]["arguments"] += tc_delta.function.arguments

        if hasattr(chunk, "usage") and chunk.usage:
            token_count = chunk.usage.total_tokens
            final_usage_chunk = chunk

    usage = await telemetry.record_llm_call(
        session,
        model=llm_params.get("model", session.config.model_name),
        response=final_usage_chunk,
        latency_ms=int((time.monotonic() - t_start) * 1000),
        finish_reason=finish_reason,
    )

    # If the model returned no content and no tool calls, treat it as a
    # transient error so the persistent retry logic kicks in.  This happens
    # when the Ollama cloud is overloaded but still responds with empty chunks.
    if not full_content and not tool_calls_acc:
        raise RuntimeError("Model returned empty response with no tool calls")

    return LLMResult(
        content=full_content or None,
        tool_calls_acc=tool_calls_acc,
        token_count=token_count,
        finish_reason=finish_reason,
        usage=usage,
    )


def _messages_to_dict(messages: list) -> list[dict]:
    """Convert litellm Message objects to Ollama-native dicts.

    Ollama's API uses a different tool_call format than OpenAI:
      - ``function.arguments`` is a dict (not a JSON string)
      - No ``id`` or ``type`` fields on tool calls
      - No ``function_call`` or ``provider_specific_fields``

    We mirror exactly what litellm's Ollama transformation does in
    ``litellm.llms.ollama.chat.transformation``.
    """
    result: list[dict] = []
    for m in messages:
        d = m.model_dump(exclude_none=True) if hasattr(m, 'model_dump') else dict(m)

        # Convert tool_calls from OpenAI format → Ollama native format
        tool_calls = d.get("tool_calls")
        if tool_calls and isinstance(tool_calls, list):
            ollama_tcs = []
            for tc in tool_calls:
                fn = tc.get("function", {})
                fn_name = fn.get("name", "")
                fn_args = fn.get("arguments", {})
                # Parse JSON string → dict if needed
                if isinstance(fn_args, str):
                    try:
                        fn_args = json.loads(fn_args)
                    except (json.JSONDecodeError, TypeError):
                        fn_args = {}
                ollama_tcs.append({
                    "function": {
                        "name": fn_name,
                        "arguments": fn_args,
                    }
                })
            d["tool_calls"] = ollama_tcs

        # Strip litellm-only fields that confuse Ollama
        d.pop("function_call", None)
        d.pop("provider_specific_fields", None)

        result.append(d)
    return result


async def _call_llm_ollama_direct(session: Session, messages, tools, llm_params) -> LLMResult:
    """Stream chat from Ollama using a direct HTTP connection.

    litellm corrupts the ``index`` field on Ollama tool calls during
    streaming, so we bypass it entirely and call /api/chat ourselves.
    """
    msg_dicts = _messages_to_dict(messages)
    logger.info("[OLLAMA_DIRECT] sending %d messages, model=%s", len(msg_dicts), llm_params.get("model", "?"))
    for i, m in enumerate(msg_dicts):
        logger.info("[OLLAMA_DIRECT] msg[%d] role=%s, content_len=%d, tc=%d", i, m.get("role"), len(m.get("content") or ""), len(m.get("tool_calls") or []))
    t_start = time.monotonic()
    llm_attempt = 0
    while True:
        full_content = ""
        tool_calls_acc: dict[int, dict] = {}
        token_count = 0
        finish_reason: str | None = None
        final_usage_chunk = None
        try:
            stream = ollama_chat_streaming(
                model=llm_params["model"],
                messages=msg_dicts,
                tools=tools,
                llm_params=llm_params,
                timeout=600,
            )
            async for chunk in stream:
                if session.is_cancelled:
                    return await _make_cancelled_result(session)

                choice = chunk.choices[0] if chunk.choices else None
                if not choice:
                    if chunk.usage:
                        token_count = chunk.usage.total_tokens
                        final_usage_chunk = chunk
                    continue

                delta = choice.delta
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                if delta.content:
                    full_content += delta.content
                    await session.send_event(
                        Event(event_type="assistant_chunk", data={"content": delta.content})
                    )

                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": "", "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        if tc_delta.id:
                            tool_calls_acc[idx]["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tool_calls_acc[idx]["function"]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                tool_calls_acc[idx]["function"]["arguments"] = tc_delta.function.arguments

                if chunk.usage:
                    token_count = chunk.usage.total_tokens
                    final_usage_chunk = chunk

            # ── Empty response detection ─────────────────────────────────
            if not full_content and not tool_calls_acc:
                raise RuntimeError("Model returned empty response with no tool calls")

            break  # ── Success: we got a valid response ──

        except ContextWindowExceededError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if not _is_transient_error(e):
                raise
            if session.is_cancelled:
                logger.info("LLM call cancelled by user during retry")
                return await _make_cancelled_result(session)
            delay = _persistent_retry_delay(e, llm_attempt)
            mins = delay // 60
            secs = delay % 60
            wait_msg = f"{mins}m{secs}s" if mins else f"{secs}s"
            logger.warning(
                "Transient Ollama error (attempt %d): %s — retrying in %s",
                llm_attempt + 1, e, wait_msg,
            )
            state = "overloaded" if _is_cloud_overloaded(e) else "connection_error"
            await session.send_event(Event(
                event_type="tool_log",
                data={"tool": "system", "log": f"{'☁️ Cloud overloaded' if state == 'overloaded' else '🔌 Connection error'} — retrying in {wait_msg} (attempt {llm_attempt + 1})…"},
            ))
            llm_attempt += 1
            await asyncio.sleep(delay)
            continue

    usage = await telemetry.record_llm_call(
        session,
        model=llm_params.get("model", session.config.model_name),
        response=final_usage_chunk,
        latency_ms=int((time.monotonic() - t_start) * 1000),
        finish_reason=finish_reason,
    )

    return LLMResult(
        content=full_content or None,
        tool_calls_acc=tool_calls_acc,
        token_count=token_count,
        finish_reason=finish_reason,
        usage=usage,
    )


async def _call_llm_ollama_non_streaming(session: Session, messages, tools, llm_params) -> LLMResult:
    """Non-streaming chat from Ollama using direct HTTP connection."""
    msg_dicts = _messages_to_dict(messages)
    t_start = time.monotonic()
    llm_attempt = 0
    while True:
        try:
            result = await ollama_chat_non_streaming(
                model=llm_params["model"],
                messages=msg_dicts,
                tools=tools,
                llm_params=llm_params,
                timeout=600,
            )
            break
        except ContextWindowExceededError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if not _is_transient_error(e):
                raise
            if session.is_cancelled:
                logger.info("LLM call cancelled by user during retry")
                return await _make_cancelled_result(session)
            delay = _persistent_retry_delay(e, llm_attempt)
            mins = delay // 60
            secs = delay % 60
            wait_msg = f"{mins}m{secs}s" if mins else f"{secs}s"
            logger.warning(
                "Transient Ollama error (attempt %d): %s — retrying in %s",
                llm_attempt + 1, e, wait_msg,
            )
            state = "overloaded" if _is_cloud_overloaded(e) else "connection_error"
            await session.send_event(Event(
                event_type="tool_log",
                data={"tool": "system", "log": f"{'☁️ Cloud overloaded' if state == 'overloaded' else '🔌 Connection error'} — retrying in {wait_msg} (attempt {llm_attempt + 1})…"},
            ))
            llm_attempt += 1
            await asyncio.sleep(delay)
            continue

    content = result._content
    tool_calls = result._tool_calls
    finish_reason = result.finish_reason

    if content:
        await session.send_event(
            Event(event_type="assistant_message", data={"content": content})
        )

    # Build tool_calls_acc in same format as streaming
    tool_calls_acc: dict[int, dict] = {}
    if tool_calls:
        for idx, tc in enumerate(tool_calls):
            # Use index from tool call if present
            tc_idx = tc.get("index", idx)
            tool_calls_acc[tc_idx] = {
                "id": tc.get("id", ""),
                "type": "function",
                "function": {
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": tc.get("function", {}).get("arguments", ""),
                },
            }

    # ── Empty response detection ──────────────────────────────────────
    if not content and not tool_calls_acc:
        raise RuntimeError("Model returned empty response with no tool calls")

    usage = await telemetry.record_llm_call(
        session,
        model=llm_params.get("model", session.config.model_name),
        response=None,
        latency_ms=int((time.monotonic() - t_start) * 1000),
        finish_reason=finish_reason,
    )

    return LLMResult(
        content=content,
        tool_calls_acc=tool_calls_acc,
        token_count=result.usage.total_tokens if result.usage else 0,
        finish_reason=finish_reason,
        usage=usage or result.usage,
    )


async def _call_llm_non_streaming(session: Session, messages, tools, llm_params) -> LLMResult:
    """Call the LLM without streaming, emit assistant_message at the end."""
    if is_codex_responses_params(llm_params):
        t_start = time.monotonic()
        result = await codex_responses_completion(
            messages=messages,
            tools=tools,
            params=llm_params,
            stream=False,
            timeout=600,
        )
        if result.content:
            await session.send_event(
                Event(event_type="assistant_message", data={"content": result.content})
            )
        usage = await telemetry.record_llm_call(
            session,
            model=llm_params.get("model", session.config.model_name),
            response=None,
            latency_ms=int((time.monotonic() - t_start) * 1000),
            finish_reason=result.finish_reason,
        )
        if not usage:
            usage = result.usage
        return LLMResult(
            content=result.content,
            tool_calls_acc=result.tool_calls_acc,
            token_count=result.token_count,
            finish_reason=result.finish_reason,
            usage=usage,
        )

    # ── Ollama direct non-streaming ────────────────────────────────────
    if (llm_params.get("model") or "").startswith("ollama/"):
        return await _call_llm_ollama_non_streaming(session, messages, tools, llm_params)

    response = None
    _healed_effort = False
    messages, tools = with_prompt_caching(messages, tools, llm_params.get("model"))
    t_start = time.monotonic()
    llm_attempt = 0
    while True:
        try:
            response = await acompletion(
                messages=messages,
                tools=tools,
                tool_choice="auto",
                stream=False,
                timeout=600,
                **llm_params,
            )
            break
        except ContextWindowExceededError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if _is_context_overflow_error(e):
                raise ContextWindowExceededError(str(e)) from e
            if not _healed_effort and _is_effort_config_error(e):
                _healed_effort = True
                llm_params = await _heal_effort_and_rebuild_params(session, e, llm_params)
                await session.send_event(Event(
                    event_type="tool_log",
                    data={"tool": "system", "log": "Reasoning effort not supported for this model — adjusting and retrying."},
                ))
                continue
            if not _is_transient_error(e):
                raise
            # Allow user interruption during retries
            if session.is_cancelled:
                logger.info("LLM call cancelled by user during retry")
                return await _make_cancelled_result(session)
            # Persistent retry with exponential backoff
            delay = _persistent_retry_delay(e, llm_attempt)
            mins = delay // 60
            secs = delay % 60
            wait_msg = f"{mins}m{secs}s" if mins else f"{secs}s"
            logger.warning(
                "Transient LLM error (attempt %d): %s — retrying in %s",
                llm_attempt + 1, e, wait_msg,
            )
            state = "overloaded" if _is_cloud_overloaded(e) else "connection_error"
            await session.send_event(Event(
                event_type="tool_log",
                data={"tool": "system", "log": f"{'☁️ Cloud overloaded' if state == 'overloaded' else '🔌 Connection error'} — retrying in {wait_msg} (attempt {llm_attempt + 1})…"},
            ))
            llm_attempt += 1
            await asyncio.sleep(delay)
            continue

    choice = response.choices[0]
    message = choice.message
    content = message.content or None
    finish_reason = choice.finish_reason
    token_count = response.usage.total_tokens if response.usage else 0

    # Build tool_calls_acc in the same format as streaming
    tool_calls_acc: dict[int, dict] = {}
    if message.tool_calls:
        for idx, tc in enumerate(message.tool_calls):
            tool_calls_acc[idx] = {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }

    # Emit the full message as a single event
    if content:
        await session.send_event(
            Event(event_type="assistant_message", data={"content": content})
        )

    usage = await telemetry.record_llm_call(
        session,
        model=llm_params.get("model", session.config.model_name),
        response=response,
        latency_ms=int((time.monotonic() - t_start) * 1000),
        finish_reason=finish_reason,
    )

    # Treat empty responses as transient errors for retry
    if not content and not tool_calls_acc:
        raise RuntimeError("Model returned empty response with no tool calls")

    return LLMResult(
        content=content,
        tool_calls_acc=tool_calls_acc,
        token_count=token_count,
        finish_reason=finish_reason,
        usage=usage,
    )


class Handlers:
    """Handler functions for each operation type"""

    @staticmethod
    async def _abandon_pending_approval(session: Session) -> None:
        """Cancel pending approval tools when the user continues the conversation.

        Injects rejection tool-result messages into the LLM context (so the
        history stays valid) and notifies the frontend that those tools were
        abandoned.
        """
        tool_calls = session.pending_approval.get("tool_calls", [])
        for tc in tool_calls:
            tool_name = tc.function.name
            abandon_msg = (
                "Task abandoned — user continued the conversation without approving."
            )

            # Keep LLM context valid: every tool_call needs a tool result
            tool_msg = Message(
                role="tool",
                content=abandon_msg,
                tool_call_id=tc.id,
                name=tool_name,
            )
            session.context_manager.add_message(tool_msg)

            await session.send_event(
                Event(
                    event_type="tool_state_change",
                    data={
                        "tool_call_id": tc.id,
                        "tool": tool_name,
                        "state": "abandoned",
                    },
                )
            )

        session.pending_approval = None
        logger.info("Abandoned %d pending approval tool(s)", len(tool_calls))

    @staticmethod
    async def run_agent(
        session: Session, text: str,
    ) -> str | None:
        """
        Handle user input (like user_input_or_turn in codex.rs:1291)
        Returns the final assistant response content, if any.
        """
        # Clear any stale cancellation flag from a previous run
        session.reset_cancel()

        # If there's a pending approval and the user sent a new message,
        # abandon the pending tools so the LLM context stays valid.
        if text and session.pending_approval:
            await Handlers._abandon_pending_approval(session)

        # Add user message to history only if there's actual content
        if text:
            user_msg = Message(role="user", content=text)
            session.context_manager.add_message(user_msg)

        # Send event that we're processing
        await session.send_event(
            Event(event_type="processing", data={"message": "Processing user input"})
        )

        # Agentic loop - continue until model doesn't call tools or max iterations is reached
        iteration = 0
        final_response = None
        errored = False
        max_iterations = session.config.max_iterations
        empty_response_retries = 0
        logger.info(
            "[AGENT] run_agent START: text=%r, model=%s, stream=%s, total_items=%d",
            (text or "")[:80], session.config.model_name,
            session.stream, len(session.context_manager.items),
        )

        while max_iterations == -1 or iteration < max_iterations:
            # ── Cancellation check: before LLM call ──
            if session.is_cancelled:
                break

            # Compact before calling the LLM if context is near the limit
            logger.info(
                "[AGENT] run_agent iter=%d: before compact, usage=%d, max=%d, threshold=%d, items=%d",
                iteration, session.context_manager.running_context_usage,
                session.context_manager.model_max_tokens,
                session.context_manager.compaction_threshold,
                len(session.context_manager.items),
            )
            await _compact_and_notify(session)
            logger.info(
                "[AGENT] run_agent iter=%d: after compact, usage=%d, items=%d",
                iteration, session.context_manager.running_context_usage,
                len(session.context_manager.items),
            )

            # Doom-loop detection: break out of repeated tool call patterns
            doom_prompt = check_for_doom_loop(session.context_manager.items)
            if doom_prompt:
                session.context_manager.add_message(
                    Message(role="user", content=doom_prompt)
                )
                await session.send_event(
                    Event(
                        event_type="tool_log",
                        data={
                            "tool": "system",
                            "log": "Doom loop detected — injecting corrective prompt",
                        },
                    )
                )

            malformed_tool = _detect_repeated_malformed(session.context_manager.items)
            if malformed_tool:
                recovery_prompt = (
                    "[SYSTEM: Repeated malformed tool arguments detected for "
                    f"'{malformed_tool}'. Stop retrying the same tool call shape. "
                    "Use a different strategy that produces smaller, valid JSON. "
                    "For large file writes, prefer bash with a heredoc or split the "
                    "edit into multiple smaller tool calls.]"
                )
                session.context_manager.add_message(
                    Message(role="user", content=recovery_prompt)
                )
                await session.send_event(
                    Event(
                        event_type="tool_log",
                        data={
                            "tool": "system",
                            "log": (
                                "Repeated malformed tool arguments detected — "
                                f"forcing a different strategy for {malformed_tool}"
                            ),
                        },
                    )
                )

            messages = session.context_manager.get_messages()
            tools = session.tool_router.get_tool_specs_for_llm()
            try:
                # ── Call the LLM (streaming or non-streaming) ──
                # Pull the per-model probed effort from the session cache when
                # available; fall back to the raw preference for models we
                # haven't probed yet (e.g. research sub-model).
                llm_params = _resolve_llm_params(
                    session.config.model_name,
                    session.hf_token,
                    reasoning_effort=session.effective_effort_for(session.config.model_name),
                    provider_keys=getattr(session, "provider_keys", None),
                )
                if session.stream:
                    session._current_llm_task = asyncio.create_task(
                        _call_llm_streaming(session, messages, tools, llm_params)
                    )
                else:
                    session._current_llm_task = asyncio.create_task(
                        _call_llm_non_streaming(session, messages, tools, llm_params)
                    )
                try:
                    llm_result = await session._current_llm_task
                except asyncio.CancelledError:
                    logger.info("LLM call cancelled by user")
                    llm_result = LLMResult(
                        content=None,
                        tool_calls_acc={},
                        token_count=0,
                        finish_reason="cancelled",
                        usage={},
                    )
                except Exception as llm_exc:
                    logger.error(
                        "[AGENT] LLM CALL RAISED: %s: %s",
                        type(llm_exc).__name__, str(llm_exc)[:200],
                        exc_info=True,
                    )
                    raise
                finally:
                    session._current_llm_task = None

                content = llm_result.content
                tool_calls_acc = llm_result.tool_calls_acc
                token_count = llm_result.token_count
                finish_reason = llm_result.finish_reason
                logger.info(
                    "[AGENT] LLM result: content_len=%d, tool_calls=%d, finish=%s, tokens=%d",
                    len(content or ""), len(tool_calls_acc or {}),
                    finish_reason, token_count or 0,
                )

                # Detect silent overflow: model returned empty content with no tool calls
                # This can happen when context is too large and the model gives up silently
                if not content and not tool_calls_acc and token_count == 0:
                    cm = session.context_manager
                    if cm.running_context_usage > cm.compaction_threshold * 0.8:
                        logger.warning(
                            "Silent overflow suspected at iteration %d — empty response "
                            "with high context usage (%d/%d). Forcing compaction.",
                            iteration, cm.running_context_usage, cm.model_max_tokens,
                        )
                        cm.force_compaction()
                        await _compact_and_notify(session)
                        continue

                # If output was truncated, all tool call args are garbage.
                # Inject a system hint so the LLM retries with smaller content.
                if finish_reason == "length" and tool_calls_acc:
                    dropped_names = [
                        tc["function"]["name"]
                        for tc in tool_calls_acc.values()
                        if tc["function"]["name"]
                    ]
                    logger.warning(
                        "Output truncated (finish_reason=length) — dropping tool calls: %s",
                        dropped_names,
                    )
                    tool_calls_acc.clear()

                    # Tell the agent what happened so it can retry differently
                    truncation_hint = (
                        "Your previous response was truncated because the output hit the "
                        "token limit. The following tool calls were lost: "
                        f"{dropped_names}. "
                        "IMPORTANT: Do NOT retry with the same large content. Instead:\n"
                        "  • For 'write': use bash with cat<<'HEREDOC' to write the file, "
                        "or split into several smaller edit calls.\n"
                        "  • For other tools: reduce the size of your arguments or use bash."
                    )
                    if content:
                        assistant_msg = Message(role="assistant", content=content)
                        session.context_manager.add_message(assistant_msg, token_count)
                    session.context_manager.add_message(
                        Message(role="user", content=f"[SYSTEM: {truncation_hint}]")
                    )
                    if session.stream:
                        await session.send_event(
                            Event(event_type="assistant_stream_end", data={})
                        )
                    await session.send_event(
                        Event(
                            event_type="tool_log",
                            data={"tool": "system", "log": f"Output truncated — retrying with smaller content ({dropped_names})"},
                        )
                    )
                    iteration += 1
                    continue  # retry this iteration

                # If the model didn't produce structured tool calls, check if
                # the text content contains a tool call as raw JSON (common with
                # smaller Ollama models that don't support the tools parameter).
                if not tool_calls_acc and content:
                    extracted, cleaned = _extract_tool_calls_from_content(content)
                    if extracted:
                        tool_calls_acc = extracted
                        content = cleaned  # strip JSON from message display
                        logger.info(
                            "Extracted %d tool call(s) from plain-text content: %s",
                            len(tool_calls_acc),
                            [tc["function"]["name"] for tc in tool_calls_acc.values()],
                        )
                        await session.send_event(Event(
                            event_type="tool_log",
                            data={"tool": "system", "log": "Model emitted tool call as text — parsed automatically."},
                        ))

                # Build tool_calls list from accumulated deltas
                tool_calls: list[ToolCall] = []
                for idx in sorted(tool_calls_acc.keys()):
                    tc_data = tool_calls_acc[idx]
                    tool_calls.append(
                        ToolCall(
                            id=tc_data["id"],
                            type="function",
                            function={
                                "name": tc_data["function"]["name"],
                                "arguments": tc_data["function"]["arguments"],
                            },
                        )
                    )

                # Signal end of streaming to the frontend
                if session.stream:
                    await session.send_event(
                        Event(event_type="assistant_stream_end", data={})
                    )

                # If the model returns neither text nor tool calls, do not end
                # the turn silently. Codex Responses can occasionally emit an
                # empty stop after tool results; ask once for an explicit final
                # response so the UI never appears to "close" without answer.
                if not tool_calls and not content:
                    empty_response_retries += 1
                    if empty_response_retries > 2:
                        fallback = (
                            "I did not receive a usable model response after tool execution. "
                            "The turn stopped to avoid looping. Please send the next instruction, "
                            "or retry with a shorter prompt."
                        )
                        assistant_msg = Message(role="assistant", content=fallback)
                        session.context_manager.add_message(assistant_msg, token_count)
                        await session.send_event(
                            Event(event_type="assistant_message", data={"content": fallback})
                        )
                        final_response = fallback
                        break
                    logger.warning(
                        "Empty LLM response with no tool calls at iteration %d/%d; nudging model to continue",
                        iteration,
                        max_iterations,
                    )
                    session.context_manager.add_message(Message(
                        role="user",
                        content=(
                            "[SYSTEM: Your previous response was empty after tool execution. "
                            "Continue now. Summarize what happened, handle any tool errors, "
                            "and either proceed with the task using available tools or explain the blocker.]"
                        ),
                    ))
                    iteration += 1
                    continue

                # If no tool calls, add assistant message and we're done
                if not tool_calls:
                    logger.debug(
                        "Agent loop ending: no tool calls. "
                        "finish_reason=%s, token_count=%d, "
                        "usage=%d, model_max_tokens=%d, "
                        "iteration=%d/%d, "
                        "response_text=%s",
                        finish_reason,
                        token_count,
                        session.context_manager.running_context_usage,
                        session.context_manager.model_max_tokens,
                        iteration,
                        max_iterations,
                        (content or "")[:500],
                    )
                    assistant_msg = Message(role="assistant", content=content)
                    session.context_manager.add_message(assistant_msg, token_count)
                    final_response = content
                    break

                # Validate tool call args (one json.loads per call, once)
                # and split into good vs bad
                good_tools: list[tuple[ToolCall, str, dict]] = []
                bad_tools: list[ToolCall] = []
                for tc in tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                        good_tools.append((tc, tc.function.name, args))
                    except (json.JSONDecodeError, TypeError, ValueError):
                        logger.warning(
                            "Malformed arguments for tool_call %s (%s) — skipping",
                            tc.id, tc.function.name,
                        )
                        tc.function.arguments = "{}"
                        bad_tools.append(tc)

                # Add assistant message with all tool calls to context
                assistant_msg = Message(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls,
                )
                session.context_manager.add_message(assistant_msg, token_count)

                # Add error results for bad tool calls so the LLM
                # knows what happened and can retry differently
                for tc in bad_tools:
                    error_msg = (
                        f"ERROR: Tool call to '{tc.function.name}' had malformed JSON "
                        f"arguments and was NOT executed. Retry with smaller content — "
                        f"for 'write', split into multiple smaller writes using 'edit'."
                    )
                    session.context_manager.add_message(Message(
                        role="tool",
                        content=error_msg,
                        tool_call_id=tc.id,
                        name=tc.function.name,
                    ))
                    await session.send_event(Event(
                        event_type="tool_call",
                        data={"tool": tc.function.name, "arguments": {}, "tool_call_id": tc.id},
                    ))
                    await session.send_event(Event(
                        event_type="tool_output",
                        data={"tool": tc.function.name, "tool_call_id": tc.id, "output": error_msg, "success": False},
                    ))

                # ── Cancellation check: before tool execution ──
                if session.is_cancelled:
                    break

                # Separate good tools into approval-required vs auto-execute
                approval_required_tools: list[tuple[ToolCall, str, dict]] = []
                non_approval_tools: list[tuple[ToolCall, str, dict]] = []
                for tc, tool_name, tool_args in good_tools:
                    if _needs_approval(tool_name, tool_args, session.config):
                        approval_required_tools.append((tc, tool_name, tool_args))
                    else:
                        non_approval_tools.append((tc, tool_name, tool_args))

                # Execute non-approval tools (in parallel when possible)
                if non_approval_tools:
                    # 1. Validate args upfront
                    parsed_tools: list[
                        tuple[ToolCall, str, dict, bool, str]
                    ] = []
                    for tc, tool_name, tool_args in non_approval_tools:
                        args_valid, error_msg = _validate_tool_args(tool_args)
                        parsed_tools.append(
                            (tc, tool_name, tool_args, args_valid, error_msg)
                        )

                    # 2. Send all tool_call events upfront (so frontend shows them all)
                    for tc, tool_name, tool_args, args_valid, _ in parsed_tools:
                        if args_valid:
                            await session.send_event(
                                Event(
                                    event_type="tool_call",
                                    data={
                                        "tool": tool_name,
                                        "arguments": tool_args,
                                        "tool_call_id": tc.id,
                                    },
                                )
                            )

                    # 3. Execute all valid tools in parallel, cancellable
                    async def _exec_tool(
                        tc: ToolCall,
                        name: str,
                        args: dict,
                        valid: bool,
                        err: str,
                    ) -> tuple[ToolCall, str, dict, str, bool]:
                        if not valid:
                            return (tc, name, args, err, False)
                        out, ok = await session.tool_router.call_tool(
                            name, args, session=session, tool_call_id=tc.id
                        )
                        return (tc, name, args, out, ok)

                    gather_task = asyncio.ensure_future(asyncio.gather(
                        *[
                            _exec_tool(tc, name, args, valid, err)
                            for tc, name, args, valid, err in parsed_tools
                        ]
                    ))
                    cancel_task = asyncio.ensure_future(session._cancelled.wait())

                    done, _ = await asyncio.wait(
                        [gather_task, cancel_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    if cancel_task in done:
                        gather_task.cancel()
                        try:
                            await gather_task
                        except asyncio.CancelledError:
                            pass
                        # Notify frontend that in-flight tools were cancelled
                        for tc, name, _args, valid, _ in parsed_tools:
                            if valid:
                                await session.send_event(Event(
                                    event_type="tool_state_change",
                                    data={"tool_call_id": tc.id, "tool": name, "state": "cancelled"},
                                ))
                        await _cleanup_on_cancel(session)
                        break

                    cancel_task.cancel()
                    results = gather_task.result()

                    # 4. Record results and send outputs (order preserved)
                    for tc, tool_name, tool_args, output, success in results:
                        tool_msg = Message(
                            role="tool",
                            content=output,
                            tool_call_id=tc.id,
                            name=tool_name,
                        )
                        session.context_manager.add_message(tool_msg)

                        await session.send_event(
                            Event(
                                event_type="tool_output",
                                data={
                                    "tool": tool_name,
                                    "tool_call_id": tc.id,
                                    "output": output,
                                    "success": success,
                                },
                            )
                        )

                # If there are tools requiring approval, ask for batch approval
                if approval_required_tools:
                    # Prepare batch approval data
                    tools_data = []
                    for tc, tool_name, tool_args in approval_required_tools:
                        # Resolve sandbox file paths for hf_jobs scripts so the
                        # frontend can display & edit the actual file content.
                        if tool_name == "hf_jobs" and isinstance(tool_args.get("script"), str):
                            from agent.tools.sandbox_tool import resolve_sandbox_script
                            sandbox = getattr(session, "sandbox", None)
                            resolved, _ = await resolve_sandbox_script(sandbox, tool_args["script"])
                            if resolved:
                                tool_args = {**tool_args, "script": resolved}

                        tools_data.append({
                            "tool": tool_name,
                            "arguments": tool_args,
                            "tool_call_id": tc.id,
                        })

                    await session.send_event(Event(
                        event_type="approval_required",
                        data={"tools": tools_data, "count": len(tools_data)},
                    ))

                    # Store all approval-requiring tools (ToolCall objects for execution)
                    session.pending_approval = {
                        "tool_calls": [tc for tc, _, _ in approval_required_tools],
                    }

                    # Return early - wait for EXEC_APPROVAL operation
                    return None

                iteration += 1

            except ContextWindowExceededError:
                # Force compact and retry this iteration
                cm = session.context_manager
                logger.warning(
                    "ContextWindowExceededError at iteration %d — forcing compaction "
                    "(usage=%d, model_max_tokens=%d, messages=%d)",
                    iteration, cm.running_context_usage, cm.model_max_tokens, len(cm.items),
                )
                cm.force_compaction()
                await _compact_and_notify(session)
                continue

            except Exception as e:
                import traceback
                logger.error(
                    "[AGENT] OUTER EXCEPTION at iter=%d: %s: %s",
                    iteration, type(e).__name__, str(e)[:200],
                )
                error_msg = _friendly_error_message(e)
                error_str = str(e)

                # Check for context overflow patterns in error messages
                if is_context_overflow(error_str):
                    cm = session.context_manager
                    logger.warning(
                        "Context overflow detected at iteration %d (pattern match) — "
                        "forcing compaction (usage=%d, model_max_tokens=%d, messages=%d)",
                        iteration, cm.running_context_usage, cm.model_max_tokens, len(cm.items),
                    )
                    cm.force_compaction()
                    await _compact_and_notify(session)
                    continue

                if error_msg is None:
                    error_msg = error_str + "\n" + traceback.format_exc()

                await session.send_event(
                    Event(
                        event_type="error",
                        data={"error": error_msg},
                    )
                )
                errored = True
                break

        if session.is_cancelled:
            await _cleanup_on_cancel(session)
            await session.send_event(Event(event_type="interrupted"))
        elif not errored:
            await session.send_event(
                Event(
                    event_type="turn_complete",
                    data={"history_size": len(session.context_manager.items)},
                )
            )

        # Increment turn counter and check for auto-save
        session.increment_turn()
        await session.auto_save_if_needed()

        return final_response

    @staticmethod
    async def undo(session: Session) -> None:
        """Remove the last complete turn and notify the frontend."""
        removed = session.context_manager.undo_last_turn()
        if not removed:
            logger.warning("Undo: no user message found to remove")
        await session.send_event(Event(event_type="undo_complete"))

    @staticmethod
    async def exec_approval(session: Session, approvals: list[dict]) -> None:
        """Handle batch job execution approval"""
        if not session.pending_approval:
            await session.send_event(
                Event(
                    event_type="error",
                    data={"error": "No pending approval to process"},
                )
            )
            return

        tool_calls = session.pending_approval.get("tool_calls", [])
        if not tool_calls:
            await session.send_event(
                Event(
                    event_type="error",
                    data={"error": "No pending tool calls found"},
                )
            )
            return

        # Create a map of tool_call_id -> approval decision
        approval_map = {a["tool_call_id"]: a for a in approvals}
        for a in approvals:
            if a.get("edited_script"):
                logger.info(
                    f"Received edited script for tool_call {a['tool_call_id']} ({len(a['edited_script'])} chars)"
                )

        # Separate approved and rejected tool calls
        approved_tasks = []
        rejected_tasks = []

        for tc in tool_calls:
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError) as e:
                # Malformed arguments — treat as failed, notify agent
                logger.warning(f"Malformed tool arguments for {tool_name}: {e}")
                tool_msg = Message(
                    role="tool",
                    content=f"Malformed arguments: {e}",
                    tool_call_id=tc.id,
                    name=tool_name,
                )
                session.context_manager.add_message(tool_msg)
                await session.send_event(
                    Event(
                        event_type="tool_output",
                        data={
                            "tool": tool_name,
                            "tool_call_id": tc.id,
                            "output": f"Malformed arguments: {e}",
                            "success": False,
                        },
                    )
                )
                continue

            approval_decision = approval_map.get(tc.id, {"approved": False})

            if approval_decision.get("approved", False):
                edited_script = approval_decision.get("edited_script")
                was_edited = False
                if edited_script and "script" in tool_args:
                    tool_args["script"] = edited_script
                    was_edited = True
                    logger.info(f"Using user-edited script for {tool_name} ({tc.id})")
                selected_namespace = approval_decision.get("namespace")
                if selected_namespace and tool_name == "hf_jobs":
                    tool_args["namespace"] = selected_namespace
                approved_tasks.append((tc, tool_name, tool_args, was_edited))
            else:
                rejected_tasks.append((tc, tool_name, approval_decision))

        # Clear pending approval immediately so a page refresh during
        # execution won't re-show the approval dialog.
        session.pending_approval = None

        # Notify frontend of approval decisions immediately (before execution)
        for tc, tool_name, tool_args, _was_edited in approved_tasks:
            await session.send_event(
                Event(
                    event_type="tool_state_change",
                    data={
                        "tool_call_id": tc.id,
                        "tool": tool_name,
                        "state": "approved",
                    },
                )
            )
        for tc, tool_name, approval_decision in rejected_tasks:
            await session.send_event(
                Event(
                    event_type="tool_state_change",
                    data={
                        "tool_call_id": tc.id,
                        "tool": tool_name,
                        "state": "rejected",
                    },
                )
            )

        # Execute all approved tools concurrently
        async def execute_tool(tc, tool_name, tool_args, was_edited):
            """Execute a single tool and return its result.

            The TraceLog already exists on the frontend (created by
            approval_required), so we send tool_state_change instead of
            tool_call to avoid creating a duplicate.
            """
            await session.send_event(
                Event(
                    event_type="tool_state_change",
                    data={
                        "tool_call_id": tc.id,
                        "tool": tool_name,
                        "state": "running",
                    },
                )
            )

            output, success = await session.tool_router.call_tool(
                tool_name, tool_args, session=session, tool_call_id=tc.id
            )

            return (tc, tool_name, output, success, was_edited)

        # Execute all approved tools concurrently (cancellable)
        if approved_tasks:
            gather_task = asyncio.ensure_future(asyncio.gather(
                *[
                    execute_tool(tc, tool_name, tool_args, was_edited)
                    for tc, tool_name, tool_args, was_edited in approved_tasks
                ],
                return_exceptions=True,
            ))
            cancel_task = asyncio.ensure_future(session._cancelled.wait())

            done, _ = await asyncio.wait(
                [gather_task, cancel_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            if cancel_task in done:
                gather_task.cancel()
                try:
                    await gather_task
                except asyncio.CancelledError:
                    pass
                # Notify frontend that approved tools were cancelled
                for tc, tool_name, _args, _was_edited in approved_tasks:
                    await session.send_event(Event(
                        event_type="tool_state_change",
                        data={"tool_call_id": tc.id, "tool": tool_name, "state": "cancelled"},
                    ))
                await _cleanup_on_cancel(session)
                await session.send_event(Event(event_type="interrupted"))
                session.increment_turn()
                await session.auto_save_if_needed()
                return

            cancel_task.cancel()
            results = gather_task.result()

            # Process results and add to context
            for result in results:
                if isinstance(result, Exception):
                    # Handle execution error
                    logger.error(f"Tool execution error: {result}")
                    continue

                tc, tool_name, output, success, was_edited = result

                if was_edited:
                    output = f"[Note: The user edited the script before execution. The output below reflects the user-modified version, not your original script.]\n\n{output}"

                # Add tool result to context
                tool_msg = Message(
                    role="tool",
                    content=output,
                    tool_call_id=tc.id,
                    name=tool_name,
                )
                session.context_manager.add_message(tool_msg)

                await session.send_event(
                    Event(
                        event_type="tool_output",
                        data={
                            "tool": tool_name,
                            "tool_call_id": tc.id,
                            "output": output,
                            "success": success,
                        },
                    )
                )

        # Process rejected tools
        for tc, tool_name, approval_decision in rejected_tasks:
            rejection_msg = "Job execution cancelled by user"
            user_feedback = approval_decision.get("feedback")
            if user_feedback:
                # Ensure feedback is a string and sanitize any problematic characters
                feedback_str = str(user_feedback).strip()
                # Remove any control characters that might break JSON parsing
                feedback_str = "".join(
                    char for char in feedback_str if ord(char) >= 32 or char in "\n\t"
                )
                rejection_msg += f". User feedback: {feedback_str}"

            # Ensure rejection_msg is a clean string
            rejection_msg = str(rejection_msg).strip()

            tool_msg = Message(
                role="tool",
                content=rejection_msg,
                tool_call_id=tc.id,
                name=tool_name,
            )
            session.context_manager.add_message(tool_msg)

            await session.send_event(
                Event(
                    event_type="tool_output",
                    data={
                        "tool": tool_name,
                        "tool_call_id": tc.id,
                        "output": rejection_msg,
                        "success": False,
                    },
                )
            )

        # Continue agent loop with empty input to process the tool results
        await Handlers.run_agent(session, "")

    @staticmethod
    async def shutdown(session: Session) -> bool:
        """Handle shutdown (like shutdown in codex.rs:1329)"""
        # Save session trajectory if enabled (fire-and-forget, returns immediately)
        if session.config.save_sessions:
            logger.info("Saving session...")
            repo_id = session.config.session_dataset_repo
            _ = session.save_and_upload_detached(repo_id)

        session.is_running = False
        await session.send_event(Event(event_type="shutdown"))
        return True


async def process_submission(session: Session, submission) -> bool:
    """
    Process a single submission and return whether to continue running.

    Returns:
        bool: True to continue, False to shutdown
    """
    op = submission.operation
    logger.debug("Received operation: %s", op.op_type.value)

    if op.op_type == OpType.USER_INPUT:
        text = op.data.get("text", "") if op.data else ""
        await Handlers.run_agent(session, text)
        return True

    if op.op_type == OpType.COMPACT:
        await _compact_and_notify(session)
        return True

    if op.op_type == OpType.UNDO:
        await Handlers.undo(session)
        return True

    if op.op_type == OpType.EXEC_APPROVAL:
        approvals = op.data.get("approvals", []) if op.data else []
        await Handlers.exec_approval(session, approvals)
        return True

    if op.op_type == OpType.SHUTDOWN:
        return not await Handlers.shutdown(session)

    logger.warning(f"Unknown operation: {op.op_type}")
    return True


async def submission_loop(
    submission_queue: asyncio.Queue,
    event_queue: asyncio.Queue,
    config: Config | None = None,
    tool_router: ToolRouter | None = None,
    session_holder: list | None = None,
    hf_token: str | None = None,
    local_mode: bool = False,
    stream: bool = True,
    prompt_interface: str = "cli",
) -> None:
    """
    Main agent loop - processes submissions and dispatches to handlers.
    This is the core of the agent (like submission_loop in codex.rs:1259-1340)
    """

    # Create session with tool router
    session = Session(
        event_queue, config=config, tool_router=tool_router, hf_token=hf_token,
        local_mode=local_mode, stream=stream, prompt_interface=prompt_interface,
    )
    if session_holder is not None:
        session_holder[0] = session
    logger.info("Agent loop started")

    # Retry any failed uploads from previous sessions (fire-and-forget)
    if config and config.save_sessions:
        Session.retry_failed_uploads_detached(
            directory="session_logs", repo_id=config.session_dataset_repo
        )

    try:
        # Main processing loop
        async with tool_router:
            # Emit ready event after initialization
            await session.send_event(
                Event(event_type="ready", data={
                    "message": "Agent initialized",
                    "tool_count": len(tool_router.tools),
                })
            )

            while session.is_running:
                submission = await submission_queue.get()

                try:
                    should_continue = await process_submission(session, submission)
                    if not should_continue:
                        break
                except asyncio.CancelledError:
                    logger.warning("Agent loop cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in agent loop: {e}")
                    await session.send_event(
                        Event(event_type="error", data={"error": str(e)})
                    )

        logger.info("Agent loop exited")

    finally:
        # Emergency save if session saving is enabled and shutdown wasn't called properly
        if session.config.save_sessions and session.is_running:
            logger.info("Emergency save: preserving session before exit...")
            try:
                local_path = session.save_and_upload_detached(
                    session.config.session_dataset_repo
                )
                if local_path:
                    logger.info("Emergency save successful, upload in progress")
            except Exception as e:
                logger.error(f"Emergency save failed: {e}")

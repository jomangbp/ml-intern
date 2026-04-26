"""Optional Telegram bot interface for ML Intern.

Started via `ml-intern gateway` which launches the backend + Telegram bot.
Uses long polling with per-chat sessions.

Gateway provides a Hermes-like live feed:
  - Single progress message (edited in-place) accumulating tool calls
  - Single stream message (edited in-place) for assistant text
  - Step counter + elapsed time ticker
  - Cron jobs that report results back to the originating chat
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx

from model_catalog import AVAILABLE_MODELS, format_models_for_text, resolve_model_choice
from prompt_cron import prompt_cron_manager
from session_manager import session_manager
from gateway.identity import identity_manager
from gateway.health import gateway_health, format_health_telegram
from events.event_store import event_store
from approvals.approval_store import approval_store
from jobs.local_job_manager import job_manager

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(os.environ.get("ML_INTERN_TELEGRAM_CONFIG", "~/.cache/ml-intern/telegram_bot.json")).expanduser()
_CRON_RE = re.compile(r"^/cron\s+(\d+(?:\.\d+)?)\s+([\s\S]+?)\s*$", re.IGNORECASE)

# ── Timing constants ─────────────────────────────────────────────
_TICKER_INTERVAL = 12.0       # Update status ticker every N seconds
_PROGRESS_EDIT_INTERVAL = 3.0  # Min seconds between progress message edits
_STREAM_EDIT_INTERVAL = 2.0    # Min seconds between stream message edits
_STREAM_BUFFER_THRESHOLD = 100 # Chars before first stream edit
_FINAL_SEND_TIMEOUT = 5.0     # Max wait for stream consumer to finish
_TYPING_INTERVAL = 6.0       # Typing indicator refresh


def _mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return value[:4] + ("*" * (len(value) - 8)) + value[-4:]


def _read_config_file() -> dict[str, Any]:
    try:
        if CONFIG_PATH.exists():
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to read Telegram config: %s", e)
    return {}


def _write_config_file(config: dict[str, Any]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_PATH.with_suffix(CONFIG_PATH.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, CONFIG_PATH)
    try:
        CONFIG_PATH.chmod(0o600)
    except OSError:
        pass


def _message_text(messages: list[dict[str, Any]]) -> str:
    """Extract text from the latest assistant message."""
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            text = content.strip()
            if text:
                return text
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    if isinstance(part.get("text"), str):
                        parts.append(part["text"])
                    elif isinstance(part.get("content"), str):
                        parts.append(part["content"])
            text = "\n".join(parts).strip()
            if text:
                return text
    return "Turn completed, but I could not extract an assistant message."


def _chunks(text: str, limit: int = 3900) -> list[str]:
    if not text:
        return [""]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        split_at = remaining.rfind("\n", 0, limit)
        if split_at < limit // 2:
            split_at = limit
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


# ── Tool display ──────────────────────────────────────────────────

_TOOL_ICONS: dict[str, str] = {
    "bash": "💻", "read_file": "📖", "write_file": "✏️", "edit_file": "📝",
    "list_directory": "📁", "search_files": "🔍", "web_search": "🌐",
    "research": "🔬", "local_training": "🏋️", "local_scheduler": "⏰",
    "hf_jobs": "🚀", "python": "🐍", "analysis": "📊",
}


def _tool_icon(name: str) -> str:
    return _TOOL_ICONS.get(name, "🔧")


def _format_tool_line(tool: str, args: dict) -> str:
    """One-line summary for progress accumulation (Hermes style)."""
    icon = _tool_icon(tool)
    if tool == "bash":
        return f'{icon} {tool}: "{str(args.get("command", ""))[:80]}"'
    if tool in ("read_file", "write_file", "edit_file"):
        return f'{icon} {tool}: "{args.get("path", "")}"'
    if tool == "list_directory":
        return f'{icon} ls: "{args.get("path", ".")}"'
    if tool == "search_files":
        return f'{icon} search: "{args.get("pattern", "")}"'
    if tool == "web_search":
        return f'{icon} web: "{args.get("query", "")}"'
    if tool == "local_training":
        return f'{icon} train: "{args.get("script", "")}"'
    if tool == "local_scheduler":
        return f'{icon} scheduler: {args.get("action", "")}'
    if tool == "hf_jobs":
        return f'{icon} hf_jobs: {args.get("action", "")}'
    # Generic
    preview = ""
    for k, v in list(args.items())[:2]:
        preview += f" {k}={str(v)[:40]}"
    return f"{icon} {tool}{preview}"


def _format_tool_result(tool: str, output: str, success: bool, max_len: int = 1000) -> str:
    """Format a tool output block."""
    icon = _tool_icon(tool)
    status = "✅" if success else "❌"
    if len(output) > max_len:
        output = output[:max_len] + f"\n... ({len(output) - max_len} more chars)"
    return f"{icon} {tool} {status}\n```\n{output}\n```"


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


# ── Stream Consumer (Hermes-style progressive edit) ───────────────

class StreamConsumer:
    """Accumulates assistant text deltas and edits a single Telegram message.

    Pattern from Hermes GatewayStreamConsumer:
      1. on_delta() receives text chunks
      2. Accumulates in buffer
      3. Periodically edits a single message with the growing text
      4. finish() sends final text
    """

    def __init__(self, bot: TelegramBotService, chat_id: int) -> None:
        self._bot = bot
        self._chat_id = chat_id
        self._buffer = ""
        self._msg_id: int | None = None
        self._last_edit = 0.0
        self._done = asyncio.Event()

    @property
    def text(self) -> str:
        return self._buffer

    def on_delta(self, text: str) -> None:
        self._buffer += text

    def finish(self) -> None:
        self._done.set()

    async def run(self) -> None:
        """Background task: periodically edit the stream message."""
        while not self._done.is_set():
            try:
                await asyncio.wait_for(self._done.wait(), timeout=_STREAM_EDIT_INTERVAL)
            except asyncio.TimeoutError:
                pass

            if not self._buffer:
                continue

            display = f"💬 {self._buffer[:3500]}"
            if self._msg_id:
                now = time.monotonic()
                if now - self._last_edit >= _STREAM_EDIT_INTERVAL:
                    await self._bot._edit_message(self._chat_id, self._msg_id, display)
                    self._last_edit = now
            else:
                if len(self._buffer) >= _STREAM_BUFFER_THRESHOLD:
                    self._msg_id = await self._bot._send_message(self._chat_id, display)

        # Final edit with complete text
        if self._msg_id and self._buffer:
            await self._bot._edit_message(
                self._chat_id, self._msg_id,
                f"💬 {self._buffer[:3500]}",
            )


# ── Progress Consumer (Hermes-style accumulated tool lines) ───────

class ProgressConsumer:
    """Accumulates tool progress lines into a single editable message.

    Pattern from Hermes send_progress_messages():
      - Each tool call adds a line: 💻 bash: "python3 train.py"
      - The message is edited in-place to show all lines
      - Finished tools get ✅/❌ appended
    """

    def __init__(self, bot: TelegramBotService, chat_id: int) -> None:
        self._bot = bot
        self._chat_id = chat_id
        self._lines: list[str] = []
        self._results: dict[int, str] = {}  # step index -> result text
        self._msg_id: int | None = None
        self._last_edit = 0.0
        self._lock = asyncio.Lock()

    @property
    def step_count(self) -> int:
        return len(self._lines)

    def add_tool(self, tool: str, args: dict) -> int:
        """Add a tool call line. Returns step index."""
        idx = len(self._lines)
        line = _format_tool_line(tool, args)
        self._lines.append(line)
        return idx

    def set_result(self, step_idx: int, tool: str, output: str, success: bool) -> None:
        """Set result for a completed tool step."""
        icon = _tool_icon(tool)
        status = "✅" if success else "❌"
        # Update the line in-place
        if step_idx < len(self._lines):
            self._lines[step_idx] = f"{self._lines[step_idx]} {status}"
        # Store full result
        self._results[step_idx] = _format_tool_result(tool, output, success)

    def cancel_step(self, step_idx: int) -> None:
        if step_idx < len(self._lines):
            self._lines[step_idx] = f"{self._lines[step_idx]} ⏹️"

    async def flush(self) -> None:
        """Send or edit the progress message."""
        async with self._lock:
            text = "\n".join(self._lines)
            now = time.monotonic()
            if self._msg_id and now - self._last_edit >= _PROGRESS_EDIT_INTERVAL:
                await self._bot._edit_message(self._chat_id, self._msg_id, text)
                self._last_edit = now
            elif not self._msg_id and text:
                self._msg_id = await self._bot._send_message(self._chat_id, text)
                self._last_edit = time.monotonic()

    async def send_results(self) -> None:
        """Send all tool results as separate messages (gateway mode)."""
        for idx, result in self._results.items():
            await self._bot._send_message(self._chat_id, result)

    @property
    def msg_id(self) -> int | None:
        return self._msg_id


# ── Telegram Bot Service ──────────────────────────────────────────

class TelegramBotService:
    def __init__(self) -> None:
        self._config = self._load_effective_config()
        self.token = str(self._config.get("token") or "").strip()
        self.allowed_chat_ids = set(self._config.get("allowed_chat_ids") or [])
        self.execution_mode = str(self._config.get("execution_mode") or "local").strip().lower()
        self.turn_timeout = int(self._config.get("turn_timeout_seconds") or 3600)
        self.base_url = f"https://api.telegram.org/bot{self.token}" if self.token else ""
        self._task: asyncio.Task | None = None
        self._client: httpx.AsyncClient | None = None
        self._offset = 0
        self._sessions: dict[int, str] = {}
        self._locks: dict[int, asyncio.Lock] = {}
        self._running = False
        self._gateway_enabled: dict[int, bool] = {}  # legacy, kept for compat
        self._current_identity = None

    def _load_effective_config(self) -> dict[str, Any]:
        file_config = _read_config_file()
        env_allowed = [
            item.strip()
            for item in os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS", "").split(",")
            if item.strip()
        ]
        token = str(file_config.get("token") or os.environ.get("TELEGRAM_BOT_TOKEN", "")).strip()
        return {
            "enabled": bool(file_config.get("enabled", bool(token))),
            "token": token,
            "allowed_chat_ids": file_config.get("allowed_chat_ids", env_allowed),
            "execution_mode": file_config.get("execution_mode", os.environ.get("TELEGRAM_EXECUTION_MODE", "local")),
            "turn_timeout_seconds": int(file_config.get("turn_timeout_seconds", os.environ.get("TELEGRAM_TURN_TIMEOUT_SECONDS", "3600"))),
        }

    def _apply_config(self, config: dict[str, Any]) -> None:
        self._config = config
        self.token = str(config.get("token") or "").strip()
        self.allowed_chat_ids = {str(x).strip() for x in config.get("allowed_chat_ids", []) if str(x).strip()}
        self.execution_mode = str(config.get("execution_mode") or "local").strip().lower()
        self.turn_timeout = int(config.get("turn_timeout_seconds") or 3600)
        self.base_url = f"https://api.telegram.org/bot{self.token}" if self.token else ""

    @property
    def enabled(self) -> bool:
        return bool(self._config.get("enabled", False) and self.token)

    @property
    def running(self) -> bool:
        return bool(self._task and not self._task.done() and self._running)

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "running": self.running,
            "configured": bool(self.token),
            "masked_token": _mask_secret(self.token),
            "allowed_chat_ids": sorted(self.allowed_chat_ids),
            "execution_mode": self.execution_mode,
            "turn_timeout_seconds": self.turn_timeout,
            "config_path": str(CONFIG_PATH),
            "commands": ["/start", "/help", "/commands", "/new", "/status", "/gateway", "/models", "/model <id|number|label>", "/sessions", "/crons", "/cancelcron <id>", "/interrupt", "/cron [minutes] <prompt>"],
        }

    async def configure(self, payload: dict[str, Any]) -> dict[str, Any]:
        config = self._load_effective_config()
        if payload.get("clear_token"):
            config["token"] = ""
        else:
            token = str(payload.get("token") or "").strip()
            if token:
                config["token"] = token
        if "enabled" in payload:
            config["enabled"] = bool(payload.get("enabled"))
        if "allowed_chat_ids" in payload:
            raw = payload.get("allowed_chat_ids")
            if isinstance(raw, str):
                ids = [item.strip() for item in raw.split(",") if item.strip()]
            elif isinstance(raw, list):
                ids = [str(item).strip() for item in raw if str(item).strip()]
            else:
                ids = []
            config["allowed_chat_ids"] = ids
        if "execution_mode" in payload:
            mode = str(payload.get("execution_mode") or "local").strip().lower()
            if mode not in {"local", "sandbox"}:
                raise ValueError("execution_mode must be local or sandbox")
            config["execution_mode"] = mode
        if "turn_timeout_seconds" in payload:
            timeout = int(payload.get("turn_timeout_seconds") or 3600)
            if timeout < 30:
                raise ValueError("turn_timeout_seconds must be >= 30")
            config["turn_timeout_seconds"] = timeout

        _write_config_file(config)
        was_running = self.running
        if was_running:
            await self.stop()
        self._apply_config(config)
        if self.enabled:
            await self.start()
        return self.status()

    async def start(self) -> None:
        if not self.enabled:
            logger.info("Telegram bot disabled")
            return
        if self._task and not self._task.done():
            return
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(45.0, connect=10.0))
        self._running = True
        self._task = asyncio.create_task(self._poll_loop(), name="ml-intern-telegram-bot")
        # Register bot commands so they appear as suggestions in the chat
        try:
            await self._api("setMyCommands", {
                "commands": [
                    {"command": "start", "description": "Show menu"},
                    {"command": "help", "description": "Show menu"},
                    {"command": "new", "description": "New session"},
                    {"command": "status", "description": "Session status"},
                    {"command": "models", "description": "Choose model"},
                    {"command": "gateway", "description": "Gateway info"},
                    {"command": "crons", "description": "List crons"},
                    {"command": "cancelcron", "description": "Cancel a cron"},
                    {"command": "sessions", "description": "Show sessions"},
                    {"command": "interrupt", "description": "Interrupt agent"},
                    {"command": "jobs", "description": "List local jobs"},
                    {"command": "logs", "description": "View job logs: /logs <id>"},
                    {"command": "kill", "description": "Kill a job: /kill <id>"},
                    {"command": "approvals", "description": "List pending approvals"},
                    {"command": "cron", "description": "Create cron: /cron <min> <prompt>"},
                ]
            })
        except Exception as e:
            logger.warning("Failed to register bot commands: %s", e)

        # Register submit factory and restore persisted crons
        prompt_cron_manager.set_submit_factory(self._make_cron_submit_fn)
        restored = await prompt_cron_manager.restore()
        if restored:
            logger.info("Restored %d Telegram crons", restored)

        # Restore pending approvals
        approval_restored = approval_store.restore()
        if approval_restored:
            logger.info("Restored %d pending approvals", approval_restored)

        # Restore jobs
        jobs_restored = job_manager.restore()
        if jobs_restored:
            logger.info("Restored %d job records", jobs_restored)

        event_store.log("gateway.started", source="gateway", payload={"adapters": ["telegram"]})
        logger.info("Telegram bot started")

    async def stop(self) -> None:
        self._running = False
        event_store.log("gateway.stopped", source="gateway")
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("Telegram bot stopped")

    # ── Telegram API ──────────────────────────────────────────────

    async def _api(self, method: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self._client:
            raise RuntimeError("Telegram client not started")
        # Retry on 429 with exponential backoff
        max_retries = 5
        for attempt in range(max_retries):
            response = await self._client.post(f"{self.base_url}/{method}", json=payload or {})
            if response.status_code == 429:
                # Parse Retry-After header or use exponential backoff
                retry_after = 1
                try:
                    body = response.json()
                    retry_after = body.get("parameters", {}).get("retry_after", 1)
                except Exception:
                    pass
                wait = max(retry_after, 2 ** attempt)
                logger.warning("TG API 429 rate limited, waiting %ds (attempt %d/%d)", wait, attempt + 1, max_retries)
                await asyncio.sleep(wait)
                continue
            response.raise_for_status()
            data = response.json()
            if not data.get("ok"):
                raise RuntimeError(f"Telegram API error: {data}")
            return data
        raise RuntimeError(f"Telegram API 429: max retries exceeded for {method}")

    async def _send_message(self, chat_id: int, text: str, parse_mode: str | None = None, reply_markup: dict | None = None) -> int | None:
        first_msg_id: int | None = None
        chunks_list = _chunks(text)
        for idx, chunk in enumerate(chunks_list):
            payload: dict[str, Any] = {
                "chat_id": chat_id,
                "text": chunk,
                "disable_web_page_preview": True,
            }
            if parse_mode:
                payload["parse_mode"] = parse_mode
            if reply_markup and idx == len(chunks_list) - 1:
                payload["reply_markup"] = reply_markup
            try:
                data = await self._api("sendMessage", payload)
            except Exception:
                # Retry without parse_mode if Markdown breaks
                if parse_mode:
                    payload.pop("parse_mode", None)
                    data = await self._api("sendMessage", payload)
                else:
                    raise
            msg_id = (data.get("result") or {}).get("message_id")
            if first_msg_id is None:
                first_msg_id = msg_id
            # Throttle between chunks to avoid 429
            if len(chunks_list) > 1 and idx < len(chunks_list) - 1:
                await asyncio.sleep(0.5)
        return first_msg_id

    async def _edit_message(self, chat_id: int, message_id: int, text: str, parse_mode: str | None = None) -> None:
        try:
            payload: dict[str, Any] = {"chat_id": chat_id, "message_id": message_id, "text": text}
            if parse_mode:
                payload["parse_mode"] = parse_mode
            await self._api("editMessageText", payload)
        except Exception:
            pass

    async def _delete_message(self, chat_id: int, message_id: int) -> None:
        try:
            await self._api("deleteMessage", {"chat_id": chat_id, "message_id": message_id})
        except Exception:
            pass

    async def _send_typing(self, chat_id: int) -> None:
        try:
            await self._api("sendChatAction", {"chat_id": chat_id, "action": "typing"})
        except Exception:
            pass

    def _allowed(self, chat_id: int) -> bool:
        return not self.allowed_chat_ids or str(chat_id) in self.allowed_chat_ids

    # ── Polling ───────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                data = await self._api("getUpdates", {
                    "offset": self._offset,
                    "timeout": 30,
                    "allowed_updates": ["message", "callback_query"],
                })
                for update in data.get("result", []):
                    self._offset = max(self._offset, int(update.get("update_id", 0)) + 1)
                    asyncio.create_task(self._handle_update(update))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("Telegram poll error: %s", e)
                await asyncio.sleep(5)

    # ── Session management ────────────────────────────────────────

    async def _ensure_session(self, chat_id: int) -> str:
        existing = self._sessions.get(chat_id)
        if existing:
            agent_session = session_manager.sessions.get(existing)
            if agent_session and agent_session.is_active:
                return existing
        local_mode = self.execution_mode != "sandbox"
        user_id = f"telegram:{chat_id}"
        provider_keys = session_manager.get_effective_provider_keys(user_id)
        session_id = await session_manager.create_session(
            user_id=user_id, local_mode=local_mode, provider_keys=provider_keys,
        )
        self._sessions[chat_id] = session_id
        return session_id

    async def _new_session(self, chat_id: int) -> str:
        existing = self._sessions.pop(chat_id, None)
        if existing:
            await session_manager.delete_session(existing)
        return await self._ensure_session(chat_id)

    # ── Message routing ───────────────────────────────────────────

    async def _handle_update(self, update: dict[str, Any]) -> None:
        # Extract user info from either message or callback_query
        user_id = None
        message = update.get("message") or {}
        callback_query = update.get("callback_query") or {}

        if callback_query:
            from_user = callback_query.get("from") or {}
            user_id = from_user.get("id")
        elif message:
            from_user = message.get("from") or {}
            user_id = from_user.get("id")

        # Resolve identity
        if user_id:
            display_name = (from_user.get("first_name", "") + " " + from_user.get("last_name", "")).strip()
            self._current_identity = identity_manager.resolve_or_create(
                platform="telegram",
                platform_user_id=user_id,
                display_name=display_name,
            )
        else:
            self._current_identity = None

        if callback_query:
            await self._handle_callback_query(callback_query)
            return

        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        text = (message.get("text") or "").strip()
        if not isinstance(chat_id, int) or not text:
            return

        # Chat-level allowlist (routing filter)
        if not self._allowed(chat_id):
            event_store.log("telegram.unauthorized", source="telegram", platform="telegram",
                            chat_id=chat_id, payload={"user_id": user_id, "text": text[:50]})
            await self._send_message(chat_id, "This chat is not allowed.")
            return

        # User-level authorization for commands
        if text.startswith("/") and self._current_identity:
            cmd_name = text.split()[0].lstrip("/").split("@")[0]  # handle /command@botname
            allowed, _ = identity_manager.check_command_permission("telegram", user_id, cmd_name)
            if not allowed:
                event_store.log("gateway.unauthorized", source="telegram", platform="telegram",
                                identity_id=self._current_identity.identity_id, chat_id=chat_id,
                                payload={"command": cmd_name, "user_id": user_id})
                await self._send_message(chat_id, "⛔ You don't have permission for this command.")
                return

        lock = self._locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            try:
                await self._route_message(chat_id, text)
            except Exception as e:
                logger.exception("Telegram update failed")
                await self._send_message(chat_id, f"❌ Error: {e}")

    async def _route_message(self, chat_id: int, text: str) -> None:
        if text in {"/start", "/help", "/commands"}:
            await self._send_main_menu(chat_id)
            return

        if text == "/new":
            sid = await self._new_session(chat_id)
            gw = "🔴 ON" if self._gateway_enabled.get(chat_id) else "⚪ OFF"
            await self._send_message(chat_id, f"✅ New session\nGateway: {gw}")
            return

        session_id = await self._ensure_session(chat_id)

        if text == "/status":
            agent_session = session_manager.sessions.get(session_id)
            model = agent_session.session.config.model_name if agent_session else "?"
            gw = "🔴 ON" if self._gateway_enabled.get(chat_id) else "⚪ OFF"
            await self._send_message(
                chat_id,
                f"📊 *Status*\nModel: `{model}`\nGateway: {gw}\nMode: `{self.execution_mode}`",
                parse_mode="Markdown",
            )
            return

        if text == "/gateway" or text == "/gateway status":
            active_sessions = sum(1 for s in session_manager.sessions.values() if s.is_active)
            crons = await prompt_cron_manager.list()
            active_crons = sum(1 for c in crons if c.get("status") in ("scheduled", "running"))
            health = gateway_health(
                telegram_running=True,
                telegram_enabled=True,
                active_sessions=active_sessions,
                active_crons=active_crons,
                running_jobs=job_manager.running_count(),
                event_stats=event_store.stats(),
            )
            await self._send_message(chat_id, format_health_telegram(health), parse_mode="Markdown")
            return

        if text == "/sessions":
            await self._send_message(chat_id, f"Session: `{session_id[:8]}...`", parse_mode="Markdown")
            return
        if text == "/models":
            await self._send_models_menu(chat_id, session_id)
            return
        if text.startswith("/model"):
            parts = text.split(maxsplit=1)
            if len(parts) != 2:
                await self._send_models_menu(chat_id, session_id)
                return
            await self._switch_model(chat_id, session_id, parts[1])
            return
        if text == "/crons":
            tasks = await prompt_cron_manager.list(user_id=f"telegram:{chat_id}")
            if not tasks:
                await self._send_message(chat_id, "No crons.")
            else:
                lines = ["⏰ Crons:"]
                for t in tasks[:20]:
                    cfg = t.get("config", {})
                    lines.append(f"- `{t['task_id']}` · {t['status']} · {cfg.get('interval_minutes')}min")
                await self._send_message(chat_id, "\n".join(lines), parse_mode="Markdown")
            return
        if text.startswith("/cancelcron"):
            parts = text.split(maxsplit=1)
            if len(parts) != 2:
                await self._send_message(chat_id, "Usage: /cancelcron <id>")
            else:
                ok = await prompt_cron_manager.cancel(parts[1].strip())
                await self._send_message(chat_id, "Cancelled." if ok else "Not found.")
            return
        if text == "/interrupt":
            ok = await session_manager.interrupt(session_id)
            await self._send_message(chat_id, "🛑 Interrupted." if ok else "Nothing to interrupt.")
            return
        if text == "/jobs":
            await self._cmd_jobs(chat_id)
            return
        if text.startswith("/logs"):
            parts = text.split(maxsplit=1)
            if len(parts) != 2:
                await self._send_message(chat_id, "Usage: /logs <job_id>")
            else:
                await self._cmd_logs(chat_id, parts[1].strip())
            return
        if text.startswith("/kill"):
            parts = text.split(maxsplit=1)
            if len(parts) != 2:
                await self._send_message(chat_id, "Usage: /kill <job_id>")
            else:
                await self._cmd_kill(chat_id, parts[1].strip())
            return
        if text == "/approvals":
            pending = approval_store.list_pending(platform="telegram", chat_id=chat_id)
            if not pending:
                await self._send_message(chat_id, "🔐 No pending approvals.")
            else:
                lines = [f"🔐 *{len(pending)} Pending Approval(s):*\n"]
                for p in pending[:5]:
                    age = int(time.time() - p.created_at)
                    lines.append(f"`{p.approval_id[:12]}` · {age}s ago")
                    lines.append(p.summary)
                    lines.append("")
                await self._send_message(chat_id, "\n".join(lines), parse_mode="Markdown")
            return

        cron_match = _CRON_RE.match(text)
        if cron_match:
            mins = float(cron_match.group(1))
            prompt = cron_match.group(2).strip()
            s = await self._create_telegram_cron(chat_id, session_id, mins, prompt)
            await self._send_message(chat_id, s, parse_mode="Markdown")
            return

        await self._run_agent_turn(chat_id, session_id, text)

    async def _send_main_menu(self, chat_id: int) -> None:
        """Send the main menu with inline keyboard buttons."""
        keyboard = [
            [
                {"text": "📊 Status", "callback_data": "cmd:status"},
                {"text": "🆕 New Session", "callback_data": "cmd:new"},
            ],
            [
                {"text": "🤖 Models", "callback_data": "cmd:models"},
                {"text": "🔌 Gateway", "callback_data": "cmd:gateway"},
            ],
            [
                {"text": "⏰ Crons", "callback_data": "cmd:crons"},
                {"text": "📂 Sessions", "callback_data": "cmd:sessions"},
            ],
            [
                {"text": "🛑 Interrupt", "callback_data": "cmd:interrupt"},
                {"text": "🔧 Jobs", "callback_data": "cmd:jobs"},
            ],
            [
                {"text": "🔐 Approvals", "callback_data": "cmd:approvals"},
            ],
        ]
        help_text = (
            "🤖 *ML Intern Gateway*\n\n"
            "Tap a button or type a command:\n\n"
            "*Quick commands:*\n"
            "/cron 30 check training progress\n"
            "/model gpt-5.5\n"
            "/cancelcron <id>\n\n"
            "Anything else → sent to the agent."
        )
        await self._api("sendMessage", {
            "chat_id": chat_id,
            "text": help_text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
            "reply_markup": {"inline_keyboard": keyboard},
        })

    # ── Job Commands ─────────────────────────────────────────────

    async def _cmd_jobs(self, chat_id: int) -> None:
        """List local jobs."""
        running = job_manager.list_jobs(status="running")
        recent = job_manager.list_jobs(limit=10)

        if not recent:
            await self._send_message(chat_id, "No jobs found.")
            return

        lines = [f"🔧 *Jobs* ({len(running)} running)\n"]
        for j in recent[:10]:
            icon = {"running": "🟢", "completed": "✅", "failed": "❌", "cancelled": "⚪", "killed": "💀"}.get(j.status, "❓")
            cmd_preview = j.command[:50]
            elapsed = j.elapsed if j.status == "running" else ""
            lines.append(f"{icon} `{j.job_id[:12]}` · {j.kind} · {j.status} {elapsed}")
            lines.append(f"   `{cmd_preview}`")

        await self._send_message(chat_id, "\n".join(lines), parse_mode="Markdown")

    async def _cmd_logs(self, chat_id: int, job_id: str) -> None:
        """Show last lines of a job log."""
        # Support partial match
        record = job_manager.get_job(job_id)
        if not record:
            for j in job_manager.list_jobs():
                if j.job_id.startswith(job_id):
                    record = j
                    break
        if not record:
            await self._send_message(chat_id, f"Job `{job_id}` not found.")
            return

        logs = job_manager.tail_logs(record.job_id, lines=50)
        header = f"📋 Logs for `{record.job_id[:12]}`:\n"
        await self._send_message(chat_id, header + "```\n" + logs[:3500] + "\n```")

    async def _cmd_kill(self, chat_id: int, job_id: str) -> None:
        """Kill a running job."""
        record = job_manager.get_job(job_id)
        if not record:
            for j in job_manager.list_jobs():
                if j.job_id.startswith(job_id):
                    record = j
                    break
        if not record:
            await self._send_message(chat_id, f"Job `{job_id}` not found.")
            return
        if record.status != "running":
            await self._send_message(chat_id, f"Job `{record.job_id[:12]}` is not running (status: {record.status}).")
            return

        result = await job_manager.kill_job(record.job_id)
        if result:
            await self._send_message(chat_id, f"💀 Killed job `{record.job_id[:12]}` (pid {record.pid})")
        else:
            await self._send_message(chat_id, f"❌ Failed to kill job.")

    # ── Cron Factory ──────────────────────────────────────────────

    def _make_cron_submit_fn(self, chat_id_str: str, session_id: str, config: dict) -> SubmitPrompt:
        """Create a submit function for a cron job. Used by prompt_cron_manager.restore()."""
        chat_id = int(chat_id_str)
        prompt = config.get("prompt", "")
        interval = config.get("interval_minutes", 10)
        # Return the same factory used by _create_telegram_cron but as a simple submit fn
        bot = self

        async def _submit(sid: str, text: str) -> bool:
            try:
                logger.info("TG cron %d (restored): firing prompt=%s", chat_id, text[:60])
                await bot._send_message(chat_id, f"⏰ *Cron:* {text[:80]}" + ("..." if len(text) > 80 else ""), parse_mode="Markdown")

                # Ensure session
                agent_session = session_manager.sessions.get(sid)
                if not agent_session or not agent_session.is_active:
                    sid = await bot._ensure_session(chat_id)
                    agent_session = session_manager.sessions[sid]

                # Wait for broadcaster
                for _ in range(50):
                    if agent_session.broadcaster is not None:
                        break
                    await asyncio.sleep(0.1)
                else:
                    await bot._send_message(chat_id, "❌ Cron: session not ready")
                    return False

                # Subscribe and submit
                broadcaster = agent_session.broadcaster
                sub_id, event_queue = broadcaster.subscribe()
                ok = await session_manager.submit_user_input(sid, text)
                if not ok:
                    broadcaster.unsubscribe(sub_id)
                    await bot._send_message(chat_id, "❌ Cron: submit failed")
                    return False

                # Wait for result
                final_text = None
                timeout = bot.turn_timeout
                try:
                    while True:
                        try:
                            event = await asyncio.wait_for(event_queue.get(), timeout=timeout)
                        except asyncio.TimeoutError:
                            await bot._send_message(chat_id, f"⏳ Cron timeout ({timeout}s)")
                            break
                        et = event.get("event_type")
                        data = event.get("data", {})
                        if et == "assistant_message":
                            final_text = data.get("content", "")
                        elif et == "turn_complete":
                            if not final_text:
                                messages = [msg.model_dump() for msg in agent_session.session.context_manager.items]
                                final_text = _message_text(messages)
                            break
                        elif et in {"error", "interrupted", "shutdown"}:
                            err = data.get("error", "") if et == "error" else "interrupted"
                            await bot._send_message(chat_id, f"❌ Cron: {err}")
                            break
                finally:
                    broadcaster.unsubscribe(sub_id)

                if final_text:
                    await bot._send_message(chat_id, f"🤖 {final_text[:3900]}")
                return True
            except Exception as e:
                logger.exception("TG cron %d: failed", chat_id)
                try:
                    await bot._send_message(chat_id, f"❌ Cron error: {e}")
                except Exception:
                    pass
                return False

        return _submit

    async def _create_telegram_cron(
        self, chat_id: int, session_id: str, interval_minutes: float, prompt: str,
    ) -> str:
        """Create a prompt cron that reports results back to this Telegram chat."""
        bot = self  # capture for closure

        async def _submit_and_report(sid: str, text: str) -> bool:
            """Submit prompt to agent, wait for response, send to Telegram."""
            try:
                logger.info("TG cron %d: firing prompt=%s", chat_id, text[:60])
                await bot._send_message(chat_id, f"⏰ *Cron:* {text[:80]}" + ("..." if len(text) > 80 else ""), parse_mode="Markdown")

                # Ensure session is alive
                agent_session = session_manager.sessions.get(sid)
                if not agent_session or not agent_session.is_active:
                    logger.info("TG cron %d: session dead, recreating", chat_id)
                    sid = await bot._new_session(chat_id)
                    agent_session = session_manager.sessions[sid]
                    logger.info("TG cron %d: new session %s", chat_id, sid[:8])

                # Wait for broadcaster
                for _ in range(50):
                    if agent_session.broadcaster is not None:
                        break
                    await asyncio.sleep(0.1)
                else:
                    logger.error("TG cron %d: broadcaster never ready", chat_id)
                    await bot._send_message(chat_id, "❌ Cron: session not ready")
                    return False

                # Subscribe to events
                broadcaster = agent_session.broadcaster
                sub_id, event_queue = broadcaster.subscribe()

                # Submit prompt (outside chat lock — the queue handles serialization)
                ok = await session_manager.submit_user_input(sid, text)
                if not ok:
                    broadcaster.unsubscribe(sub_id)
                    await bot._send_message(chat_id, "❌ Cron: failed to submit")
                    return False

                # Wait for turn_complete with timeout
                final_text = None
                timeout = bot.turn_timeout
                try:
                    while True:
                        try:
                            event = await asyncio.wait_for(event_queue.get(), timeout=timeout)
                        except asyncio.TimeoutError:
                            logger.warning("TG cron %d: turn timeout after %ds", chat_id, timeout)
                            await bot._send_message(chat_id, f"⏳ Cron timeout ({timeout}s)")
                            break

                        et = event.get("event_type")
                        data = event.get("data", {})

                        if et == "assistant_message":
                            final_text = data.get("content", "")
                        elif et == "turn_complete":
                            if not final_text:
                                messages = [msg.model_dump() for msg in agent_session.session.context_manager.items]
                                final_text = _message_text(messages)
                            break
                        elif et in {"error", "interrupted", "shutdown"}:
                            err = data.get("error", "") if et == "error" else "interrupted"
                            await bot._send_message(chat_id, f"❌ Cron: {err}")
                            final_text = None
                            break
                finally:
                    broadcaster.unsubscribe(sub_id)

                # Send response
                if final_text:
                    logger.info("TG cron %d: sending response (%d chars)", chat_id, len(final_text))
                    await bot._send_message(chat_id, f"🤖 {final_text[:3900]}")
                return True

            except Exception as e:
                logger.exception("TG cron %d: execution failed", chat_id)
                try:
                    await bot._send_message(chat_id, f"❌ Cron error: {e}")
                except Exception:
                    pass
                return False

        result = await prompt_cron_manager.create(
            session_id=session_id,
            user_id=f"telegram:{chat_id}",
            interval_minutes=interval_minutes,
            prompt=prompt,
            submit_prompt=_submit_and_report,
            task_name=f"TG cron {interval_minutes:g}m",
            repeat=True,
            max_runs=0,
        )
        task_id = result["task_id"]
        logger.info("TG %d: created cron %s every %gm", chat_id, task_id, interval_minutes)
        return f"⏰ Cron `{task_id}` every {interval_minutes:g}m\nPrompt: _{prompt[:100]}_"

    # ── Model menu ────────────────────────────────────────────────

    async def _send_models_menu(self, chat_id: int, session_id: str) -> None:
        agent_session = session_manager.sessions.get(session_id)
        current = agent_session.session.config.model_name if agent_session else ""
        rows: list[list[dict[str, str]]] = []
        row: list[dict[str, str]] = []
        for m in AVAILABLE_MODELS:
            label = f"✓ {m['label']}" if m["id"] == current else m["label"]
            row.append({"text": label, "callback_data": f"model:{m['id']}"})
            if len(row) == 2:
                rows.append(row)
                row = []
        if row:
            rows.append(row)
        await self._api("sendMessage", {
            "chat_id": chat_id,
            "text": f"Current: *{current}*\nTap to switch:",
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
            "reply_markup": {"inline_keyboard": rows},
        })

    async def _switch_model(self, chat_id: int, session_id: str, choice: str) -> None:
        model_id = resolve_model_choice(choice)
        if not model_id:
            await self._send_message(chat_id, "Unknown model. /models to list.")
            return
        agent_session = session_manager.sessions.get(session_id)
        if not agent_session or not agent_session.is_active:
            await self._send_message(chat_id, "No active session. /new first.")
            return
        agent_session.session.update_model(model_id)
        await self._send_message(chat_id, f"✅ → `{model_id}`", parse_mode="Markdown")

    async def _handle_callback_query(self, callback_query: dict[str, Any]) -> None:
        query_id = callback_query.get("id", "")
        chat = (callback_query.get("message") or {}).get("chat") or {}
        chat_id = chat.get("id")
        data = callback_query.get("data", "")
        from_user = callback_query.get("from") or {}
        user_id = from_user.get("id")

        try:
            await self._api("answerCallbackQuery", {"callback_query_id": query_id})
        except Exception:
            pass
        if not isinstance(chat_id, int) or not data:
            return
        if not self._allowed(chat_id):
            return

        # User-level authorization for callback buttons
        if user_id:
            action = data.split(":")[-1] if ":" in data else data
            if data.startswith("model:"):
                action = "model"
            elif data.startswith("cmd:"):
                action = data[4:]
            allowed, _ = identity_manager.check_command_permission("telegram", user_id, action)
            if not allowed:
                event_store.log("gateway.unauthorized", source="telegram", platform="telegram",
                                chat_id=chat_id, payload={"callback_data": data, "user_id": user_id})
                await self._send_message(chat_id, "⛔ Not authorized.")
                return

        lock = self._locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            try:
                if data.startswith("model:"):
                    session_id = await self._ensure_session(chat_id)
                    await self._switch_model(chat_id, session_id, data[len("model:"):])
                elif data.startswith("cmd:"):
                    await self._handle_menu_button(chat_id, data[4:])
                elif data.startswith("approve:"):
                    await self._handle_approve(chat_id, data[len("approve:"):])
                elif data.startswith("reject:"):
                    await self._handle_reject(chat_id, data[len("reject:"):])
                elif data.startswith("details:"):
                    await self._handle_approval_details(chat_id, data[len("details:"):])
            except Exception as e:
                logger.exception("Callback failed")
                await self._send_message(chat_id, f"❌ {e}")

    # ── Approval Flow ──────────────────────────────────────────

    async def _send_approval_request(self, chat_id: int, session_id: str, tools: list[dict]) -> None:
        """Send an approval request with inline buttons to Telegram."""
        record = approval_store.create(
            session_id=session_id,
            tools=tools,
            platform="telegram",
            chat_id=chat_id,
            identity_id=getattr(self, '_current_identity', None) and self._current_identity.identity_id or "",
            expiry_seconds=600,
        )

        # Build message
        lines = ["⚠️ *Approval Required*\n"]
        lines.append(f"Session: `{session_id[:8]}...`")
        lines.append(f"Expires: 10 min\n")
        for t in tools:
            name = t.get("tool", "?")
            args = t.get("arguments", {})
            if name == "bash":
                cmd = str(args.get("command", ""))[:100]
                lines.append(f"💻 `{cmd}`")
            elif name in ("write_file", "edit_file"):
                path = str(args.get("path", ""))[:80]
                lines.append(f"✏️ {name}: `{path}`")
            elif name == "local_training":
                script = str(args.get("script", ""))[:80]
                lines.append(f"🏋️ train: `{script}`")
            else:
                preview = json.dumps(args, default=str)[:80]
                lines.append(f"🔧 {name}: `{preview}`")

        keyboard = {
            "inline_keyboard": [
                [
                    {"text": "✅ Approve", "callback_data": f"approve:{record.approval_id}"},
                    {"text": "❌ Reject", "callback_data": f"reject:{record.approval_id}"},
                ],
                [
                    {"text": "📋 Details", "callback_data": f"details:{record.approval_id}"},
                ],
            ],
        }

        msg_id = await self._send_message(
            chat_id, "\n".join(lines), parse_mode="Markdown", reply_markup=keyboard,
        )
        if msg_id:
            approval_store.set_message_id(record.approval_id, msg_id)

    async def _handle_approve(self, chat_id: int, approval_id: str) -> None:
        """Handle approve button press."""
        record = await approval_store.approve(approval_id)
        if not record:
            await self._send_message(chat_id, "❌ Approval not found.")
            return
        if record.status == "expired":
            await self._send_message(chat_id, "⏰ This approval has expired.")
            return
        if record.status == "approved":
            # Update the approval message
            if record.message_id:
                await self._edit_message(chat_id, record.message_id, "✅ *Approved* (from Telegram)", parse_mode="Markdown")
            else:
                await self._send_message(chat_id, "✅ Approved.")
        else:
            await self._send_message(chat_id, f"❌ Approval failed: {record.status}")

    async def _handle_reject(self, chat_id: int, approval_id: str) -> None:
        """Handle reject button press."""
        record = await approval_store.reject(approval_id)
        if not record:
            await self._send_message(chat_id, "❌ Approval not found.")
            return
        if record.status == "rejected":
            if record.message_id:
                await self._edit_message(chat_id, record.message_id, "❌ *Rejected* (from Telegram)", parse_mode="Markdown")
            else:
                await self._send_message(chat_id, "❌ Rejected.")
        else:
            await self._send_message(chat_id, f"⚠️ Unexpected status: {record.status}")

    async def _handle_approval_details(self, chat_id: int, approval_id: str) -> None:
        """Show full details of an approval request."""
        record = approval_store.get(approval_id)
        if not record:
            await self._send_message(chat_id, "Approval not found.")
            return
        await self._send_message(chat_id, f"📋 *Approval Details*\n\n{record.details}")

    # ── Menu Buttons ───────────────────────────────────────────

    async def _handle_menu_button(self, chat_id: int, action: str) -> None:
        """Handle inline keyboard menu button presses."""
        if action == "status":
            session_id = await self._ensure_session(chat_id)
            agent_session = session_manager.sessions.get(session_id)
            model = agent_session.session.config.model_name if agent_session else "?"
            await self._send_message(
                chat_id,
                f"📊 *Status*\nModel: `{model}`\nGateway: 🔴 Active\nMode: `{self.execution_mode}`",
                parse_mode="Markdown",
            )
        elif action == "new":
            sid = await self._new_session(chat_id)
            await self._send_message(chat_id, f"✅ New session `{sid[:8]}...`", parse_mode="Markdown")
        elif action == "models":
            session_id = await self._ensure_session(chat_id)
            await self._send_models_menu(chat_id, session_id)
        elif action == "gateway":
            await self._send_message(chat_id, "🔌 *Gateway Active*\nLive tool feed is always on.", parse_mode="Markdown")
        elif action == "crons":
            tasks = await prompt_cron_manager.list(user_id=f"telegram:{chat_id}")
            if not tasks:
                await self._send_message(chat_id, "No active crons.\nCreate one: `/cron 10 check training`", parse_mode="Markdown")
            else:
                lines = ["⏰ *Active Crons:*"]
                for t in tasks[:10]:
                    cfg = t.get("config", {})
                    runs = t.get("runs_completed", 0)
                    status_icon = "🟢" if t.get("status") == "scheduled" else "🔴" if t.get("status") == "running" else "⚪"
                    prompt_preview = cfg.get("prompt", "?")[:50]
                    lines.append(f"{status_icon} `{t['task_id'][:12]}` · {cfg.get('interval_minutes')}m · {runs} runs")
                    lines.append(f"   _{prompt_preview}_")
                await self._send_message(chat_id, "\n".join(lines), parse_mode="Markdown")
        elif action == "sessions":
            session_id = await self._ensure_session(chat_id)
            await self._send_message(chat_id, f"📂 Session: `{session_id[:8]}...`", parse_mode="Markdown")
        elif action == "interrupt":
            session_id = await self._ensure_session(chat_id)
            ok = await session_manager.interrupt(session_id)
            await self._send_message(chat_id, "🛑 Interrupted." if ok else "Nothing to interrupt.")
        elif action == "jobs":
            await self._cmd_jobs(chat_id)
        elif action == "approvals":
            pending = approval_store.list_pending(platform="telegram", chat_id=chat_id)
            if not pending:
                await self._send_message(chat_id, "🔐 No pending approvals.")
            else:
                lines = [f"🔐 *{len(pending)} Pending Approval(s):*\n"]
                for p in pending[:5]:
                    age = int(time.time() - p.created_at)
                    lines.append(f"`{p.approval_id[:12]}` · {age}s ago")
                    lines.append(p.summary)
                    lines.append("")
                await self._send_message(chat_id, "\n".join(lines))

    # ── Gateway: Agent Turn ────────────────────────────────────────

    async def _run_agent_turn(self, chat_id: int, session_id: str, text: str) -> None:
        logger.info("TG %d: agent turn start session=%s text=%s", chat_id, session_id[:8], text[:80])
        agent_session = session_manager.sessions.get(session_id)
        if not agent_session or not agent_session.is_active:
            logger.info("TG %d: dead session, creating new", chat_id)
            session_id = await self._new_session(chat_id)
            agent_session = session_manager.sessions[session_id]
        if agent_session.is_processing:
            logger.warning("TG %d: already processing", chat_id)
            await self._send_message(chat_id, "⚠️ Still processing. /interrupt to stop.")
            return

        # Wait for broadcaster to be ready (set in _run_session)
        for _ in range(50):  # 5 seconds max
            if agent_session.broadcaster is not None:
                break
            await asyncio.sleep(0.1)
        else:
            logger.error("TG %d: broadcaster never ready", chat_id)
            await self._send_message(chat_id, "❌ Session not ready. Try /new.")
            return

        gateway = True  # Gateway is always active when bot is running
        broadcaster = agent_session.broadcaster
        sub_id, event_queue = broadcaster.subscribe()
        await self._send_typing(chat_id)

        success = await session_manager.submit_user_input(session_id, text)
        if not success:
            broadcaster.unsubscribe(sub_id)
            await self._send_message(chat_id, "❌ Session dead. /new")
            return

        # ── Setup gateway consumers ──
        start_time = time.monotonic()
        typing_task = asyncio.create_task(self._typing_loop(chat_id))

        progress: ProgressConsumer | None = None
        stream: StreamConsumer | None = None
        progress_task: asyncio.Task | None = None
        stream_task: asyncio.Task | None = None
        ticker_msg_id: int | None = None

        # Map tool_call_id -> step index in progress consumer
        tc_step_map: dict[str, int] = {}

        if gateway:
            progress = ProgressConsumer(self, chat_id)
            stream = StreamConsumer(self, chat_id)
            ticker_msg_id = await self._send_message(chat_id, "⏱ 0s · starting...")
            progress_task = asyncio.create_task(self._progress_loop(progress, chat_id, ticker_msg_id, start_time))
            stream_task = asyncio.create_task(stream.run())

        terminal_event = "timeout"

        try:
            while True:
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=self.turn_timeout)
                except asyncio.TimeoutError:
                    terminal_event = "timeout"
                    break

                et = event.get("event_type")
                data = event.get("data", {})

                if et == "tool_call":
                    tc_id = data.get("tool_call_id", "")
                    tool_name = data.get("tool", "unknown")
                    args = data.get("arguments", {})
                    logger.debug("TG %d: tool_call %s", chat_id, tool_name)
                    if gateway and progress:
                        idx = progress.add_tool(tool_name, args)
                        if tc_id:
                            tc_step_map[tc_id] = idx
                        await progress.flush()

                elif et == "tool_output":
                    tc_id = data.get("tool_call_id", "")
                    tool_name = data.get("tool", "unknown")
                    output = str(data.get("output", ""))
                    ok = data.get("success", True)
                    if gateway and progress:
                        step_idx = tc_step_map.pop(tc_id, -1)
                        if step_idx >= 0:
                            progress.set_result(step_idx, tool_name, output, ok)
                        await progress.flush()

                elif et == "tool_state_change":
                    state = data.get("state", "")
                    tc_id = data.get("tool_call_id", "")
                    if state == "cancelled" and gateway and progress:
                        step_idx = tc_step_map.pop(tc_id, -1)
                        if step_idx >= 0:
                            progress.cancel_step(step_idx)
                        await progress.flush()

                elif et == "compacted" and gateway:
                    old_t = data.get("old_tokens", 0)
                    new_t = data.get("new_tokens", 0)
                    await self._send_message(chat_id, f"📦 Compacted: {old_t // 1000}k → {new_t // 1000}k tokens")

                elif et == "assistant_chunk":
                    chunk = data.get("content", "")
                    if chunk and gateway and stream:
                        stream.on_delta(chunk)

                elif et == "assistant_message":
                    content = data.get("content", "")
                    if content and gateway and stream:
                        stream.on_delta(content)

                elif et in {"turn_complete", "approval_required", "error", "interrupted", "shutdown"}:
                    terminal_event = et
                    elapsed = _fmt_elapsed(time.monotonic() - start_time)
                    steps = progress.step_count if progress else 0
                    logger.info("TG %d: terminal event=%s elapsed=%s steps=%d", chat_id, et, elapsed, steps)

                    if et == "error":
                        err = str(data.get("error", "unknown"))[:1000]
                        await self._send_message(chat_id, f"❌ Error ({elapsed}):\n{err}")
                    elif et == "approval_required":
                        tools = data.get("tools", [])
                        await self._send_approval_request(chat_id, session_id, tools)
                    break

        finally:
            # Cleanup background tasks
            if stream:
                stream.finish()
            if stream_task:
                try:
                    await asyncio.wait_for(stream_task, timeout=_FINAL_SEND_TIMEOUT)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    stream_task.cancel()
            if progress_task:
                progress_task.cancel()
                try:
                    await progress_task
                except asyncio.CancelledError:
                    pass
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass
            broadcaster.unsubscribe(sub_id)

        # ── Final delivery ──
        elapsed = _fmt_elapsed(time.monotonic() - start_time)
        steps = progress.step_count if progress else 0
        logger.info("TG %d: final delivery event=%s elapsed=%s steps=%d", chat_id, terminal_event, elapsed, steps)

        # Remove stream placeholder (gateway)
        if gateway and stream and stream._msg_id:
            await self._delete_message(chat_id, stream._msg_id)

        # Update ticker → done
        if gateway and ticker_msg_id:
            await self._edit_message(chat_id, ticker_msg_id, f"✅ Done · {elapsed} · {steps} steps")

        # Send tool results (gateway)
        if gateway and progress:
            await progress.send_results()

        # Send final assistant response
        if terminal_event == "turn_complete":
            if stream and stream.text:
                final_text = stream.text
            else:
                messages = [msg.model_dump() for msg in agent_session.session.context_manager.items]
                final_text = _message_text(messages)
            if final_text:
                footer = f"\n\n_{elapsed}, {steps} steps_" if gateway else ""
                await self._send_message(chat_id, f"🤖 {final_text}{footer}")
        elif terminal_event == "timeout":
            await self._send_message(chat_id, f"⏳ Still running ({elapsed}). Open Web UI for live view.")
        elif terminal_event == "interrupted":
            await self._send_message(chat_id, f"🛑 Interrupted · {elapsed} · {steps} steps.")

    async def _progress_loop(
        self,
        progress: ProgressConsumer,
        chat_id: int,
        ticker_msg_id: int,
        start_time: float,
    ) -> None:
        """Background task: periodically update ticker with elapsed + step count."""
        try:
            while True:
                await asyncio.sleep(_TICKER_INTERVAL)
                elapsed = _fmt_elapsed(time.monotonic() - start_time)
                steps = progress.step_count
                await self._edit_message(chat_id, ticker_msg_id, f"⏱ {elapsed} · 🔧 {steps} steps...")
        except asyncio.CancelledError:
            pass

    async def _typing_loop(self, chat_id: int) -> None:
        try:
            while True:
                await self._send_typing(chat_id)
                await asyncio.sleep(_TYPING_INTERVAL)
        except asyncio.CancelledError:
            pass


telegram_bot_service = TelegramBotService()

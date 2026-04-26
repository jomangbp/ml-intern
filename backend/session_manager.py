"""Session manager for handling multiple concurrent agent sessions."""

import asyncio
import base64
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from agent.config import load_config
from agent.core.agent_loop import process_submission
from agent.core.session import Event, OpType, Session
from agent.core.tools import ToolRouter

# Get project root (parent of backend directory)
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG_PATH = str(PROJECT_ROOT / "configs" / "frontend_agent_config.json")


# These dataclasses match agent/main.py structure
@dataclass
class Operation:
    """Operation to be executed by the agent."""

    op_type: OpType
    data: Optional[dict[str, Any]] = None


@dataclass
class Submission:
    """Submission to the agent loop."""

    id: str
    operation: Operation


logger = logging.getLogger(__name__)


class EventBroadcaster:
    """Reads from the agent's event queue and fans out to SSE subscribers.

    Events that arrive when no subscribers are listening are discarded.
    With SSE each turn is a separate request, so there is no reconnect
    scenario that would need buffered replay.
    """

    def __init__(self, event_queue: asyncio.Queue):
        self._source = event_queue
        self._subscribers: dict[int, asyncio.Queue] = {}
        self._counter = 0

    def subscribe(self) -> tuple[int, asyncio.Queue]:
        """Create a new subscriber. Returns (id, queue)."""
        self._counter += 1
        sub_id = self._counter
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers[sub_id] = q
        return sub_id, q

    def unsubscribe(self, sub_id: int) -> None:
        self._subscribers.pop(sub_id, None)

    async def run(self) -> None:
        """Main loop — reads from source queue and broadcasts."""
        while True:
            try:
                event: Event = await self._source.get()
                msg = {"event_type": event.event_type, "data": event.data}
                for q in self._subscribers.values():
                    await q.put(msg)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"EventBroadcaster error: {e}")


@dataclass
class AgentSession:
    """Wrapper for an agent session with its associated resources."""

    session_id: str
    session: Session
    tool_router: ToolRouter
    submission_queue: asyncio.Queue
    user_id: str = "dev"  # Owner of this session
    hf_token: str | None = None  # User's HF OAuth token for tool execution
    provider_keys: dict[str, str] = field(default_factory=dict)  # per-user provider API keys
    local_mode: bool = False  # True => local filesystem/bash tools (no sandbox)
    task: asyncio.Task | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True
    is_processing: bool = False  # True while a submission is being executed
    broadcaster: Any = None
    # True once this session has been counted against the user's daily
    # Claude quota. Guards double-counting when the user re-selects an
    # Anthropic model mid-session.
    claude_counted: bool = False


class SessionCapacityError(Exception):
    """Raised when no more sessions can be created."""

    def __init__(self, message: str, error_type: str = "global") -> None:
        super().__init__(message)
        self.error_type = error_type  # "global" or "per_user"


# ── Capacity limits ─────────────────────────────────────────────────
# Sized for HF Spaces 8 vCPU / 32 GB RAM.
# Each session uses ~10-20 MB (context, tools, queues, task); 200 × 20 MB
# = 4 GB worst case, leaving plenty of headroom for the Python runtime
# and per-request overhead.
MAX_SESSIONS: int = 200
MAX_SESSIONS_PER_USER: int = 10


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse boolean env vars like 1/true/yes/on."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class SessionManager:
    """Manages multiple concurrent agent sessions."""

    def __init__(self, config_path: str | None = None) -> None:
        self.config = load_config(config_path or DEFAULT_CONFIG_PATH)
        self.default_local_mode = _env_flag("ML_INTERN_LOCAL_MODE", default=False)
        self.sessions: dict[str, AgentSession] = {}

        # Per-user workspace root for persisted UI settings (.env files).
        self._workspace_root = Path(
            os.environ.get(
                "ML_INTERN_USER_WORKSPACE_ROOT",
                str(Path.home() / ".ml-intern" / "workspaces"),
            )
        )

        # Legacy JSON file from previous implementation (auto-migrated).
        self._legacy_provider_keys_file = Path(
            os.environ.get(
                "ML_INTERN_PROVIDER_KEYS_FILE",
                str(Path.home() / ".ml-intern" / "provider_keys.json"),
            )
        )

        # In-memory cache, hydrated on demand from each user's .env.
        self.user_provider_keys: dict[str, dict[str, str]] = {}
        self._migrate_legacy_provider_keys_file()

        self._lock = asyncio.Lock()

        if self.default_local_mode:
            logger.warning(
                "ML_INTERN_LOCAL_MODE=1 enabled by default: new web sessions get LOCAL "
                "bash/read/write/edit tools with direct access to the host filesystem."
            )

    def _count_user_sessions(self, user_id: str) -> int:
        """Count active sessions owned by a specific user."""
        return sum(
            1
            for s in self.sessions.values()
            if s.user_id == user_id and s.is_active
        )

    @staticmethod
    def _normalize_provider_keys(keys: dict[str, str] | None) -> dict[str, str]:
        """Normalize provider key payload to supported keys only."""
        if not keys:
            return {}
        out: dict[str, str] = {}
        minimax = (keys.get("minimax") or "").strip()
        zai = (keys.get("zai") or "").strip()
        if minimax:
            out["minimax"] = minimax
        if zai:
            out["zai"] = zai
        return out

    @staticmethod
    def _encode_user_id_for_path(user_id: str) -> str:
        """Stable, filesystem-safe directory name for a user id."""
        encoded = base64.urlsafe_b64encode(user_id.encode("utf-8")).decode("ascii")
        return encoded.rstrip("=") or "dev"

    def _workspace_for_user(self, user_id: str) -> Path:
        return self._workspace_root / self._encode_user_id_for_path(user_id)

    def _env_file_for_user(self, user_id: str) -> Path:
        return self._workspace_for_user(user_id) / ".env"

    @staticmethod
    def _parse_env_value(raw: str) -> str:
        value = raw.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            quote = value[0]
            value = value[1:-1]
            if quote == '"':
                value = (
                    value.replace(r"\\", "\\")
                    .replace(r"\"", '"')
                    .replace(r"\n", "\n")
                    .replace(r"\r", "\r")
                )
        return value

    @staticmethod
    def _format_env_value(value: str) -> str:
        safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._:/@")
        if value and all(c in safe_chars for c in value):
            return value
        escaped = value.replace("\\", r"\\").replace('"', r'\"').replace("\n", r"\n").replace("\r", r"\r")
        return f'"{escaped}"'

    def _read_provider_keys_from_env(self, user_id: str) -> dict[str, str]:
        """Read provider keys from user's workspace .env (best-effort)."""
        path = self._env_file_for_user(user_id)
        if not path.exists():
            return {}

        out: dict[str, str] = {}
        try:
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = self._parse_env_value(value)

                if key == "MINIMAX_API_KEY" and value:
                    out["minimax"] = value
                elif key == "ZAI_API_KEY" and value:
                    out["zai"] = value
        except Exception as e:
            logger.warning("Failed reading provider .env for user %s at %s: %s", user_id, path, e)
            return {}

        return self._normalize_provider_keys(out)

    def _write_provider_keys_to_env(self, user_id: str, keys: dict[str, str]) -> None:
        """Persist provider keys to the user's workspace .env."""
        path = self._env_file_for_user(user_id)

        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            lines = [
                "# Auto-generated by ML Intern settings UI",
                "# Per-user provider keys",
            ]
            if keys.get("minimax"):
                lines.append(f"MINIMAX_API_KEY={self._format_env_value(keys['minimax'])}")
            if keys.get("zai"):
                lines.append(f"ZAI_API_KEY={self._format_env_value(keys['zai'])}")

            content = "\n".join(lines) + "\n"

            tmp_path = path.with_suffix(path.suffix + ".tmp")
            tmp_path.write_text(content, encoding="utf-8")
            try:
                os.chmod(tmp_path, 0o600)
            except Exception:
                pass

            tmp_path.replace(path)
            try:
                os.chmod(path, 0o600)
            except Exception:
                pass
        except Exception as e:
            logger.warning("Failed to persist provider keys for user %s to %s: %s", user_id, path, e)

    def _write_provider_keys_to_repo_env(self, keys: dict[str, str]) -> None:
        """Persist provider keys to the repo-root .env file.

        This is the canonical location shared by all sessions (web, Telegram,
        etc.).  Keys written here are available via ``os.environ`` after the
        next server restart, and are read directly by the provider key lookup.
        """
        from pathlib import Path

        # Resolve repo root: backend/..
        repo_env = Path(__file__).parent.parent / ".env"

        try:
            # Read existing .env to preserve other vars
            existing_lines: list[str] = []
            if repo_env.exists():
                existing_lines = repo_env.read_text(encoding="utf-8").splitlines()

            # Build updated .env: keep non-provider lines, replace/add provider lines
            provider_vars = {"MINIMAX_API_KEY", "ZAI_API_KEY"}
            kept: list[str] = []
            for line in existing_lines:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    var_name = stripped.split("=", 1)[0].strip() if "=" in stripped else ""
                    if var_name not in provider_vars:
                        kept.append(line)
                else:
                    kept.append(line)

            # Append provider keys
            if keys.get("minimax"):
                kept.append(f"MINIMAX_API_KEY={keys['minimax']}")
            if keys.get("zai"):
                kept.append(f"ZAI_API_KEY={keys['zai']}")

            content = "\n".join(kept) + "\n"
            tmp = repo_env.with_suffix(".env.tmp")
            tmp.write_text(content, encoding="utf-8")
            tmp.replace(repo_env)
            try:
                repo_env.chmod(0o600)
            except Exception:
                pass

            # Also inject into the current process env so sessions
            # created during this process lifetime see the new keys.
            if keys.get("minimax"):
                os.environ["MINIMAX_API_KEY"] = keys["minimax"]
            if keys.get("zai"):
                os.environ["ZAI_API_KEY"] = keys["zai"]

            logger.info("Provider keys saved to %s", repo_env)
        except Exception as e:
            logger.warning("Failed to persist provider keys to repo .env: %s", e)

    def _migrate_legacy_provider_keys_file(self) -> None:
        """Migrate old JSON provider key storage into per-user workspace .env files."""
        path = self._legacy_provider_keys_file
        if not path.exists():
            return

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return

        if not isinstance(raw, dict):
            return

        migrated = 0
        for user_id, keys in raw.items():
            if not isinstance(user_id, str) or not isinstance(keys, dict):
                continue
            normalized = self._normalize_provider_keys(keys)
            if not normalized:
                continue
            env_path = self._env_file_for_user(user_id)
            if env_path.exists():
                continue
            self._write_provider_keys_to_env(user_id, normalized)
            self.user_provider_keys[user_id] = normalized
            migrated += 1

        if migrated:
            logger.info(
                "Migrated provider keys for %d user(s) from legacy file %s",
                migrated,
                path,
            )

    def get_user_provider_keys(self, user_id: str) -> dict[str, str]:
        """Get a user's provider keys, loading from user workspace .env if needed."""
        if user_id not in self.user_provider_keys:
            self.user_provider_keys[user_id] = self._read_provider_keys_from_env(user_id)
        return self.user_provider_keys.get(user_id, {}).copy()

    def get_user_workspace_env_file(self, user_id: str) -> str:
        """Return the .env path used for this user's persisted provider keys."""
        return str(self._env_file_for_user(user_id))

    def get_effective_provider_keys(self, user_id: str) -> dict[str, str]:
        """Get provider keys with env fallback.

        Priority:
          1) user-scoped keys set via /auth/providers/tokens (.env in user workspace)
          2) process env (MINIMAX_API_KEY / ZAI_API_KEY)
        """
        keys = self.get_user_provider_keys(user_id)
        if not keys.get("minimax"):
            env_minimax = (os.environ.get("MINIMAX_API_KEY") or "").strip()
            if env_minimax:
                keys["minimax"] = env_minimax
        if not keys.get("zai"):
            env_zai = (os.environ.get("ZAI_API_KEY") or "").strip()
            if env_zai:
                keys["zai"] = env_zai
        return keys

    def set_user_provider_keys(self, user_id: str, keys: dict[str, str] | None) -> dict[str, str]:
        """Set provider keys for a user and propagate to active sessions.

        Keys are persisted to the repo-root .env so all sessions (web,
        Telegram, etc.) can access them via environment variables.
        """
        normalized = self._normalize_provider_keys(keys)

        # Keep cache in sync
        self.user_provider_keys[user_id] = normalized

        # Persist to repo-root .env
        self._write_provider_keys_to_repo_env(normalized)

        # Also write to workspace for backwards compat
        self._write_provider_keys_to_env(user_id, normalized)

        # Propagate to ALL active sessions (not just same user)
        # since repo .env is shared across all sessions.
        for s in self.sessions.values():
            s.provider_keys = normalized.copy()
            s.session.provider_keys = normalized.copy()

        return normalized.copy()

    async def create_session(
        self,
        user_id: str = "dev",
        hf_token: str | None = None,
        model: str | None = None,
        local_mode: bool | None = None,
        provider_keys: dict[str, str] | None = None,
    ) -> str:
        """Create a new agent session and return its ID.

        Session() and ToolRouter() constructors contain blocking I/O
        (e.g. HfApi().whoami(), litellm.get_max_tokens()) so they are
        executed in a thread pool to avoid freezing the async event loop.

        Args:
            user_id: The ID of the user who owns this session.
            hf_token: The user's HF OAuth token, stored for tool execution.
            model: Optional model override. When set, replaces ``model_name``
                on the per-session config clone. None falls back to the
                config default.
            local_mode: Override execution mode for this session.
                True => local host tools (no sandbox_create),
                False => sandbox tools, None => use server default.

        Raises:
            SessionCapacityError: If the server or user has reached the
                maximum number of concurrent sessions.
        """
        # ── Capacity checks ──────────────────────────────────────────
        async with self._lock:
            active_count = self.active_session_count
            if active_count >= MAX_SESSIONS:
                raise SessionCapacityError(
                    f"Server is at capacity ({active_count}/{MAX_SESSIONS} sessions). "
                    "Please try again later.",
                    error_type="global",
                )
            if user_id != "dev":
                user_count = self._count_user_sessions(user_id)
                if user_count >= MAX_SESSIONS_PER_USER:
                    raise SessionCapacityError(
                        f"You have reached the maximum of {MAX_SESSIONS_PER_USER} "
                        "concurrent sessions. Please close an existing session first.",
                        error_type="per_user",
                    )

        session_local_mode = (
            self.default_local_mode if local_mode is None else bool(local_mode)
        )

        session_id = str(uuid.uuid4())

        # Create queues for this session
        submission_queue: asyncio.Queue = asyncio.Queue()
        event_queue: asyncio.Queue = asyncio.Queue()

        # Run blocking constructors in a thread to keep the event loop responsive.
        # Without this, Session.__init__ → ContextManager → litellm.get_max_tokens()
        # blocks all HTTP/SSE handling.
        import time as _time

        effective_provider_keys = provider_keys or self.user_provider_keys.get(user_id, {}).copy()

        def _create_session_sync():
            t0 = _time.monotonic()
            tool_router = ToolRouter(
                self.config.mcpServers,
                hf_token=hf_token,
                local_mode=session_local_mode,
            )
            # Deep-copy config so each session's model switches independently —
            # tab A picking GLM doesn't flip tab B off Claude.
            session_config = self.config.model_copy(deep=True)
            if model:
                session_config.model_name = model
            session = Session(
                event_queue,
                config=session_config,
                tool_router=tool_router,
                hf_token=hf_token,
                provider_keys=effective_provider_keys,
                local_mode=session_local_mode,
            )
            t1 = _time.monotonic()
            logger.info(f"Session initialized in {t1 - t0:.2f}s")
            return tool_router, session

        tool_router, session = await asyncio.to_thread(_create_session_sync)

        # Create wrapper
        agent_session = AgentSession(
            session_id=session_id,
            session=session,
            tool_router=tool_router,
            submission_queue=submission_queue,
            user_id=user_id,
            hf_token=hf_token,
            provider_keys=effective_provider_keys,
            local_mode=session_local_mode,
        )

        async with self._lock:
            self.sessions[session_id] = agent_session

        # Start the agent loop task
        task = asyncio.create_task(
            self._run_session(session_id, submission_queue, event_queue, tool_router)
        )
        agent_session.task = task

        mode = "local" if session_local_mode else "sandbox"
        logger.info(f"Created session {session_id} for user {user_id} (mode={mode})")
        return session_id

    async def seed_from_summary(self, session_id: str, messages: list[dict]) -> int:
        """Rehydrate a session from cached prior messages via summarization.

        Runs the standard summarization prompt (same one compaction uses)
        over the provided messages, then seeds the new session's context
        with that summary. Tool-call pairing concerns disappear because the
        output is plain text. Returns the number of messages summarized.
        """
        from litellm import Message

        from agent.context_manager.manager import _RESTORE_PROMPT, summarize_messages

        agent_session = self.sessions.get(session_id)
        if not agent_session:
            raise ValueError(f"Session {session_id} not found")

        # Parse into Message objects, tolerating malformed entries.
        parsed: list[Message] = []
        for raw in messages:
            if raw.get("role") == "system":
                continue  # the new session has its own system prompt
            try:
                parsed.append(Message.model_validate(raw))
            except Exception as e:
                logger.warning("Dropping malformed message during seed: %s", e)

        if not parsed:
            return 0

        session = agent_session.session
        # Pass the real tool specs so the summarizer sees what the agent
        # actually has — otherwise Anthropic's modify_params injects a
        # dummy tool and the summarizer editorializes that the original
        # tool calls were fabricated.
        tool_specs = None
        try:
            tool_specs = agent_session.tool_router.get_tool_specs_for_llm()
        except Exception:
            pass
        try:
            summary, _ = await summarize_messages(
                parsed,
                model_name=session.config.model_name,
                hf_token=session.hf_token,
                max_tokens=4000,
                prompt=_RESTORE_PROMPT,
                tool_specs=tool_specs,
                provider_keys=getattr(session, "provider_keys", None),
            )
        except Exception as e:
            logger.error("Summary call failed during seed: %s", e)
            raise

        seed = Message(
            role="user",
            content=(
                "[SYSTEM: Your prior memory of this conversation — written "
                "in your own voice right before restart. Continue from here.]\n\n"
                + (summary or "(no summary returned)")
            ),
        )
        session.context_manager.items.append(seed)
        return len(parsed)

    @staticmethod
    async def _cleanup_sandbox(session: Session) -> None:
        """Delete the sandbox Space if one was created for this session."""
        sandbox = getattr(session, "sandbox", None)
        if sandbox and getattr(sandbox, "_owns_space", False):
            space_id = getattr(sandbox, "space_id", None)
            try:
                logger.info(f"Deleting sandbox {space_id}...")
                await asyncio.to_thread(sandbox.delete)
                from agent.core import telemetry
                await telemetry.record_sandbox_destroy(session, sandbox)
            except Exception as e:
                logger.warning(f"Failed to delete sandbox {space_id}: {e}")

    async def _run_session(
        self,
        session_id: str,
        submission_queue: asyncio.Queue,
        event_queue: asyncio.Queue,
        tool_router: ToolRouter,
    ) -> None:
        """Run the agent loop for a session and broadcast events via EventBroadcaster."""
        agent_session = self.sessions.get(session_id)
        if not agent_session:
            logger.error(f"Session {session_id} not found")
            return

        session = agent_session.session

        # Start event broadcaster task
        broadcaster = EventBroadcaster(event_queue)
        agent_session.broadcaster = broadcaster
        broadcast_task = asyncio.create_task(broadcaster.run())

        try:
            async with tool_router:
                # Send ready event
                await session.send_event(
                    Event(event_type="ready", data={"message": "Agent initialized"})
                )

                while session.is_running:
                    try:
                        # Wait for submission with timeout to allow checking is_running
                        submission = await asyncio.wait_for(
                            submission_queue.get(), timeout=1.0
                        )
                        agent_session.is_processing = True
                        try:
                            should_continue = await process_submission(session, submission)
                        finally:
                            agent_session.is_processing = False
                        if not should_continue:
                            break
                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        logger.info(f"Session {session_id} cancelled")
                        break
                    except Exception as e:
                        logger.error(f"Error in session {session_id}: {e}")
                        await session.send_event(
                            Event(event_type="error", data={"error": str(e)})
                        )

        finally:
            broadcast_task.cancel()
            try:
                await broadcast_task
            except asyncio.CancelledError:
                pass

            await self._cleanup_sandbox(session)

            # Final-flush: always save on session death so we capture ended
            # sessions even if the client disconnects without /shutdown.
            # Idempotent via session_id key; detached subprocess.
            if session.config.save_sessions:
                try:
                    session.save_and_upload_detached(session.config.session_dataset_repo)
                except Exception as e:
                    logger.warning(f"Final-flush failed for {session_id}: {e}")

            async with self._lock:
                if session_id in self.sessions:
                    self.sessions[session_id].is_active = False

            logger.info(f"Session {session_id} ended")

    async def submit(self, session_id: str, operation: Operation) -> bool:
        """Submit an operation to a session."""
        async with self._lock:
            agent_session = self.sessions.get(session_id)

        if not agent_session or not agent_session.is_active:
            logger.warning(f"Session {session_id} not found or inactive")
            return False

        submission = Submission(id=f"sub_{uuid.uuid4().hex[:8]}", operation=operation)
        await agent_session.submission_queue.put(submission)
        return True

    async def submit_user_input(self, session_id: str, text: str) -> bool:
        """Submit user input to a session."""
        operation = Operation(op_type=OpType.USER_INPUT, data={"text": text})
        return await self.submit(session_id, operation)

    async def submit_approval(
        self, session_id: str, approvals: list[dict[str, Any]]
    ) -> bool:
        """Submit tool approvals to a session."""
        operation = Operation(
            op_type=OpType.EXEC_APPROVAL, data={"approvals": approvals}
        )
        return await self.submit(session_id, operation)

    async def interrupt(self, session_id: str) -> bool:
        """Interrupt a session by signalling cancellation directly (bypasses queue)."""
        agent_session = self.sessions.get(session_id)
        if not agent_session or not agent_session.is_active:
            return False
        agent_session.session.cancel()
        return True

    async def undo(self, session_id: str) -> bool:
        """Undo last turn in a session."""
        operation = Operation(op_type=OpType.UNDO)
        return await self.submit(session_id, operation)

    async def truncate(self, session_id: str, user_message_index: int) -> bool:
        """Truncate conversation to before a specific user message (direct, no queue)."""
        async with self._lock:
            agent_session = self.sessions.get(session_id)
        if not agent_session or not agent_session.is_active:
            return False
        return agent_session.session.context_manager.truncate_to_user_message(user_message_index)

    async def compact(self, session_id: str) -> bool:
        """Compact context in a session."""
        operation = Operation(op_type=OpType.COMPACT)
        return await self.submit(session_id, operation)

    async def shutdown_session(self, session_id: str) -> bool:
        """Shutdown a specific session."""
        operation = Operation(op_type=OpType.SHUTDOWN)
        success = await self.submit(session_id, operation)

        if success:
            async with self._lock:
                agent_session = self.sessions.get(session_id)
                if agent_session and agent_session.task:
                    # Wait for task to complete
                    try:
                        await asyncio.wait_for(agent_session.task, timeout=5.0)
                    except asyncio.TimeoutError:
                        agent_session.task.cancel()

        return success

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session entirely."""
        async with self._lock:
            agent_session = self.sessions.pop(session_id, None)

        if not agent_session:
            return False

        # Clean up sandbox Space before cancelling the task
        await self._cleanup_sandbox(agent_session.session)

        # Cancel the task if running
        if agent_session.task and not agent_session.task.done():
            agent_session.task.cancel()
            try:
                await agent_session.task
            except asyncio.CancelledError:
                pass

        return True

    def get_session_owner(self, session_id: str) -> str | None:
        """Get the user_id that owns a session, or None if session doesn't exist."""
        agent_session = self.sessions.get(session_id)
        if not agent_session:
            return None
        return agent_session.user_id

    def verify_session_access(self, session_id: str, user_id: str) -> bool:
        """Check if a user has access to a session.

        Returns True if:
        - The session exists AND the user owns it
        - The user_id is "dev" (dev mode bypass)
        """
        owner = self.get_session_owner(session_id)
        if owner is None:
            return False
        if user_id == "dev" or owner == "dev":
            return True
        return owner == user_id

    def get_session_info(self, session_id: str) -> dict[str, Any] | None:
        """Get information about a session."""
        agent_session = self.sessions.get(session_id)
        if not agent_session:
            return None

        # Extract pending approval tools if any
        pending_approval = None
        pa = agent_session.session.pending_approval
        if pa and pa.get("tool_calls"):
            pending_approval = []
            for tc in pa["tool_calls"]:
                import json
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, AttributeError):
                    args = {}
                pending_approval.append({
                    "tool": tc.function.name,
                    "tool_call_id": tc.id,
                    "arguments": args,
                })

        return {
            "session_id": session_id,
            "created_at": agent_session.created_at.isoformat(),
            "is_active": agent_session.is_active,
            "is_processing": agent_session.is_processing,
            "message_count": len(agent_session.session.context_manager.items),
            "user_id": agent_session.user_id,
            "pending_approval": pending_approval,
            "model": agent_session.session.config.model_name,
            "execution_mode": "local" if agent_session.local_mode else "sandbox",
        }

    def list_sessions(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """List sessions, optionally filtered by user.

        Args:
            user_id: If provided, only return sessions owned by this user.
                     If "dev", return all sessions (dev mode).
        """
        results = []
        for sid in self.sessions:
            info = self.get_session_info(sid)
            if not info:
                continue
            if user_id and user_id != "dev" and info.get("user_id") != user_id:
                continue
            results.append(info)
        return results

    @property
    def active_session_count(self) -> int:
        """Get count of active sessions."""
        return sum(1 for s in self.sessions.values() if s.is_active)


# Global session manager instance
session_manager = SessionManager()

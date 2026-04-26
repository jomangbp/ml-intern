"""Approval store — persistent approval tracking for gateway.

When the agent emits an approval_required event, the Telegram gateway
creates an approval record and sends inline buttons. Approvals persist
to disk so they survive gateway restarts.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from events.event_store import event_store

APPROVAL_DIR = Path(os.environ.get(
    "ML_INTERN_APPROVAL_DIR",
    str(Path.home() / ".cache" / "ml-intern" / "approvals"),
))
DEFAULT_EXPIRY_SECONDS = 600  # 10 minutes


def _new_approval_id() -> str:
    return f"appr_{uuid.uuid4().hex[:10]}"


def _persist_approval(approval: dict[str, Any]) -> None:
    """Save approval state to disk."""
    try:
        APPROVAL_DIR.mkdir(parents=True, exist_ok=True)
        path = APPROVAL_DIR / f"{approval['approval_id']}.json"
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(approval, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        pass


def _delete_approval_file(approval_id: str) -> None:
    try:
        (APPROVAL_DIR / f"{approval_id}.json").unlink(missing_ok=True)
    except Exception:
        pass


class ApprovalRecord:
    """A single approval request."""

    def __init__(
        self,
        approval_id: str,
        session_id: str,
        tools: list[dict[str, Any]],
        *,
        platform: str = "",
        chat_id: str | int | None = None,
        identity_id: str = "",
        expires_at: float | None = None,
    ) -> None:
        self.approval_id = approval_id
        self.session_id = session_id
        self.tools = tools
        self.platform = platform
        self.chat_id = chat_id
        self.identity_id = identity_id
        self.status: str = "pending"  # pending, approved, rejected, expired
        self.created_at = time.time()
        self.resolved_at: float | None = None
        self.expires_at = expires_at or (self.created_at + DEFAULT_EXPIRY_SECONDS)
        self.message_id: int | None = None  # Telegram message ID for editing

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    @property
    def summary(self) -> str:
        """One-line summary of what's being approved."""
        parts = []
        for t in self.tools:
            name = t.get("tool", "?")
            args = t.get("arguments", {})
            if name == "bash":
                cmd = str(args.get("command", ""))[:60]
                parts.append(f"💻 bash: `{cmd}`")
            elif name in ("write_file", "edit_file"):
                path = str(args.get("path", ""))[:60]
                parts.append(f"✏️ {name}: `{path}`")
            elif name == "local_training":
                script = str(args.get("script", ""))[:60]
                parts.append(f"🏋️ train: `{script}`")
            else:
                parts.append(f"🔧 {name}")
        return "\n".join(parts)

    @property
    def details(self) -> str:
        """Full details for the approval."""
        lines = []
        for t in self.tools:
            name = t.get("tool", "?")
            args = t.get("arguments", {})
            tc_id = t.get("tool_call_id", "?")
            lines.append(f"Tool: {name} (id: {tc_id[:12]}...)")
            for k, v in list(args.items())[:5]:
                v_str = str(v)
                if len(v_str) > 200:
                    v_str = v_str[:200] + "..."
                lines.append(f"  {k}: {v_str}")
            lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "session_id": self.session_id,
            "tools": self.tools,
            "platform": self.platform,
            "chat_id": self.chat_id,
            "identity_id": self.identity_id,
            "status": self.status,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
            "expires_at": self.expires_at,
            "message_id": self.message_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ApprovalRecord:
        record = cls(
            approval_id=data["approval_id"],
            session_id=data["session_id"],
            tools=data["tools"],
            platform=data.get("platform", ""),
            chat_id=data.get("chat_id"),
            identity_id=data.get("identity_id", ""),
            expires_at=data.get("expires_at"),
        )
        record.status = data.get("status", "pending")
        record.created_at = data.get("created_at", time.time())
        record.resolved_at = data.get("resolved_at")
        record.message_id = data.get("message_id")
        return record


class ApprovalStore:
    """Manages approval lifecycle with persistence."""

    def __init__(self) -> None:
        self._pending: dict[str, ApprovalRecord] = {}

    def create(
        self,
        *,
        session_id: str,
        tools: list[dict[str, Any]],
        platform: str = "",
        chat_id: str | int | None = None,
        identity_id: str = "",
        expiry_seconds: int = DEFAULT_EXPIRY_SECONDS,
    ) -> ApprovalRecord:
        """Create a new approval request."""
        record = ApprovalRecord(
            approval_id=_new_approval_id(),
            session_id=session_id,
            tools=tools,
            platform=platform,
            chat_id=chat_id,
            identity_id=identity_id,
            expires_at=time.time() + expiry_seconds,
        )
        self._pending[record.approval_id] = record
        _persist_approval(record.to_dict())

        event_store.log(
            "approval.required",
            source="approval_store",
            session_id=session_id,
            platform=platform,
            chat_id=chat_id,
            payload={
                "approval_id": record.approval_id,
                "tools": [t.get("tool") for t in tools],
                "expires_at": record.expires_at,
            },
        )
        return record

    def get(self, approval_id: str) -> ApprovalRecord | None:
        return self._pending.get(approval_id)

    def set_message_id(self, approval_id: str, message_id: int) -> None:
        record = self._pending.get(approval_id)
        if record:
            record.message_id = message_id
            _persist_approval(record.to_dict())

    async def approve(
        self, approval_id: str, tool_edits: dict[str, str] | None = None,
    ) -> ApprovalRecord | None:
        """Approve a pending request."""
        record = self._pending.get(approval_id)
        if not record:
            return None
        if record.is_expired:
            record.status = "expired"
            _persist_approval(record.to_dict())
            return record
        if record.status != "pending":
            return record

        from session_manager import session_manager

        approvals = []
        for t in record.tools:
            entry: dict[str, Any] = {
                "tool_call_id": t.get("tool_call_id", ""),
                "approved": True,
            }
            if tool_edits and t.get("tool_call_id") in tool_edits:
                entry["edited_script"] = tool_edits[t["tool_call_id"]]
            approvals.append(entry)

        ok = await session_manager.submit_approval(record.session_id, approvals)

        record.status = "approved" if ok else "failed"
        record.resolved_at = time.time()
        _persist_approval(record.to_dict())

        event_store.log(
            "approval.approved",
            source="approval_store",
            session_id=record.session_id,
            platform=record.platform,
            chat_id=record.chat_id,
            payload={"approval_id": approval_id, "submit_ok": ok},
        )
        return record

    async def reject(self, approval_id: str) -> ApprovalRecord | None:
        """Reject a pending request."""
        record = self._pending.get(approval_id)
        if not record:
            return None
        if record.status != "pending":
            return record

        from session_manager import session_manager

        rejections = [
            {"tool_call_id": t.get("tool_call_id", ""), "approved": False}
            for t in record.tools
        ]

        ok = await session_manager.submit_approval(record.session_id, rejections)

        record.status = "rejected"
        record.resolved_at = time.time()
        _persist_approval(record.to_dict())

        event_store.log(
            "approval.rejected",
            source="approval_store",
            session_id=record.session_id,
            platform=record.platform,
            chat_id=record.chat_id,
            payload={"approval_id": approval_id, "submit_ok": ok},
        )
        return record

    def list_pending(self, *, platform: str = "", chat_id: str | int | None = None) -> list[ApprovalRecord]:
        """List pending approvals, optionally filtered."""
        results = []
        for record in self._pending.values():
            if record.status != "pending":
                continue
            if record.is_expired:
                record.status = "expired"
                _persist_approval(record.to_dict())
                continue
            if platform and record.platform != platform:
                continue
            if chat_id is not None and str(record.chat_id) != str(chat_id):
                continue
            results.append(record)
        return results

    def cleanup_expired(self) -> int:
        """Remove expired approvals. Returns count cleaned."""
        cleaned = 0
        for approval_id in list(self._pending.keys()):
            record = self._pending[approval_id]
            if record.is_expired or record.status in ("approved", "rejected", "expired"):
                if record.status == "pending" and record.is_expired:
                    record.status = "expired"
                    _persist_approval(record.to_dict())
                    event_store.log(
                        "approval.expired",
                        source="approval_store",
                        session_id=record.session_id,
                        payload={"approval_id": approval_id},
                    )
                # Keep in memory for a while, just mark
                cleaned += 1
        return cleaned

    def restore(self) -> int:
        """Restore pending approvals from disk."""
        restored = 0
        if not APPROVAL_DIR.exists():
            return restored
        for path in APPROVAL_DIR.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                record = ApprovalRecord.from_dict(data)
                if record.status == "pending" and not record.is_expired:
                    self._pending[record.approval_id] = record
                    restored += 1
                elif record.status == "pending" and record.is_expired:
                    record.status = "expired"
                    _persist_approval(record.to_dict())
            except Exception:
                pass
        return restored


# Singleton
approval_store = ApprovalStore()

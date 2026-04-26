"""Gateway adapter base — common interface for all platform adapters.

Each adapter (Telegram, CLI, Web, Discord, etc.) must implement this protocol.
The adapter should be transport-only: receive platform events, convert to
internal commands, and render responses back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class GatewayCommand:
    """Platform-neutral command passed to the command router."""

    def __init__(
        self,
        *,
        source: str,
        command: str,
        args: list[str] = (),
        raw_text: str = "",
        chat_id: str | int | None = None,
        user_id: str | int | None = None,
        identity_id: str = "",
        session_id: str = "",
        platform: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.source = source
        self.command = command
        self.args = list(args)
        self.raw_text = raw_text
        self.chat_id = chat_id
        self.user_id = user_id
        self.identity_id = identity_id
        self.session_id = session_id
        self.platform = platform or source
        self.extra = extra or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "command": self.command,
            "args": self.args,
            "raw_text": self.raw_text,
            "chat_id": self.chat_id,
            "user_id": self.user_id,
            "identity_id": self.identity_id,
            "session_id": self.session_id,
            "platform": self.platform,
        }


@dataclass
class GatewayTarget:
    """Where to send a response."""
    platform: str
    chat_id: str | int | None = None
    thread_id: str | int | None = None
    user_id: str | int | None = None


@dataclass
class GatewayMessage:
    """A message to send to a target."""
    text: str
    parse_mode: str | None = None
    reply_markup: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ApprovalRequest:
    """An approval request to send to a target."""
    approval_id: str
    session_id: str
    tool_name: str
    summary: str
    details: str
    status: str = "pending"  # pending, approved, rejected, expired


@dataclass
class GatewayEvent:
    """A gateway event."""
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class GatewayAdapter(Protocol):
    """Protocol that all platform adapters must implement."""

    name: str

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def send_message(self, target: GatewayTarget, message: GatewayMessage) -> int | None: ...
    async def send_approval(self, target: GatewayTarget, approval: ApprovalRequest) -> int | None: ...
    async def send_event(self, target: GatewayTarget, event: GatewayEvent) -> None: ...

"""Gateway command router — shared command parsing and dispatch.

All platform adapters convert their input into GatewayCommand objects,
then pass them here. The router handles auth, session resolution,
and dispatches to the appropriate handler.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Awaitable

from gateway.adapter_base import GatewayCommand, GatewayTarget, GatewayMessage
from gateway.identity import identity_manager
from events.event_store import event_store

logger = logging.getLogger(__name__)

# Type for command handlers
CommandHandler = Callable[[GatewayCommand], Awaitable[dict[str, Any] | None]]


class CommandRouter:
    """Shared command router for all platform adapters."""

    def __init__(self) -> None:
        self._handlers: dict[str, CommandHandler] = {}
        self._fallback_handler: CommandHandler | None = None

    def register(self, command: str, handler: CommandHandler) -> None:
        """Register a handler for a command name."""
        self._handlers[command] = handler

    def set_fallback(self, handler: CommandHandler) -> None:
        """Set handler for unrecognized commands (e.g. send to agent)."""
        self._fallback_handler = handler

    async def dispatch(self, cmd: GatewayCommand) -> dict[str, Any] | None:
        """Route a command through auth + handler.

        Returns a response dict or None.
        """
        # Log the command
        event_store.log(
            f"gateway.command.received",
            source=cmd.source,
            platform=cmd.platform,
            session_id=cmd.session_id,
            identity_id=cmd.identity_id,
            chat_id=cmd.chat_id,
            payload={
                "command": cmd.command,
                "args": cmd.args,
                "raw_text": cmd.raw_text,
            },
        )

        # Auth check
        allowed, identity = identity_manager.check_command_permission(
            cmd.platform, cmd.user_id or 0, cmd.command,
        )

        if not allowed:
            event_store.log(
                "gateway.unauthorized",
                source=cmd.source,
                platform=cmd.platform,
                identity_id=cmd.identity_id,
                chat_id=cmd.chat_id,
                payload={"command": cmd.command, "user_id": cmd.user_id},
            )
            return {
                "response": "⛔ You don't have permission for this command.",
                "parse_mode": None,
            }

        # Find handler
        handler = self._handlers.get(cmd.command)
        if handler:
            try:
                result = await handler(cmd)
                event_store.log(
                    "gateway.command.completed",
                    source=cmd.source,
                    platform=cmd.platform,
                    session_id=cmd.session_id,
                    payload={"command": cmd.command},
                )
                return result
            except Exception as e:
                logger.exception("Command handler failed for %s", cmd.command)
                event_store.log(
                    "gateway.command.failed",
                    source=cmd.source,
                    platform=cmd.platform,
                    session_id=cmd.session_id,
                    payload={"command": cmd.command, "error": str(e)},
                )
                return {
                    "response": f"❌ Error executing /{cmd.command}: {e}",
                    "parse_mode": None,
                }

        # Fallback — send to agent
        if self._fallback_handler:
            try:
                return await self._fallback_handler(cmd)
            except Exception as e:
                logger.exception("Fallback handler failed")
                return {
                    "response": f"❌ Error: {e}",
                    "parse_mode": None,
                }

        return {
            "response": f"Unknown command: /{cmd.command}",
            "parse_mode": None,
        }


# Singleton
command_router = CommandRouter()

"""Gateway health — aggregated status of all gateway subsystems."""

from __future__ import annotations

from typing import Any


def gateway_health(
    *,
    telegram_running: bool = False,
    telegram_enabled: bool = False,
    active_sessions: int = 0,
    active_crons: int = 0,
    running_jobs: int = 0,
    event_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute gateway health status."""
    return {
        "gateway": "online",
        "telegram": "connected" if telegram_running else "disabled" if not telegram_enabled else "error",
        "cron_runner": "running" if active_crons >= 0 else "stopped",
        "active_sessions": active_sessions,
        "active_crons": active_crons,
        "running_jobs": running_jobs,
        "event_stats": event_stats or {},
    }


def format_health_telegram(health: dict[str, Any]) -> str:
    """Format health for Telegram display."""
    lines = [
        f"📡 *Gateway:* {health['gateway']}",
        f"🤖 *Telegram:* {health['telegram']}",
        f"⏰ *Crons:* {health['active_crons']} active",
        f"📂 *Sessions:* {health['active_sessions']} active",
        f"🔧 *Jobs:* {health['running_jobs']} running",
    ]
    stats = health.get("event_stats", {})
    if stats.get("total_events"):
        lines.append(f"📊 *Events:* {stats['total_events']} logged")
    return "\n".join(lines)

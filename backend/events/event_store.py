"""Gateway event store — persistent append-only log for all gateway events.

Stores events as JSONL in ~/.cache/ml-intern/events/events.jsonl
Event logging failure never crashes the calling code.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

EVENT_STORE_DIR = Path(os.environ.get(
    "ML_INTERN_EVENTS_DIR",
    str(Path.home() / ".cache" / "ml-intern" / "events"),
))


def _new_event_id() -> str:
    return f"evt_{uuid.uuid4().hex[:12]}"


class EventStore:
    """Append-only persistent event store.

    Usage:
        store = EventStore()
        store.log("gateway.started", source="gateway")
        store.log("telegram.command.received", source="telegram", payload={"command": "/status"})
        recent = store.tail(50)
    """

    def __init__(self, path: Path | str | None = None) -> None:
        if path is None:
            EVENT_STORE_DIR.mkdir(parents=True, exist_ok=True)
            path = EVENT_STORE_DIR / "events.jsonl"
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def log(
        self,
        event_type: str,
        *,
        source: str = "",
        session_id: str = "",
        identity_id: str = "",
        platform: str = "",
        chat_id: str | int | None = None,
        task_id: str = "",
        job_id: str = "",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append an event to the store. Never raises."""
        event = {
            "event_id": _new_event_id(),
            "type": event_type,
            "source": source,
            "platform": platform,
            "session_id": session_id,
            "identity_id": identity_id,
            "task_id": task_id,
            "job_id": job_id,
            "timestamp": time.time(),
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "payload": payload or {},
        }
        if chat_id is not None:
            event["chat_id"] = str(chat_id)

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception:
            # Event logging failure must never crash the caller
            pass
        return event

    def tail(self, limit: int = 100, event_type: str | None = None) -> list[dict[str, Any]]:
        """Read last N events, optionally filtered by type."""
        events: list[dict[str, Any]] = []
        if not self._path.exists():
            return events
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        evt = json.loads(line)
                        if event_type and evt.get("type") != event_type:
                            continue
                        events.append(evt)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
        return events[-limit:]

    def query(
        self,
        *,
        source: str | None = None,
        platform: str | None = None,
        session_id: str | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query events with filters."""
        results: list[dict[str, Any]] = []
        for evt in self.tail(limit=10000):
            if source and evt.get("source") != source:
                continue
            if platform and evt.get("platform") != platform:
                continue
            if session_id and evt.get("session_id") != session_id:
                continue
            if event_type and evt.get("type") != event_type:
                continue
            results.append(evt)
        return results[-limit:]

    def stats(self) -> dict[str, Any]:
        """Return basic stats about the event store."""
        total = 0
        types: dict[str, int] = {}
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            evt = json.loads(line)
                            total += 1
                            t = evt.get("type", "unknown")
                            types[t] = types.get(t, 0) + 1
                        except json.JSONDecodeError:
                            total += 1
            except Exception:
                pass
        return {
            "total_events": total,
            "event_types": types,
            "path": str(self._path),
        }


# Singleton
event_store = EventStore()

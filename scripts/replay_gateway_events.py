#!/usr/bin/env python3
"""Replay gateway events from the JSONL event store.

Usage:
    python scripts/replay_gateway_events.py
    python scripts/replay_gateway_events.py --limit 50
    python scripts/replay_gateway_events.py --type gateway.unauthorized
    python scripts/replay_gateway_events.py --source telegram
"""

import argparse
import json
import sys
from pathlib import Path

EVENT_STORE_PATH = Path.home() / ".cache" / "ml-intern" / "events" / "events.jsonl"


def main():
    parser = argparse.ArgumentParser(description="Replay gateway events")
    parser.add_argument("--path", default=str(EVENT_STORE_PATH), help="Path to events.jsonl")
    parser.add_argument("--limit", type=int, default=50, help="Number of events to show")
    parser.add_argument("--type", default=None, help="Filter by event type")
    parser.add_argument("--source", default=None, help="Filter by source")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"No event store found at {path}")
        sys.exit(1)

    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
                if args.type and evt.get("type") != args.type:
                    continue
                if args.source and evt.get("source") != args.source:
                    continue
                events.append(evt)
            except json.JSONDecodeError:
                continue

    events = events[-args.limit:]

    if not events:
        print("No matching events found.")
        return

    if args.json:
        for evt in events:
            print(json.dumps(evt, default=str))
        return

    for evt in events:
        ts = evt.get("timestamp_iso", evt.get("timestamp", "?"))
        etype = evt.get("type", "?")
        source = evt.get("source", "")
        payload = evt.get("payload", {})
        chat_id = evt.get("chat_id", "")
        session = evt.get("session_id", "")

        extra = ""
        if chat_id:
            extra += f" chat={chat_id}"
        if session:
            extra += f" sess={session[:8]}"
        if payload:
            preview = json.dumps(payload, default=str)[:80]
            extra += f" {preview}"

        print(f"[{ts}] {source:>12} {etype}{extra}")

    print(f"\nShowing {len(events)} events (total in file: see --json)")


if __name__ == "__main__":
    main()

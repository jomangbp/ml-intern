#!/usr/bin/env python3
"""Smoke-check that custom tools are registered."""

from agent.core.tools import create_builtin_tools

required = {"codex_login", "hf_catalog"}
tools = {t.name for t in create_builtin_tools(local_mode=True)}
missing = sorted(required - tools)

if missing:
    raise SystemExit(f"Missing required tools: {', '.join(missing)}")

print("Tool registry OK")

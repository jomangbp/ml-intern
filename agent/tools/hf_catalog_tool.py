"""
HF Model Catalog Tool — inspect HF router catalog and resolve model info.

Lets the agent (or user) discover which models are live on the HuggingFace
Inference Router, check pricing, context length, and tool-call support,
and resolve any HF model id into LiteLLM kwargs without hard-coding.

This enables dynamic model selection — the agent can say "use the cheapest
live model with tool support" and actually do it.
"""

from typing import Any

import httpx

from agent.tools.types import ToolResult

_CATALOG_URL = "https://router.huggingface.co/v1/models"
_HTTP_TIMEOUT = 8.0


async def hf_catalog_handler(arguments: dict[str, Any]) -> tuple[str, bool]:
    """Query the HF Inference Router catalog.

    Args:
        model_id   (str, optional): Narrow to a specific model org/name.
            Supports fuzzy partial match. If omitted, returns all live models.
        provider   (str, optional): Filter to a specific provider name
            (e.g., "novita", "cerebras", "mistralai").
        tools      (bool, optional): If True, only return models that advertise
            tool-call support. Default: False.
        limit      (int, optional): Max results to return. Default: 20.
        show_price (bool, optional): Include input/output price info.
            Default: True.

    Returns:
        ToolResult with a formatted table of models and their properties.
    """
    model_id = arguments.get("model_id", "")
    provider = arguments.get("provider", "")
    tools_only = bool(arguments.get("tools", False))
    limit = min(arguments.get("limit", 20), 100)
    show_price = arguments.get("show_price", True)

    try:
        resp = httpx.get(_CATALOG_URL, timeout=_HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except httpx.TimeoutException:
        return "HF Router catalog timed out. Check your connection and retry.", False
    except httpx.HTTPStatusError as e:
        return f"HF Router catalog error ({e.response.status_code}): {e}", False
    except Exception as e:
        return f"Failed to fetch HF Router catalog: {e}", False

    entries = data.get("data", [])
    if not entries:
        return "HF Router catalog returned no models.", True

    # Filter
    filtered = []
    for entry in entries:
        entry_id = entry.get("id", "")

        # model_id partial match
        if model_id and model_id.lower() not in entry_id.lower():
            continue

        # Provider filter
        providers = entry.get("providers", []) or []
        live_provs = [p for p in providers if p.get("status") == "live"]
        if provider:
            live_provs = [p for p in live_provs if provider.lower() in p.get("provider", "").lower()]
        if not live_provs:
            continue

        # Tools filter
        if tools_only:
            if not any(p.get("supports_tools", False) for p in live_provs):
                continue

        filtered.append((entry_id, live_provs))

    if not filtered:
        return (
            f"No models match filters.\n"
            f"  model_id={model_id!r}, provider={provider!r}, tools_only={tools_only}\n"
            f"Try widening the search or omitting filters.",
            True,
        )

    # Sort by id
    filtered.sort(key=lambda x: x[0])
    filtered = filtered[:limit]

    # Format output
    lines = [f"**HF Router Catalog — {len(filtered)} model(s)**\n"]

    for entry_id, live_provs in filtered:
        lines.append(f"### {entry_id}")
        for p in live_provs:
            prov_name = p.get("provider", "?")
            ctx = p.get("context_length")
            ctx_str = f"{ctx:,} ctx" if ctx else "ctx n/a"
            tools_str = "✅ tools" if p.get("supports_tools") else "❌ no tools"

            if show_price:
                inp = p.get("pricing", {}).get("input")
                outp = p.get("pricing", {}).get("output")
                if inp is not None and outp is not None:
                    price_str = f" ${inp:.3f}/${outp:.3f} per M tok"
                else:
                    price_str = ""
            else:
                price_str = ""

            lines.append(
                f"  • **{prov_name}** | {ctx_str} | {tools_str}{price_str}"
            )
        lines.append("")

    return "\n".join(lines), True


# Tool specification
HF_CATALOG_TOOL_SPEC = {
    "name": "hf_catalog",
    "description": (
        "Query the HuggingFace Inference Router catalog to discover live models,\n"
        "check pricing, context length, and tool-call support.\n\n"
        "Use this to:\n"
        "  • Find models by partial name match (e.g., 'glm' or 'minimax')\n"
        "  • Filter by provider (e.g., 'cerebras', 'mistralai', 'novita')\n"
        "  • Find the cheapest tool-capable model\n"
        "  • Verify a model is live before switching to it\n\n"
        "Returns a formatted list of models with provider, context length,\n"
        "tool support, and price information. Results are sorted by model id\n"
        "and limited to the top N matches."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "model_id": {
                "type": "string",
                "description": "Filter models by partial name match (case-insensitive).",
            },
            "provider": {
                "type": "string",
                "description": "Filter by provider name (e.g., 'cerebras', 'mistralai').",
            },
            "tools": {
                "type": "boolean",
                "description": "Only return models that support tool calls. Default: False.",
                "default": False,
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return. Default: 20, max: 100.",
                "default": 20,
            },
            "show_price": {
                "type": "boolean",
                "description": "Include pricing info. Default: True.",
                "default": True,
            },
        },
        "additionalProperties": False,
    },
}

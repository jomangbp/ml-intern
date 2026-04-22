"""LiteLLM kwargs resolution for the model ids this agent accepts.

Kept separate from ``agent_loop`` so tools (research, context compaction, etc.)
can import it without pulling in the whole agent loop / tool router and
creating circular imports.
"""

import os
from dataclasses import dataclass
from typing import Optional

# Registry of local / self-hosted providers. Each entry defines how to call
# the model via LiteLLM's OpenAI-compatible adapter.
#
# Format:  prefix (lowercase) -> LocalProvider(
#   api_base  : base URL (no /v1 suffix — we append it)
#   api_key   : token to send, or None for "no auth" (ollama default)
# )
#
# Users can add their own entries here or via a LOCAL_PROVIDERS env var
# whose value is a JSON dict (same shape, overrides hard-coded entries).
@dataclass
class LocalProvider:
    api_base: str        # e.g. "http://localhost:11434/v1"
    api_key: Optional[str]  # None = no auth header
    protocol: str = "openai"  # "openai" | "anthropic"


_LOCAL_PROVIDER_REGISTRY: dict[str, LocalProvider] = {
    # Ollama: runs locally at localhost:11434, no API key needed by default.
    # Use the model name as it appears in `ollama list`, e.g. "ollama/llama3.2".
    "ollama": LocalProvider(api_base="http://localhost:11434/v1", api_key=None),

    # LM Studio: OpenAI-compatible server on localhost.
    "lmstudio": LocalProvider(api_base="http://localhost:1234/v1", api_key="not-needed"),

    # Jan: another OpenAI-compatible local server.
    "jan": LocalProvider(api_base="http://localhost:1337/v1", api_key="not-needed"),

    # MiniMax: direct Anthropic-compatible endpoint (set MINIMAX_API_KEY env var).
    "minimax": LocalProvider(
        api_base="https://api.minimax.io/anthropic",
        api_key=os.environ.get("MINIMAX_API_KEY"),
        protocol="anthropic",
    ),

    # Z.ai: direct Anthropic-compatible endpoint (set ZAI_API_KEY env var).
    "zai": LocalProvider(
        api_base="https://api.z.ai/anthropic/v1",
        api_key=os.environ.get("ZAI_API_KEY"),
        protocol="anthropic",
    ),
}

# Merge any user-supplied providers from the environment
_env_providers_raw = os.environ.get("LOCAL_PROVIDERS", "").strip()
if _env_providers_raw:
    import json
    try:
        extra = json.loads(_env_providers_raw)
        for prefix, cfg in extra.items():
            _LOCAL_PROVIDER_REGISTRY[prefix.lower()] = LocalProvider(
                api_base=cfg["api_base"],
                api_key=cfg.get("api_key"),
                protocol=cfg.get("protocol", "openai"),
            )
    except Exception as e:
        import warnings
        warnings.warn(f"LOCAL_PROVIDERS env var is invalid JSON — ignoring: {e}")


_PROVIDER_OVERRIDES: dict[str, tuple[str, str]] = {
    # HF router–style ids -> (api_base, provider_key)
    # MiniMax M2.7 / M2.5 / M2.1 via Anthropic-compatible endpoint.
    # Set MINIMAX_API_KEY in your environment.
    "MiniMaxAI/": ("https://api.minimax.io/anthropic", "minimax"),
    # Z.ai GLM-5.1 / GLM-4 via Anthropic-compatible endpoint.
    # Set ZAI_API_KEY in your environment.
    "zai-org/": ("https://api.z.ai/anthropic/v1", "zai"),
}

# Provider key env vars. Values are read dynamically at request-time so users
# can export keys after process startup and still switch models without restart.
_PROVIDER_KEY_ENV: dict[str, str] = {
    "minimax": "MINIMAX_API_KEY",
    "zai": "ZAI_API_KEY",
}


def _get_provider_key(provider_key: str) -> str:
    env_name = _PROVIDER_KEY_ENV.get(provider_key)
    if not env_name:
        return ""
    return os.environ.get(env_name, "")


# HF router reasoning models only accept "low" | "medium" | "high" (e.g.
# MiniMax M2 actually *requires* reasoning to be enabled). OpenAI's GPT-5
# also accepts "minimal" for near-zero thinking. We map "minimal" to "low"
# for HF so the user doesn't get a 400.
_HF_ALLOWED_EFFORTS = {"low", "medium", "high"}


def _resolve_llm_params(
    model_name: str,
    session_hf_token: str | None = None,
    reasoning_effort: str | None = None,
) -> dict:
    """
    Build LiteLLM kwargs for a given model id.

    • ``anthropic/<model>`` / ``openai/<model>`` — passed straight through; the
      user's own ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` env vars are picked
      up by LiteLLM. ``reasoning_effort`` is forwarded as a top-level param
      (GPT-5 / o-series accept "minimal" | "low" | "medium" | "high"; Claude
      extended-thinking models accept "low" | "medium" | "high" and LiteLLM
      translates to the thinking config).

    • ``ollama/<model>``, ``lmstudio/<model>``, ``jan/<model>``,
      ``minimax/<model>``, ``zai/<model>`` — any entry in
      ``_LOCAL_PROVIDER_REGISTRY`` is called via the OpenAI/Anthropic-compatible
      adapter configured for that provider. The model name is forwarded as-is so
      it matches what the server expects.

    • Anything else is treated as a HuggingFace router id. We hit the
      auto-routing OpenAI-compatible endpoint at
      ``https://router.huggingface.co/v1``, which bypasses LiteLLM's stale
      per-provider HF adapter entirely. The id can be bare or carry an HF
      routing suffix:

          MiniMaxAI/MiniMax-M2.7              # auto = fastest + failover
          MiniMaxAI/MiniMax-M2.7:cheapest
          moonshotai/Kimi-K2.6:novita         # pin a specific provider

      A leading ``huggingface/`` is stripped for convenience. ``reasoning_effort``
      is forwarded via ``extra_body`` (LiteLLM's OpenAI adapter refuses it as a
      top-level kwarg for non-OpenAI models). "minimal" is normalized to "low".

    Token precedence (first non-empty wins):
      1. INFERENCE_TOKEN env — shared key on the hosted Space (inference is
         free for users, billed to the Space owner via ``X-HF-Bill-To``).
      2. session.hf_token — the user's own token (CLI / OAuth / cache file).
      3. HF_TOKEN env — belt-and-suspenders fallback for CLI users.
    """
    # Check provider overrides (MiniMax / Z.ai via Anthropic-compatible endpoints)
    # These must be checked before the standard anthropic/ prefix so that
    # HF router ids like "MiniMaxAI/MiniMax-M2.7" route correctly.
    for model_prefix, (api_base, prov_key) in _PROVIDER_OVERRIDES.items():
        stripped = model_name.removeprefix("huggingface/")
        if stripped.startswith(model_prefix):
            api_key = _get_provider_key(prov_key)
            params: dict = {
                "model": f"anthropic/{stripped}",
                "api_base": api_base,
            }
            if api_key:
                params["api_key"] = api_key
            elif os.environ.get("ANTHROPIC_API_KEY"):
                params["api_key"] = os.environ.get("ANTHROPIC_API_KEY")
            if reasoning_effort:
                params["reasoning_effort"] = reasoning_effort
            return params

    if model_name.startswith(("anthropic/", "openai/")):
        params: dict = {"model": model_name}
        if reasoning_effort:
            params["reasoning_effort"] = reasoning_effort
        return params


    # Check local provider registry (case-insensitive)
    for prefix, provider in _LOCAL_PROVIDER_REGISTRY.items():
        if model_name.lower().startswith(prefix + "/"):
            # Strip the prefix; what remains is the model name the server expects
            bare_model = model_name[len(prefix) + 1:]
            model_prefix = "anthropic" if provider.protocol == "anthropic" else "openai"
            params: dict = {
                "model": f"{model_prefix}/{bare_model}",
                "api_base": provider.api_base,
            }
            # Prefer dynamic env lookup for providers with dedicated key vars,
            # fallback to static api_key from registry for custom providers.
            dynamic_key = _get_provider_key(prefix)
            if dynamic_key:
                params["api_key"] = dynamic_key
            elif provider.api_key:
                params["api_key"] = provider.api_key
            return params

    hf_model = model_name.removeprefix("huggingface/")
    api_key = (
        os.environ.get("INFERENCE_TOKEN")
        or session_hf_token
        or os.environ.get("HF_TOKEN")
    )
    params = {
        "model": f"openai/{hf_model}",
        "api_base": "https://router.huggingface.co/v1",
        "api_key": api_key,
    }
    if os.environ.get("INFERENCE_TOKEN"):
        params["extra_headers"] = {"X-HF-Bill-To": "huggingface"}
    if reasoning_effort:
        hf_level = "low" if reasoning_effort == "minimal" else reasoning_effort
        if hf_level in _HF_ALLOWED_EFFORTS:
            params["extra_body"] = {"reasoning_effort": hf_level}
    return params

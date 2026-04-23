"""LiteLLM kwargs resolution for the model ids this agent accepts.

Kept separate from ``agent_loop`` so tools (research, context compaction, etc.)
can import it without pulling in the whole agent loop / tool router and
creating circular imports.
"""

import base64
import json
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

    # Z.ai: OpenAI-compatible endpoint (set ZAI_API_KEY env var).
    "zai": LocalProvider(
        # For DevPack coding plans, Z.AI recommends the coding endpoint.
        # Override with ZAI_API_BASE if needed (e.g. general plan).
        api_base=os.environ.get("ZAI_API_BASE", "https://api.z.ai/api/coding/paas/v4"),
        api_key=os.environ.get("ZAI_API_KEY"),
        protocol="openai",
    ),
}

# Merge any user-supplied providers from the environment
_env_providers_raw = os.environ.get("LOCAL_PROVIDERS", "").strip()
if _env_providers_raw:
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
    # Z.ai GLM models via OpenAI-compatible endpoint.
    # Set ZAI_API_KEY in your environment.
    "zai-org/": (os.environ.get("ZAI_API_BASE", "https://api.z.ai/api/coding/paas/v4"), "zai"),
}

# Provider key env vars. Values are read dynamically at request-time so users
# can export keys after process startup and still switch models without restart.
_PROVIDER_KEY_ENV: dict[str, str] = {
    "minimax": "MINIMAX_API_KEY",
    "zai": "ZAI_API_KEY",
}


def _get_provider_key(provider_key: str) -> str:
    env_name = _PROVIDER_KEY_ENV.get(provider_key)
    if env_name:
        return os.environ.get(env_name, "")
    return ""


def _read_hf_cached_token() -> str:
    """Best-effort fallback to Hugging Face CLI token cache."""
    try:
        token_path = os.path.expanduser("~/.cache/huggingface/token")
        with open(token_path, "r", encoding="utf-8") as f:
            return (f.read() or "").strip()
    except Exception:
        return ""


def _extract_codex_account_id(access_token: str) -> str:
    """Extract ChatGPT account id from Codex OAuth JWT access token."""
    try:
        parts = access_token.split(".")
        if len(parts) != 3:
            return ""
        payload_b64 = parts[1] + ("=" * (-len(parts[1]) % 4))
        payload = json.loads(base64.urlsafe_b64decode(payload_b64.encode("utf-8")).decode("utf-8"))
        auth_claim = payload.get("https://api.openai.com/auth") if isinstance(payload, dict) else None
        if isinstance(auth_claim, dict):
            account_id = auth_claim.get("chatgpt_account_id")
            return account_id if isinstance(account_id, str) else ""
        return ""
    except Exception:
        return ""


def _map_openai_model_to_codex_backend(model_id: str) -> str:
    """Map UI OpenAI model ids to ChatGPT Codex backend model ids."""
    model = (model_id or "").strip().lower()
    if model == "gpt-5.3":
        return "gpt-5.3-codex"
    if model == "gpt-5.4-codex":
        return "gpt-5.4"
    if model == "gpt-5.5-codex":
        return "gpt-5.5"
    return model_id


# HF router reasoning models only accept "low" | "medium" | "high" (e.g.
# MiniMax M2 actually *requires* reasoning to be enabled). OpenAI's GPT-5
# also accepts "minimal" for near-zero thinking. We map "minimal" to "low"
# for HF so the user doesn't get a 400.
_HF_ALLOWED_EFFORTS = {"low", "medium", "high"}


def _resolve_llm_params(
    model_name: str,
    session_hf_token: str | None = None,
    reasoning_effort: str | None = None,
    provider_keys: dict[str, str] | None = None,
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
      2. session.hf_token — the user's own token (OAuth header/cookie).
      3. HF_TOKEN env — explicit process-level fallback.
      4. ~/.cache/huggingface/token — local CLI cache fallback.
    """
    # Check provider overrides (MiniMax direct endpoint + ZAI OpenAI-compatible endpoint)
    # These must be checked before the standard anthropic/openai prefixes so
    # HF router ids like "MiniMaxAI/MiniMax-M2.7" route correctly.
    provider_keys = provider_keys or {}

    for model_prefix, (api_base, prov_key) in _PROVIDER_OVERRIDES.items():
        stripped = model_name.removeprefix("huggingface/")
        if stripped.startswith(model_prefix):
            api_key = provider_keys.get(prov_key, "") or _get_provider_key(prov_key)

            # HF-router ids are often namespaced (e.g. "zai-org/GLM-5.1").
            # Direct provider endpoints typically expect the raw model slug
            # without org prefix.
            provider_model = stripped.split("/", 1)[1] if "/" in stripped else stripped

            # ZAI uses an OpenAI-compatible endpoint; use lowercase model slugs
            # like glm-5.1 and avoid Anthropic-specific params.
            if prov_key == "zai":
                provider_model = provider_model.lower()
                params: dict = {
                    "model": f"openai/{provider_model}",
                    "api_base": api_base,
                    # ZAI OpenAI-compatible endpoints can reject some optional
                    # OpenAI-only params (stream_options, etc). Let LiteLLM
                    # prune unsupported fields instead of failing the turn.
                    "drop_params": True,
                }
                if api_key:
                    params["api_key"] = api_key
                # Avoid forcing reasoning_effort for non-OpenAI providers.
                return params

            params: dict = {
                "model": f"anthropic/{provider_model}",
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

        # OpenAI models can use either OPENAI_API_KEY or a Codex OAuth token
        # saved by the local Codex CLI login flow.
        if model_name.startswith("openai/"):
            explicit_openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
            codex_key = ""
            if not explicit_openai_key:
                try:
                    from agent.tools.codex_tool import codex_auth_token

                    codex_key = (codex_auth_token() or "").strip()
                except Exception:
                    codex_key = ""

            openai_model_id = model_name.split("/", 1)[1] if "/" in model_name else model_name

            # If user authenticated via Codex CLI (no OPENAI_API_KEY), route GPT-5.x
            # through ChatGPT's Codex backend — same family PI uses.
            if codex_key and not explicit_openai_key:
                codex_model_id = _map_openai_model_to_codex_backend(openai_model_id)
                codex_account_id = _extract_codex_account_id(codex_key)
                if codex_account_id:
                    params = {
                        "model": f"openai/{codex_model_id}",
                        "api_key": codex_key,
                        "api_base": os.environ.get(
                            "OPENAI_CODEX_API_BASE",
                            "https://chatgpt.com/backend-api/codex",
                        ),
                        "extra_headers": {
                            "chatgpt-account-id": codex_account_id,
                            "originator": "ml-intern",
                            "OpenAI-Beta": "responses=experimental",
                        },
                        # Codex backend rejects several optional OpenAI params;
                        # pruning them avoids hard failures in chat turns.
                        "drop_params": True,
                        # Codex backend requires store=false and an instructions
                        # field even when messages are provided.
                        "extra_body": {
                            "store": False,
                            "instructions": "You are a helpful assistant.",
                        },
                    }
                    if reasoning_effort:
                        params["reasoning_effort"] = reasoning_effort
                    return params

            api_key = explicit_openai_key or codex_key
            if api_key:
                params["api_key"] = api_key
            if os.environ.get("OPENAI_API_BASE"):
                params["api_base"] = os.environ.get("OPENAI_API_BASE")

        if reasoning_effort:
            params["reasoning_effort"] = reasoning_effort
        return params


    if model_name.startswith("bedrock/"):
        # Route directly through LiteLLM Bedrock adapter (AWS creds from env).
        return {"model": model_name}

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
            dynamic_key = provider_keys.get(prefix, "") or _get_provider_key(prefix)
            if dynamic_key:
                params["api_key"] = dynamic_key
            elif provider.api_key:
                params["api_key"] = provider.api_key
            if prefix == "zai":
                params["drop_params"] = True
            return params

    hf_model = model_name.removeprefix("huggingface/")
    api_key = (
        os.environ.get("INFERENCE_TOKEN")
        or session_hf_token
        or os.environ.get("HF_TOKEN")
        or _read_hf_cached_token()
    )
    params = {
        "model": f"openai/{hf_model}",
        "api_base": "https://router.huggingface.co/v1",
        "api_key": api_key,
    }
    if os.environ.get("INFERENCE_TOKEN"):
        bill_to = os.environ.get("HF_BILL_TO", "smolagents")
        params["extra_headers"] = {"X-HF-Bill-To": bill_to}
    if reasoning_effort:
        hf_level = "low" if reasoning_effort == "minimal" else reasoning_effort
        if hf_level in _HF_ALLOWED_EFFORTS:
            params["extra_body"] = {"reasoning_effort": hf_level}
    return params

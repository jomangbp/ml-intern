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


def _patch_litellm_effort_validation() -> None:
    """Neuter LiteLLM 1.83's hardcoded effort-level validation.

    Context: at ``litellm/llms/anthropic/chat/transformation.py:~1443`` the
    Anthropic adapter validates ``output_config.effort ∈ {high, medium,
    low, max}`` and gates ``max`` behind an ``_is_opus_4_6_model`` check
    that only matches the substring ``opus-4-6`` / ``opus_4_6``. Result:

    * ``xhigh`` — valid on Anthropic's real API for Claude 4.7 — is
      rejected pre-flight with "Invalid effort value: xhigh".
    * ``max`` on Opus 4.7 is rejected with "effort='max' is only supported
      by Claude Opus 4.6", even though Opus 4.7 accepts it in practice.

    We don't want to maintain a parallel model table, so we let the
    Anthropic API itself be the validator: widen ``_is_opus_4_6_model``
    to also match ``opus-4-7``+ families, and drop the valid-effort-set
    check entirely. If Anthropic rejects an effort level, we see a 400
    and the cascade walks down — exactly the behavior we want for any
    future model family.

    Removable once litellm ships 1.83.8-stable (which merges PR #25867,
    "Litellm day 0 opus 4.7 support") — see commit 0868a82 on their main
    branch. Until then, this one-time patch is the escape hatch.
    """
    try:
        from litellm.llms.anthropic.chat import transformation as _t
    except Exception:
        return

    cfg = getattr(_t, "AnthropicConfig", None)
    if cfg is None:
        return

    original = getattr(cfg, "_is_opus_4_6_model", None)
    if original is None or getattr(original, "_hf_agent_patched", False):
        return

    def _widened(model: str) -> bool:
        m = model.lower()
        # Original 4.6 match plus any future Opus >= 4.6. We only need this
        # to return True for families where "max" / "xhigh" are acceptable
        # at the API; the cascade handles the case when they're not.
        return any(
            v in m for v in (
                "opus-4-6", "opus_4_6", "opus-4.6", "opus_4.6",
                "opus-4-7", "opus_4_7", "opus-4.7", "opus_4.7",
            )
        )

    _widened._hf_agent_patched = True  # type: ignore[attr-defined]
    cfg._is_opus_4_6_model = staticmethod(_widened)


_patch_litellm_effort_validation()


# Effort levels accepted on the wire.
#   Anthropic (4.6+):  low | medium | high | xhigh | max   (output_config.effort)
#   OpenAI direct:     minimal | low | medium | high | xhigh (reasoning_effort top-level)
#   HF router:         low | medium | high                 (extra_body.reasoning_effort)
#
# We validate *shape* here and let the probe cascade walk down on rejection;
# we deliberately do NOT maintain a per-model capability table.
_ANTHROPIC_EFFORTS = {"low", "medium", "high", "xhigh", "max"}
_OPENAI_EFFORTS = {"minimal", "low", "medium", "high", "xhigh"}
_HF_EFFORTS = {"low", "medium", "high"}


class UnsupportedEffortError(ValueError):
    """The requested effort isn't valid for this provider's API surface.

    Raised synchronously before any network call so the probe cascade can
    skip levels the provider can't accept (e.g. ``max`` on HF router).
    """


def _resolve_llm_params(
    model_name: str,
    session_hf_token: str | None = None,
    reasoning_effort: str | None = None,
    provider_keys: dict[str, str] | None = None,
    strict: bool = False,
) -> dict:
    """
    Build LiteLLM kwargs for a given model id.

    • ``anthropic/<model>`` — native thinking config. We bypass LiteLLM's
      ``reasoning_effort`` → ``thinking`` mapping (which lags new Claude
      releases like 4.7 and sends the wrong API shape). Instead we pass
      both ``thinking={"type": "adaptive"}`` and ``output_config=
      {"effort": <level>}`` as top-level kwargs — LiteLLM's Anthropic
      adapter forwards unknown top-level kwargs into the request body
      verbatim (confirmed by live probe; ``extra_body`` does NOT work
      here because Anthropic's API rejects it as "Extra inputs are not
      permitted"). This is the stable API for 4.6 and 4.7. Older
      extended-thinking models that only accept ``thinking.type.enabled``
      will reject this; the probe's cascade catches that and falls back
      to no thinking.

    • ``openai/<model>`` — ``reasoning_effort`` forwarded as a top-level
      kwarg (GPT-5 / o-series). LiteLLM uses the user's ``OPENAI_API_KEY``.

    • ``ollama/<model>``, ``lmstudio/<model>``, ``jan/<model>``,
      ``minimax/<model>``, ``zai/<model>`` — any entry in
      ``_LOCAL_PROVIDER_REGISTRY`` is called via the OpenAI/Anthropic-compatible
      adapter configured for that provider. The model name is forwarded as-is so
      it matches what the server expects.

    • Anything else is treated as a HuggingFace router id. We hit the
      auto-routing OpenAI-compatible endpoint at
      ``https://router.huggingface.co/v1``. The id can be bare or carry an
      HF routing suffix (``:fastest`` / ``:cheapest`` / ``:<provider>``).
      A leading ``huggingface/`` is stripped. ``reasoning_effort`` is
      forwarded via ``extra_body`` (LiteLLM's OpenAI adapter refuses it as
      a top-level kwarg for non-OpenAI models). "minimal" normalizes to
      "low".

    ``strict=True`` raises ``UnsupportedEffortError`` when the requested
    effort isn't in the provider's accepted set, instead of silently
    dropping it. The probe cascade uses strict mode so it can walk down
    (``max`` → ``xhigh`` → ``high`` …) without making an API call. Regular
    runtime callers leave ``strict=False``, so a stale cached effort
    can't crash a turn — it just doesn't get sent.

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
            level = reasoning_effort
            if level == "minimal":
                level = "low"
            if level not in _ANTHROPIC_EFFORTS:
                if strict:
                    raise UnsupportedEffortError(
                        f"Anthropic doesn't accept effort={level!r}"
                    )
            else:
                # Adaptive thinking + output_config.effort is the stable
                # Anthropic API for Claude 4.6 / 4.7. Both kwargs are
                # passed top-level: LiteLLM forwards unknown params into
                # the request body for Anthropic, so ``output_config``
                # reaches the API. ``extra_body`` does NOT work here —
                # Anthropic rejects it as "Extra inputs are not
                # permitted".
                params["thinking"] = {"type": "adaptive"}
                params["output_config"] = {"effort": level}
        return params

    if model_name.startswith("openai/"):
        params = {"model": model_name}
        if reasoning_effort:
            if reasoning_effort not in _OPENAI_EFFORTS:
                if strict:
                    raise UnsupportedEffortError(
                        f"OpenAI doesn't accept effort={reasoning_effort!r}"
                    )
            else:
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
        if hf_level not in _HF_EFFORTS:
            if strict:
                raise UnsupportedEffortError(
                    f"HF router doesn't accept effort={hf_level!r}"
                )
        else:
            params["extra_body"] = {"reasoning_effort": hf_level}
    return params

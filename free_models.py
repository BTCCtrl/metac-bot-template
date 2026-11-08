"""
Pick the best OpenRouter model that is currently FREE (`:free` model ids, $0).

Used by main.py when BOT_MODE=free (the default in this fork).

- `fetch_models()` downloads the public model list (no API key needed).
- `pick_free_model(models_json)` returns the best `:free` id by a preference
  ranking (strongest reasoning/general models first), skipping tiny or
  special-purpose models.
- Env var `FREE_MODEL` (or `FREE_PARSER_MODEL` for the parser) overrides the
  choice, but must still be a `:free` id.
- If the model list cannot be fetched, `FALLBACK_FREE_MODELS` is used in order.

Hard rule: nothing in this module ever returns an id that does not end with
":free". Free model ids rotate on OpenRouter, which is why the choice is made
at every run instead of being hard-coded.

Only the Python standard library is used, so this file is cheap to test.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from typing import Any, Sequence

logger = logging.getLogger(__name__)

MODELS_URL = "https://openrouter.ai/api/v1/models"
FREE_SUFFIX = ":free"

# Regexes matched (re.search) against the lower-cased model id WITHOUT ":free".
# Order = preference. Strongest general/reasoning models first.
DEFAULT_PREFERENCES: tuple[str, ...] = (
    r"^deepseek/deepseek-v3\.[1-9]",  # deepseek-v3.1 / v3.2(-exp) ...
    r"^deepseek/deepseek-chat-v3\.[1-9]",  # deepseek-chat-v3.1
    r"^deepseek/deepseek-r1-0528$",
    r"^deepseek/deepseek-r1$",
    r"^x-ai/grok-4",
    r"^qwen/qwen3-max",
    r"^qwen/qwen3-235b-a22b",
    r"^deepseek/deepseek-chat-v3-0324$",
    r"^deepseek/deepseek-chat(-v3)?$",
    r"^openai/gpt-oss-120b",
    r"^moonshotai/kimi-k2",
    r"^tngtech/deepseek-r1t2-chimera",
    r"^z-ai/glm-4\.[5-9]",
    r"^minimax/minimax-m[1-9]",
    r"^tngtech/deepseek-r1t-chimera",
    r"^qwen/qwen3-next-80b",
    r"^meta-llama/llama-4-maverick",
    r"^nousresearch/hermes-[34]-llama-3\.1-405b",
    r"^meta-llama/llama-3\.1-405b",
    r"^meta-llama/llama-3\.3-70b",
    r"^microsoft/mai-ds-r1",
    r"^qwen/qwen3-(32b|30b)",
    r"^qwen/qwq-32b",
    r"^google/gemini-2\.\d-flash",
    r"^google/gemma-3-27b",
    r"^mistralai/mistral-small-3",
    r"^meta-llama/llama-4-scout",
    r"^openai/gpt-oss-20b",
)

# The parser only turns the forecaster's text into JSON. Fast non-reasoning
# models first (reasoning models are slower and can time out on free endpoints).
PARSER_PREFERENCES: tuple[str, ...] = (
    r"^deepseek/deepseek-chat-v3",
    r"^deepseek/deepseek-v3\.[1-9]",
    r"^meta-llama/llama-3\.3-70b",
    r"^moonshotai/kimi-k2",
    r"^google/gemma-3-27b",
    r"^mistralai/mistral-small-3",
    r"^meta-llama/llama-4-maverick",
    r"^z-ai/glm-4\.[5-9]",
    r"^openai/gpt-oss-120b",
) + DEFAULT_PREFERENCES

# Tried in this order when the live list cannot be downloaded.
FALLBACK_FREE_MODELS: tuple[str, ...] = (
    "deepseek/deepseek-chat-v3.1:free",
    "deepseek/deepseek-r1-0528:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "qwen/qwen3-235b-a22b:free",
    "openai/gpt-oss-120b:free",
    "moonshotai/kimi-k2:free",
    "z-ai/glm-4.5-air:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-3-27b-it:free",
    "mistralai/mistral-small-3.2-24b-instruct:free",
)

MIN_PARAMS_BILLIONS = 20  # "tiny" = fewer parameters than this (when the id says)
_TINY_TOKENS = {"nano", "tiny", "micro"}
_SPECIAL_PURPOSE_TOKENS = {
    "coder", "code", "vl", "vision", "guard", "embed", "embedding", "audio",
    "tts", "ocr", "image", "prover", "math", "distill", "safeguard",
}
_REASONING_HINT = re.compile(r"(-r1\b|-r1-|r1t|reason|think|qwq|gpt-oss)")


class NotAFreeModelError(ValueError):
    """Raised when something tries to use a model id that is not `:free`."""


def assert_free(model_id: str) -> str:
    """Return `model_id` unchanged if it is a `:free` id, otherwise raise."""
    if not isinstance(model_id, str) or not model_id.endswith(FREE_SUFFIX):
        raise NotAFreeModelError(
            f"Refusing non-free model id {model_id!r}: only ':free' ids are allowed"
        )
    return model_id


def fetch_models(url: str = MODELS_URL, timeout: float = 20) -> dict | None:
    """Download OpenRouter's public model list. Returns None on any failure."""
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "metac-bot-free-mode/1.0"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("data"), list):
            logger.warning("OpenRouter model list has an unexpected shape")
            return None
        return data
    except Exception as error:  # network, HTTP, JSON... all mean "use fallback"
        logger.warning(f"Could not fetch OpenRouter model list: {error!r}")
        return None


def _params_billions(model_id: str) -> float | None:
    sizes = [float(n) for n in re.findall(r"(\d+(?:\.\d+)?)b(?![a-z])", model_id)]
    return max(sizes) if sizes else None


def _tokens(model_id: str) -> set[str]:
    return set(re.split(r"[/\-_.:]+", model_id.lower()))


def _is_zero_price(entry: dict[str, Any]) -> bool:
    pricing = entry.get("pricing")
    if not isinstance(pricing, dict):
        return True  # no pricing info: rely on the ":free" suffix
    for value in pricing.values():
        try:
            if float(value) != 0:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _outputs_text(entry: dict[str, Any]) -> bool:
    architecture = entry.get("architecture")
    if not isinstance(architecture, dict):
        return True
    outputs = architecture.get("output_modalities")
    if isinstance(outputs, list) and outputs:
        return "text" in outputs
    modality = architecture.get("modality")
    if isinstance(modality, str) and "->" in modality:
        return "text" in modality.split("->", 1)[1]
    return True


def is_usable_free_entry(entry: Any) -> bool:
    """True for a `:free`, zero-priced, text-out, general-purpose, non-tiny model."""
    if not isinstance(entry, dict):
        return False
    model_id = entry.get("id")
    if not isinstance(model_id, str) or not model_id.endswith(FREE_SUFFIX):
        return False
    if model_id.startswith("openrouter/"):  # meta-routers (auto/free) are opaque
        return False
    if not _is_zero_price(entry) or not _outputs_text(entry):
        return False
    tokens = _tokens(model_id)
    if tokens & (_TINY_TOKENS | _SPECIAL_PURPOSE_TOKENS):
        return False
    size = _params_billions(model_id.lower())
    if size is not None and size < MIN_PARAMS_BILLIONS:
        return False
    return True


def rank_free_models(
    models_json: dict | None,
    preferences: Sequence[str] = DEFAULT_PREFERENCES,
) -> list[str]:
    """
    All usable `:free` ids, best first. Preferred patterns first (newest id
    first within a pattern), then any other usable free model by size.
    With no usable list, returns FALLBACK_FREE_MODELS.
    """
    entries = models_json.get("data") if isinstance(models_json, dict) else None
    available = sorted(
        {e["id"] for e in entries or [] if is_usable_free_entry(e)}
    )
    if not available:
        return list(FALLBACK_FREE_MODELS)

    ranked: list[str] = []
    for pattern in preferences:
        regex = re.compile(pattern)
        matches = [
            m for m in available
            if m not in ranked and regex.search(m.lower().removesuffix(FREE_SUFFIX))
        ]
        ranked.extend(sorted(matches, reverse=True))
    rest = [m for m in available if m not in ranked]
    rest.sort(key=lambda m: (-(_params_billions(m.lower()) or 0), m))
    ranked.extend(rest)
    return [assert_free(m) for m in ranked]


def _override(env_var: str) -> str | None:
    value = (os.getenv(env_var) or "").strip()
    if not value:
        return None
    value = value.removeprefix("openrouter/")
    return assert_free(value)  # an override can never select a paid model


def free_model_candidates(
    models_json: dict | None,
    *,
    preferences: Sequence[str] = DEFAULT_PREFERENCES,
    env_var: str = "FREE_MODEL",
    limit: int = 5,
) -> list[str]:
    """Best choice first, then backups to try if a model is unavailable."""
    ranked = rank_free_models(models_json, preferences)
    override = _override(env_var)
    if override:
        if models_json and override not in ranked:
            logger.warning(f"{env_var}={override} is not in the current free list")
        ranked = [override] + [m for m in ranked if m != override]
    return [assert_free(m) for m in ranked[:limit]]


def pick_free_model(
    models_json: dict | None,
    *,
    preferences: Sequence[str] = DEFAULT_PREFERENCES,
    env_var: str = "FREE_MODEL",
) -> str:
    """The single best currently-available `:free` model id (no provider prefix)."""
    return free_model_candidates(
        models_json, preferences=preferences, env_var=env_var, limit=1
    )[0]


def is_reasoning_model(model_id: str) -> bool:
    return bool(_REASONING_HINT.search(model_id.lower()))

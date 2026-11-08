"""Shared fixtures. All tests are offline: no OpenRouter, no Metaculus calls."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Env vars the bot reads. Cleared for every test so a local .env or shell
# cannot change results (and real secrets are never needed).
BOT_ENV_VARS = (
    "BOT_MODE",
    "FREE_MODEL",
    "FREE_PARSER_MODEL",
    "MAX_LLM_CALLS_PER_RUN",
    "MAX_QUESTIONS_PER_RUN",
    "PREDICTIONS_PER_REPORT",
    "MIN_SECONDS_BETWEEN_CALLS",
    "FREE_RESEARCH",
    "FREE_RESEARCH_DAYS",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "PERPLEXITY_API_KEY",
    "EXA_API_KEY",
    "ASKNEWS_CLIENT_ID",
    "ASKNEWS_SECRET",
    "ASKNEWS_API_KEY",
    "GITHUB_STEP_SUMMARY",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in BOT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("METACULUS_TOKEN", "fake-metaculus-token-for-tests")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-openrouter-key-for-tests")


@pytest.fixture(autouse=True)
def no_gdelt_network(monkeypatch):
    """[FREE RESEARCH] GDELT is never contacted in tests (CI has network).
    Research then fails fast and the bot falls back to the no-research note,
    exactly as on a GDELT outage. Tests that need headlines pass a fake."""
    import free_research

    async def offline(url, timeout):
        raise ConnectionError("network disabled in tests")

    monkeypatch.setattr(free_research, "_http_get", offline)
    monkeypatch.setattr(free_research, "MIN_SECONDS_BETWEEN_REQUESTS", 0.0)


@pytest.fixture
def models_json():
    return json.loads((FIXTURES / "openrouter_models.json").read_text("utf-8"))

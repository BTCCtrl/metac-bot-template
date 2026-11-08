import copy
import io
import json
import random

import pytest

import free_models
from free_models import (
    FALLBACK_FREE_MODELS,
    NotAFreeModelError,
    fetch_models,
    free_model_candidates,
    pick_free_model,
    rank_free_models,
)


def _without(models_json, *ids):
    data = copy.deepcopy(models_json)
    data["data"] = [m for m in data["data"] if m["id"] not in ids]
    return data


def _price_is_zero(models_json, model_id):
    entry = next(m for m in models_json["data"] if m["id"] == model_id)
    return all(float(v) == 0 for v in entry["pricing"].values())


# ---------------------------------------------------------------- ranking


def test_picks_strongest_free_model(models_json):
    assert pick_free_model(models_json) == "deepseek/deepseek-r1-0528:free"


def test_full_ranking_order(models_json):
    assert rank_free_models(models_json) == [
        "deepseek/deepseek-r1-0528:free",
        "qwen/qwen3-235b-a22b:free",
        "deepseek/deepseek-chat-v3-0324:free",
        "openai/gpt-oss-120b:free",
        "moonshotai/kimi-k2:free",
        "z-ai/glm-4.5-air:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemma-3-27b-it:free",
        "mistralai/mistral-small-3.2-24b-instruct:free",
        "openai/gpt-oss-20b:free",
    ]


def test_skips_tiny_special_paid_and_mispriced(models_json):
    ranked = rank_free_models(models_json)
    for excluded in (
        "qwen/qwen3-4b:free",  # tiny
        "meta-llama/llama-3.2-3b-instruct:free",  # tiny
        "nvidia/nemotron-nano-9b-v2:free",  # nano
        "qwen/qwen3-coder:free",  # special purpose
        "deepseek/deepseek-r1-distill-llama-70b:free",  # distill
        "google/gemini-2.5-flash-image-preview:free",  # image output
        "example/strange-model-405b:free",  # ':free' id but non-zero price
        "deepseek/deepseek-r1-0528",  # paid variant
        "openai/gpt-4o-mini",
        "openrouter/auto",
    ):
        assert excluded not in ranked


def test_next_best_when_top_model_disappears(models_json):
    data = _without(models_json, "deepseek/deepseek-r1-0528:free")
    assert pick_free_model(data) == "qwen/qwen3-235b-a22b:free"


def test_newer_deepseek_v3_is_preferred_when_listed(models_json):
    data = copy.deepcopy(models_json)
    newer = copy.deepcopy(data["data"][5])  # deepseek-chat-v3-0324:free
    newer["id"] = "deepseek/deepseek-v3.2-exp:free"
    data["data"].append(newer)
    assert pick_free_model(data) == "deepseek/deepseek-v3.2-exp:free"


def test_unknown_free_models_are_ranked_by_size_after_known_ones(models_json):
    data = _without(models_json, *[m["id"] for m in models_json["data"]])
    template = models_json["data"][3]
    for model_id in ("acme/foo-30b:free", "acme/bar-120b:free", "acme/baz-7b:free"):
        entry = copy.deepcopy(template)
        entry["id"] = model_id
        data["data"].append(entry)
    assert rank_free_models(data) == ["acme/bar-120b:free", "acme/foo-30b:free"]


def test_parser_prefers_fast_non_reasoning_model(models_json):
    parser = pick_free_model(
        models_json,
        preferences=free_models.PARSER_PREFERENCES,
        env_var="FREE_PARSER_MODEL",
    )
    assert parser == "deepseek/deepseek-chat-v3-0324:free"
    assert not free_models.is_reasoning_model(parser)
    assert free_models.is_reasoning_model("deepseek/deepseek-r1-0528:free")


# ---------------------------------------------------------------- override


def test_env_override_wins(models_json, monkeypatch):
    monkeypatch.setenv("FREE_MODEL", "openrouter/google/gemma-3-27b-it:free")
    assert pick_free_model(models_json) == "google/gemma-3-27b-it:free"
    candidates = free_model_candidates(models_json)
    assert candidates[0] == "google/gemma-3-27b-it:free"
    assert len(candidates) == len(set(candidates))


def test_env_override_must_be_free(models_json, monkeypatch):
    monkeypatch.setenv("FREE_MODEL", "openai/gpt-4o")
    with pytest.raises(NotAFreeModelError):
        pick_free_model(models_json)


def test_blank_override_is_ignored(models_json, monkeypatch):
    monkeypatch.setenv("FREE_MODEL", "   ")
    assert pick_free_model(models_json) == "deepseek/deepseek-r1-0528:free"


# ---------------------------------------------------------------- fallback


@pytest.mark.parametrize(
    "bad", [None, {}, {"data": []}, {"data": "oops"}, {"data": [{"id": 3}, "x"]}]
)
def test_fallback_when_list_unusable(bad):
    assert pick_free_model(bad) == FALLBACK_FREE_MODELS[0]
    assert free_model_candidates(bad, limit=99) == list(FALLBACK_FREE_MODELS)


def test_fallback_when_only_paid_models_listed(models_json):
    data = copy.deepcopy(models_json)
    data["data"] = [m for m in data["data"] if not m["id"].endswith(":free")]
    assert pick_free_model(data) == FALLBACK_FREE_MODELS[0]


def test_fetch_models_returns_none_on_network_error(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("blocked")

    monkeypatch.setattr(free_models.urllib.request, "urlopen", boom)
    assert fetch_models() is None
    assert pick_free_model(fetch_models()) == FALLBACK_FREE_MODELS[0]


def test_fetch_models_parses_json_with_20s_timeout(monkeypatch, models_json):
    seen = {}

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        return FakeResponse(json.dumps(models_json).encode())

    monkeypatch.setattr(free_models.urllib.request, "urlopen", fake_urlopen)
    assert fetch_models() == models_json
    assert seen == {"url": "https://openrouter.ai/api/v1/models", "timeout": 20}


def test_fetch_models_rejects_unexpected_shape(monkeypatch):
    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(
        free_models.urllib.request,
        "urlopen",
        lambda request, timeout: FakeResponse(b'["not", "a", "dict"]'),
    )
    assert fetch_models() is None


# ---------------------------------------------------------------- never paid


def test_never_returns_a_paid_model(models_json):
    rng = random.Random(1234)
    entries = models_json["data"]
    for _ in range(300):
        subset = {"data": rng.sample(entries, rng.randint(0, len(entries)))}
        for preferences in (
            free_models.DEFAULT_PREFERENCES,
            free_models.PARSER_PREFERENCES,
        ):
            for model_id in free_model_candidates(
                subset, preferences=preferences, limit=99
            ):
                assert model_id.endswith(":free")
                if any(m["id"] == model_id for m in subset["data"]):
                    assert _price_is_zero(models_json, model_id)


def test_all_fallbacks_are_free():
    assert all(m.endswith(":free") for m in FALLBACK_FREE_MODELS)


def test_assert_free():
    assert free_models.assert_free("a/b:free") == "a/b:free"
    for bad in ("openai/gpt-4o", "a/b:freebie", "", None):
        with pytest.raises(NotAFreeModelError):
            free_models.assert_free(bad)

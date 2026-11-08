"""
Free-mode smoke + offline end-to-end tests.

The LLM is faked at litellm's `acompletion` (the guard still wraps it), the
model list comes from the fixture, and Metaculus is never contacted.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone

import httpx
import litellm
import pytest
import tenacity
from forecasting_tools import (
    BinaryQuestion,
    ForecastReport,
    GeneralLlm,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.ai_models import general_llm
from forecasting_tools.ai_models.model_interfaces import retryable_model
from litellm import ModelResponse

import free_mode
import free_models
import main

NOW = datetime(2026, 11, 2, 12, tzinfo=timezone.utc)


# ---------------------------------------------------------------- helpers


@pytest.fixture(autouse=True)
def offline(monkeypatch, models_json):
    """No network: fixture model list, no retry waits, no request spacing."""
    monkeypatch.setattr(free_models, "fetch_models", lambda *a, **k: models_json)
    monkeypatch.setattr(
        retryable_model, "wait_random_exponential", lambda **kw: tenacity.wait_none()
    )
    monkeypatch.setenv("MIN_SECONDS_BETWEEN_CALLS", "0")


class FakeOpenRouter:
    """Stands in for litellm.acompletion and answers like a decent model."""

    def __init__(self, fail_with=None):
        self.models = []
        self.prompts = []
        self.fail_with = fail_with

    async def __call__(self, *args, **kwargs):
        model = kwargs["model"]
        prompt = kwargs["messages"][-1]["content"]
        self.models.append(model)
        self.prompts.append(prompt)
        if self.fail_with is not None:
            raise self.fail_with
        return ModelResponse(
            model=model,
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": self.answer(prompt)},
                }
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        )

    @staticmethod
    def answer(prompt):
        if "convert the text" in prompt:  # structure_output (parser)
            if "PredictedOptionList" in prompt:
                return json.dumps(
                    {
                        "predicted_options": [
                            {"option_name": "Red", "probability": 0.5},
                            {"option_name": "Green", "probability": 0.3},
                            {"option_name": "Blue", "probability": 0.2},
                        ]
                    }
                )
            if "Percentile" in prompt:
                return json.dumps(
                    [
                        {"percentile": p, "value": v}
                        for p, v in (
                            (0.1, 10), (0.2, 20), (0.4, 40),
                            (0.6, 60), (0.8, 80), (0.9, 90),
                        )
                    ]
                )
            raise AssertionError("unexpected parser prompt")
        if "The options are" in prompt:
            return "Reasoning...\nRed: 50%\nGreen: 30%\nBlue: 20%"
        if "Percentile 10" in prompt:
            return (
                "Reasoning...\nPercentile 10: 10\nPercentile 20: 20\nPercentile 40: 40\n"
                "Percentile 60: 60\nPercentile 80: 80\nPercentile 90: 90"
            )
        return "(a) weeks\n(b) No\nSo, all things considered:\nProbability: 37%"

    def parser_calls(self):
        return sum("convert the text" in p for p in self.prompts)


@pytest.fixture
def fake_llm(monkeypatch):
    fake = FakeOpenRouter()
    # The guard wraps the fake exactly like it wraps the real function.
    monkeypatch.setattr(general_llm, "acompletion", free_mode._guarded(fake))
    return fake


def binary(n, hours_to_close=24, forecasted=False):
    return BinaryQuestion(
        question_text=f"Will thing {n} happen?",
        id_of_post=n,
        id_of_question=1000 + n,
        page_url=f"https://www.metaculus.com/questions/{n}/",
        background_info="bg",
        resolution_criteria="rc",
        fine_print="fp",
        close_time=NOW + timedelta(hours=hours_to_close),
        already_forecasted=forecasted,
    )


def multiple_choice(n, hours_to_close=24):
    return MultipleChoiceQuestion(
        question_text=f"Which colour {n}?",
        id_of_post=n,
        id_of_question=1000 + n,
        page_url=f"https://www.metaculus.com/questions/{n}/",
        options=["Red", "Green", "Blue"],
        close_time=NOW + timedelta(hours=hours_to_close),
        already_forecasted=False,
    )


def numeric(n, hours_to_close=24):
    return NumericQuestion(
        question_text=f"How many {n}?",
        id_of_post=n,
        id_of_question=1000 + n,
        page_url=f"https://www.metaculus.com/questions/{n}/",
        unit_of_measure="units",
        upper_bound=100,
        lower_bound=0,
        open_upper_bound=True,
        open_lower_bound=True,
        close_time=NOW + timedelta(hours=hours_to_close),
        already_forecasted=False,
    )


def build_free_bot(monkeypatch, publish=False, **env):
    monkeypatch.setenv("BOT_MODE", "free")
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    bot = main.build_bot(publish_to_metaculus=True)
    assert bot.publish_reports_to_metaculus is True  # comments are posted
    bot.publish_reports_to_metaculus = publish
    return bot


def run(bot, questions):
    return asyncio.run(bot.forecast_questions(questions, return_exceptions=True))


# ---------------------------------------------------------------- config smoke


def test_free_config_every_llm_role_is_free(monkeypatch):
    monkeypatch.setenv("BOT_MODE", "free")
    bot = main.build_bot(publish_to_metaculus=True)
    roles = bot.make_llm_dict()
    assert set(roles) == {"default", "summarizer", "researcher", "parser"}
    assert bot.get_llm("researcher") == "no_research"
    for role in ("default", "summarizer", "parser"):
        llm = bot.get_llm(role, "llm")
        assert isinstance(llm, free_mode.FreeOpenRouterLlm)
        assert llm.model.startswith("openrouter/")
        assert llm.model.endswith(":free")
        assert llm.litellm_kwargs["model"] == llm.model
        assert llm.allowed_tries == 2
        assert llm.litellm_kwargs["temperature"] == 0.3
    assert bot.get_llm("default", "llm").model == "openrouter/deepseek/deepseek-r1-0528:free"
    assert bot.get_llm("parser", "llm").model == "openrouter/deepseek/deepseek-chat-v3-0324:free"
    assert bot.get_llm("default", "llm").litellm_kwargs["timeout"] == 300  # reasoning
    assert bot.get_llm("parser", "llm").litellm_kwargs["timeout"] == 180


def test_free_config_saves_requests_and_keeps_comments(monkeypatch):
    monkeypatch.setenv("BOT_MODE", "free")
    bot = main.build_bot(publish_to_metaculus=True)
    assert bot.publish_reports_to_metaculus is True  # comments required for prizes
    assert bot.enable_summarize_research is False
    assert bot._structure_output_validation_samples == 1
    assert bot._structure_output_allowed_tries == 1  # failing parse costs 1 request
    assert bot.skip_previously_forecasted_questions is True
    assert bot.research_reports_per_question == 1
    assert bot.predictions_per_research_report == 3
    assert bot._free_runtime.budget.max_calls == 6


def test_bot_mode_defaults_to_free(monkeypatch):
    bot = main.build_bot(publish_to_metaculus=True)  # BOT_MODE unset
    assert bot._free_runtime is not None


def test_free_settings_from_env(monkeypatch):
    monkeypatch.setenv("MAX_LLM_CALLS_PER_RUN", "120")
    monkeypatch.setenv("MAX_QUESTIONS_PER_RUN", "40")
    monkeypatch.setenv("PREDICTIONS_PER_REPORT", "5")
    monkeypatch.setenv("MIN_SECONDS_BETWEEN_CALLS", "")
    settings = free_mode.FreeSettings.from_env()
    assert settings == free_mode.FreeSettings(5, 120, 40, 4.0)
    monkeypatch.setenv("MAX_LLM_CALLS_PER_RUN", "zero")
    with pytest.raises(ValueError):
        free_mode.FreeSettings.from_env()
    monkeypatch.setenv("MAX_LLM_CALLS_PER_RUN", "0")
    with pytest.raises(ValueError):
        free_mode.FreeSettings.from_env()


def test_paid_mode_is_the_original_template(monkeypatch):
    def must_not_fetch(*a, **k):
        raise AssertionError("paid mode must not touch the free model list")

    monkeypatch.setattr(free_models, "fetch_models", must_not_fetch)
    monkeypatch.setenv("BOT_MODE", "paid")
    bot = main.build_bot(publish_to_metaculus=True)
    assert bot._free_runtime is None
    assert bot.predictions_per_research_report == 5
    assert bot._structure_output_validation_samples == 2
    assert bot._structure_output_allowed_tries == 3  # structure_output's default
    assert bot.enable_summarize_research is True


def test_invalid_bot_mode_is_a_config_error(monkeypatch):
    monkeypatch.setenv("BOT_MODE", "cheap")
    with pytest.raises(ValueError):
        main.build_bot(publish_to_metaculus=True)


def test_missing_openrouter_key_exits(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(SystemExit) as exit_info:
        free_mode.require_openrouter_key()
    assert exit_info.value.code == 1


# ---------------------------------------------------------------- paid guard


def test_guard_blocks_paid_models_before_any_request(monkeypatch):
    sent = []

    async def would_send(*args, **kwargs):
        sent.append(kwargs.get("model"))

    monkeypatch.setattr(general_llm, "acompletion", would_send)
    monkeypatch.setattr(litellm, "acompletion", would_send)
    free_mode.install_paid_model_guard()
    for model in (
        "openrouter/openai/gpt-4.1-mini",  # structure_output's built-in default
        "openai/gpt-4o-mini",  # forecasting-tools default parser/summarizer
        "gpt-4o",
        "openrouter/deepseek/deepseek-r1-0528",  # paid twin of a free model
        "deepseek/deepseek-r1-0528:free",  # free but not via OpenRouter
    ):
        llm = GeneralLlm(model=model, allowed_tries=1)
        with pytest.raises(free_mode.PaidModelBlockedError):
            asyncio.run(llm.invoke("hi"))
        with pytest.raises(free_mode.PaidModelBlockedError):
            asyncio.run(litellm.acompletion(model=model, messages=[]))
    assert sent == []
    free_mode.install_paid_model_guard()  # idempotent: no double wrapping
    assert not getattr(general_llm.acompletion.__wrapped__, "_free_mode_paid_guard", False)


def test_free_llm_refuses_non_free_ids():
    budget = free_mode.CallBudget(5, 0)
    with pytest.raises(free_models.NotAFreeModelError):
        free_mode.FreeOpenRouterLlm("openai/gpt-4o", budget=budget)


# ---------------------------------------------------------------- end to end


def test_end_to_end_only_free_models_and_expected_request_count(monkeypatch, fake_llm):
    bot = build_free_bot(monkeypatch, MAX_LLM_CALLS_PER_RUN=20)
    reports = run(bot, [numeric(3, 30), multiple_choice(2, 20), binary(1, 10)])
    assert all(isinstance(r, ForecastReport) for r in reports), reports
    # closing soonest first
    assert [r.question.id_of_post for r in reports] == [1, 2, 3]
    assert reports[0].prediction == pytest.approx(0.37)
    # 3 predictions each: binary 3 (parsed locally) + MC 3x2 + numeric 3x2 = 15
    assert bot._free_runtime.budget.calls_made == len(fake_llm.models) == 15
    assert fake_llm.parser_calls() == 6
    assert all(
        m.startswith("openrouter/") and m.endswith(":free") for m in fake_llm.models
    )
    assert free_mode.NO_RESEARCH_NOTE in reports[0].explanation
    assert free_mode.exit_code_for_run(reports, strict=True) == 0


def test_publishing_posts_forecast_and_comment(monkeypatch, fake_llm):
    bot = build_free_bot(monkeypatch, publish=True)
    posted = {}
    monkeypatch.setattr(
        bot.metaculus_client,
        "post_binary_question_prediction",
        lambda question_id, prediction: posted.setdefault("prediction", prediction),
    )
    monkeypatch.setattr(
        bot.metaculus_client,
        "post_question_comment",
        lambda post_id, comment: posted.setdefault("comment", comment),
    )
    reports = run(bot, [binary(7)])
    assert isinstance(reports[0], ForecastReport), reports
    assert posted["prediction"] == pytest.approx(0.37)
    assert "Probability: 37%" in posted["comment"]


def test_budget_caps_requests_and_defers_the_rest(monkeypatch, fake_llm):
    bot = build_free_bot(monkeypatch)  # defaults: 6 requests, 6 questions
    questions = [binary(n, hours_to_close=100 - n) for n in range(1, 11)]
    reports = run(bot, questions)
    # backlog -> 1 prediction each -> 6 questions, the 6 closing soonest
    assert [r.question.id_of_post for r in reports] == [10, 9, 8, 7, 6, 5]
    assert all(isinstance(r, ForecastReport) for r in reports)
    assert bot._free_runtime.budget.calls_made == 6
    assert len(fake_llm.models) == 6
    assert bot.predictions_per_research_report == 3  # restored after the run


def test_mixed_backlog_stays_inside_budget(monkeypatch, fake_llm):
    bot = build_free_bot(monkeypatch)
    questions = [numeric(1, 1), numeric(2, 2), numeric(3, 3), binary(4, 4), binary(5, 5)]
    reports = run(bot, questions)
    # numeric costs 2 at 1 prediction: 2+2+2 = 6 -> binaries wait for next run
    assert [r.question.id_of_post for r in reports] == [1, 2, 3]
    assert len(fake_llm.models) == 6


def test_already_forecasted_questions_are_skipped(monkeypatch, fake_llm):
    bot = build_free_bot(monkeypatch)
    reports = run(bot, [binary(1, forecasted=True), binary(2)])
    assert [r.question.id_of_post for r in reports] == [2]


def test_question_cap(monkeypatch, fake_llm):
    bot = build_free_bot(monkeypatch, MAX_QUESTIONS_PER_RUN=2, MAX_LLM_CALLS_PER_RUN=50)
    reports = run(bot, [binary(n, n) for n in range(1, 6)])
    assert [r.question.id_of_post for r in reports] == [1, 2]


def test_daily_limit_429_stops_requests_and_exits_zero(monkeypatch):
    error = litellm.RateLimitError(
        message='OpenrouterException - {"error":{"message":"Rate limit exceeded: '
        'free-models-per-day. Add 10 credits to unlock 1000 free model requests '
        'per day","code":429}}',
        llm_provider="openrouter",
        model="openrouter/deepseek/deepseek-r1-0528:free",
    )
    fake = FakeOpenRouter(fail_with=error)
    monkeypatch.setattr(general_llm, "acompletion", free_mode._guarded(fake))
    bot = build_free_bot(monkeypatch)
    reports = run(bot, [binary(n, n) for n in range(1, 5)])
    assert len(fake.models) == 1  # nothing else is sent after the daily-limit 429
    assert bot._free_runtime.budget.daily_limit_hit
    assert len(reports) == 1 and isinstance(reports[0], BaseException)
    assert free_mode.is_rate_limit_error(reports[0])
    assert free_mode.exit_code_for_run(reports, strict=False) == 0
    assert free_mode.exit_code_for_run(reports, strict=True) == 1
    bot.log_report_summary(reports, raise_errors=False)  # must not raise


def test_unavailable_model_switches_to_backup(monkeypatch):
    gone = litellm.NotFoundError(
        message='OpenrouterException - {"error":{"message":"No endpoints found for '
        'deepseek/deepseek-r1-0528:free.","code":404}}',
        llm_provider="openrouter",
        model="openrouter/deepseek/deepseek-r1-0528:free",
    )
    fake = FakeOpenRouter()
    ok = fake.__call__

    async def first_model_gone(*args, **kwargs):
        if kwargs["model"].endswith("deepseek-r1-0528:free"):
            fake.models.append(kwargs["model"])
            raise gone
        return await ok(*args, **kwargs)

    monkeypatch.setattr(general_llm, "acompletion", free_mode._guarded(first_model_gone))
    bot = build_free_bot(monkeypatch)
    reports = run(bot, [binary(1)])
    assert isinstance(reports[0], ForecastReport), reports
    assert fake.models[0] == "openrouter/deepseek/deepseek-r1-0528:free"
    assert fake.models[1] == "openrouter/qwen/qwen3-235b-a22b:free"
    assert all(m.endswith(":free") for m in fake.models)


def test_data_policy_error_is_hard_and_not_retried_on_other_models():
    error = litellm.NotFoundError(
        message="OpenrouterException - No endpoints found matching your data policy "
        "(Free model publication). Configure: https://openrouter.ai/settings/privacy",
        llm_provider="openrouter",
        model="openrouter/deepseek/deepseek-r1-0528:free",
    )
    assert not free_mode.is_model_unavailable_error(error)
    assert free_mode.is_hard_error(error)
    assert free_mode.exit_code_for_run([_wrapped(error)], strict=False) == 1


# ---------------------------------------------------------------- exit codes


def _wrapped(inner):
    try:
        try:
            raise inner
        except Exception as error:
            raise RuntimeError("Error while processing question url: 'x'") from error
    except RuntimeError as outer:
        return outer


def test_exit_codes():
    rate = litellm.RateLimitError(
        message="Rate limit exceeded: free-models-per-min.",
        llm_provider="openrouter",
        model="openrouter/x/y:free",
    )
    auth = litellm.AuthenticationError(
        message='OpenrouterException - {"error":{"message":"User not found.","code":401}}',
        llm_provider="openrouter",
        model="openrouter/x/y:free",
    )
    parse_problem = ValueError("Could not parse: the model wrote 'unauthorized data policy'")
    assert free_mode.exit_code_for_run([], strict=False) == 0
    assert free_mode.exit_code_for_run([_wrapped(rate)], strict=False) == 0
    assert free_mode.exit_code_for_run(
        [ExceptionGroup("errors", [rate, parse_problem])], strict=False
    ) == 0
    # model text inside a parse error never counts as an API error
    assert not free_mode.is_hard_error(parse_problem)
    # one bad question among successful ones: retried next run, job stays green
    a_forecast = object()
    assert free_mode.exit_code_for_run([a_forecast, _wrapped(parse_problem)], strict=False) == 0
    assert free_mode.exit_code_for_run([a_forecast, _wrapped(parse_problem)], strict=True) == 1
    assert free_mode.exit_code_for_run([_wrapped(auth)], strict=False) == 1
    assert free_mode.exit_code_for_run(
        [_wrapped(free_mode.PaidModelBlockedError("x"))], strict=False
    ) == 1
    assert free_mode.exit_code_for_run([_wrapped(AttributeError("bug"))], strict=False) == 1


def test_finish_run_writes_step_summary(monkeypatch, tmp_path):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    runtime = free_mode.FreeRuntime(
        settings=free_mode.FreeSettings(),
        budget=free_mode.CallBudget(6, 0),
        default_model="openrouter/a/b:free",
        parser_model="openrouter/c/d:free",
    )
    assert free_mode.finish_run(runtime, [], strict=False) == 0
    text = summary.read_text("utf-8")
    assert "0 forecast(s)" in text and "0/6 LLM requests" in text
    assert "key" not in text.lower()


# ---------------------------------------------------------------- local parsing


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Reasoning...\nProbability: 37%", 0.37),
        ("Reasoning...\n**Probability:** 5.5%\n", 0.055),
        ("Reasoning...\nProbability: ~12 %", 0.12),
        ("Reasoning...\nprobability = 100%", 1.0),
        ("No final line here, maybe 30% or so", None),
        ("Probability: 40%\n" + "more text " * 100, None),  # not the last thing
        ("The question has already resolved.\nProbability: 99%", None),
        ("Probability: 120%", None),
        # review M1: an intermediate percentage must never be submitted
        ("Base rate probability: 60% for similar events.\n\n"
         "**Final Probability Estimate: 22%**", None),
        ("(c) scenario probability: 60%.\n(d) scenario probability: 15%.\n"
         "Overall I land at 25%.", None),
        ("Probability: 30%\nActually, reconsidering, I would put it at 20%.", None),
        ("Probability: 30% (up from a 10% prior)", None),
        ("", None),
    ],
)
def test_parse_final_probability(text, expected):
    result = free_mode.parse_final_probability(text)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


def test_binary_falls_back_to_llm_parser_when_line_missing(monkeypatch, fake_llm):
    monkeypatch.setattr(
        FakeOpenRouter,
        "answer",
        staticmethod(
            lambda prompt: '{"prediction_in_decimal": 0.42}'
            if "convert the text" in prompt
            else "I think it is fairly unlikely, around forty-two percent."
        ),
    )
    bot = build_free_bot(monkeypatch, PREDICTIONS_PER_REPORT=1)
    reports = run(bot, [binary(1)])
    assert reports[0].prediction == pytest.approx(0.42)
    assert fake_llm.parser_calls() == 1
    assert len(fake_llm.models) == 2


# ---------------------------------------------------------------- review fixes


def test_intermediate_percentage_goes_to_the_llm_parser(monkeypatch, fake_llm):
    """M1: the reviewer's case ends up as 0.22 (parser), never 0.60."""
    text = "Base rate probability: 60% for similar events.\n\n**Final Probability Estimate: 22%**"
    monkeypatch.setattr(
        FakeOpenRouter,
        "answer",
        staticmethod(
            lambda prompt: '{"prediction_in_decimal": 0.22}'
            if "convert the text" in prompt
            else text
        ),
    )
    bot = build_free_bot(monkeypatch, PREDICTIONS_PER_REPORT=1)
    reports = run(bot, [binary(1)])
    assert reports[0].prediction == pytest.approx(0.22)
    assert fake_llm.parser_calls() == 1


def test_intermediate_percentage_without_budget_is_skipped_not_guessed(monkeypatch, fake_llm):
    """M1: no budget left for the parser -> no forecast at all (never 0.60)."""
    monkeypatch.setattr(
        FakeOpenRouter,
        "answer",
        staticmethod(lambda prompt: "Base rate probability: 60%.\nI'd say 22%"),
    )
    bot = build_free_bot(monkeypatch, MAX_LLM_CALLS_PER_RUN=1)
    reports = run(bot, [binary(1)])
    assert len(reports) == 1 and isinstance(reports[0], BaseException)
    assert len(fake_llm.models) == 1  # the parser call was refused by the budget


def _conditional(parent, child, live_parent=False, live_child=False):
    from forecasting_tools import ConditionalQuestion
    from forecasting_tools.data_models.questions import BinaryTimestampedPrediction

    def live(question):
        return question.model_copy(
            update={
                "previous_forecasts": [
                    BinaryTimestampedPrediction(
                        prediction_in_decimal=0.5,
                        timestamp=NOW - timedelta(days=1),
                        timestamp_end=None,
                    )
                ]
            }
        )

    if live_parent:
        parent = live(parent)
    if live_child:
        child = live(child)
    return ConditionalQuestion(
        question_text="conditional",
        id_of_post=900,
        id_of_question=9000,
        page_url="https://www.metaculus.com/questions/900/",
        close_time=NOW + timedelta(hours=1),
        parent=parent,
        child=child,
        question_yes=child,
        question_no=child,
        already_forecasted=False,
    )


def test_conditional_estimate_counts_reaffirmed_parts_as_free():
    """S1: parent/child with a live forecast are reaffirmed at 0 requests."""
    big = _conditional(binary(1), numeric(2))
    assert free_mode.estimate_calls(big, 1) == 1 + 3 * 2  # 7 > 6
    small = _conditional(binary(1), numeric(2), live_parent=True, live_child=True)
    assert free_mode.estimate_calls(small, 1) == 2 * 2  # only yes/no versions
    both_binary = _conditional(binary(1), binary(2))
    assert free_mode.estimate_calls(both_binary, 1) == 4


def test_too_big_question_is_skipped_with_log_and_others_still_run(
    monkeypatch, fake_llm, caplog
):
    """S1: never-fitting questions are skipped explicitly, not starving others."""
    bot = build_free_bot(monkeypatch)  # 6 requests per run
    big = _conditional(binary(1), numeric(2))  # 7 requests even at P=1
    with caplog.at_level("WARNING"):
        reports = run(bot, [big, binary(3, 2)])
    assert [r.question.id_of_post for r in reports] == [3]
    assert any("SKIPPED" in m and "MAX_LLM_CALLS_PER_RUN" in m for m in caplog.messages)


def test_failing_question_costs_little_and_does_not_starve_the_queue(monkeypatch, fake_llm):
    """S3: an unparsable question uses 2 requests at most, then the next ones run."""
    normal_answer = FakeOpenRouter.answer

    def answer(prompt):
        if "convert the text" in prompt and "I refuse" in prompt:
            return "<<REQUESTED TYPE WAS NOT FOUND IN TEXT>>"
        if "How many 1?" in prompt:
            return "I refuse to give percentiles."
        return normal_answer(prompt)

    monkeypatch.setattr(FakeOpenRouter, "answer", staticmethod(answer))
    bot = build_free_bot(monkeypatch)
    duplicate = numeric(1, 1)
    reports = run(bot, [numeric(1, 1), duplicate, binary(2, 2), binary(3, 3)])
    assert isinstance(reports[0], BaseException)  # the bad question, tried once
    assert [r.question.id_of_post for r in reports[1:]] == [2, 3]
    bad_requests = sum("How many 1?" in p for p in fake_llm.prompts)
    assert bad_requests == 2  # 1 reasoning + 1 parser try (not 1 + 3)
    assert free_mode.exit_code_for_run(reports, strict=False) == 0  # others worked


def test_switching_model_updates_timeout():
    """S4: 180 s for normal models, 300 s for reasoning ones, also after a switch."""
    llm = free_mode.FreeOpenRouterLlm(
        "meta-llama/llama-3.3-70b-instruct:free",
        budget=free_mode.CallBudget(5, 0),
        backup_model_ids=["deepseek/deepseek-r1-0528:free"],
        timeout=free_mode.timeout_for("meta-llama/llama-3.3-70b-instruct:free"),
    )
    assert llm.litellm_kwargs["timeout"] == 180
    llm._switch_model("openrouter/deepseek/deepseek-r1-0528:free")
    assert llm.litellm_kwargs["timeout"] == 300


class _FakeClient:
    def __init__(self, failing):
        self.failing = failing

    def get_all_open_questions_from_tournament(self, tournament_id):
        if tournament_id == self.failing:
            raise ConnectionError("Metaculus is down for this one")
        return [binary(1 if tournament_id == "seasonal" else 2)]


def test_one_tournament_failing_does_not_block_the_other():
    """S5: seasonal and MiniBench are loaded separately."""
    questions, errors = free_mode.load_open_questions(
        _FakeClient(failing="minibench"), ["seasonal", "minibench"]
    )
    assert [q.id_of_post for q in questions] == [1]
    assert len(errors) == 1 and "minibench" in str(errors[0])
    assert isinstance(errors[0].__cause__, ConnectionError)
    # a temporary fetch problem next to real forecasts keeps the job green
    assert free_mode.exit_code_for_run([object(), *errors], strict=False) == 0


@pytest.mark.parametrize(
    "error",
    [
        TypeError("unsupported operand"),
        KeyError("predicted_options"),
        ValueError("Expected at least 0.5 successful predictions"),
        AttributeError("'NoneType' object has no attribute 'x'"),
    ],
)
def test_code_errors_on_every_question_exit_one(error):
    """S2: a green run with zero forecasts must not happen for code errors."""
    assert free_mode.exit_code_for_run([_wrapped(error)], strict=False) == 1
    assert free_mode.exit_code_for_run([_wrapped(error), _wrapped(error)], strict=False) == 1


def _api_error(cls, message, **extra):
    return cls(message=message, llm_provider="openrouter", model="openrouter/x/y:free", **extra)


@pytest.mark.parametrize(
    "error",
    [
        _api_error(litellm.RateLimitError, "Rate limit exceeded: free-models-per-day."),
        _api_error(litellm.Timeout, "Request timed out"),
        _api_error(litellm.InternalServerError, "OpenrouterException - Internal Server Error"),
        _api_error(litellm.APIConnectionError, "Connection reset by peer"),
        _api_error(
            litellm.APIError,
            "LiteLLM: provider returned 'choices' that is not a list (NoneType).",
            status_code=200,
        ),
        _api_error(litellm.APIError, "OpenrouterException - Provider returned error", status_code=200),
        _api_error(
            litellm.NotFoundError,
            "OpenrouterException - No endpoints found for x/y:free.",
        ),
        AssertionError(
            "Answer is not a string and is of type: <class 'NoneType'>. Answer: None. "
            "Finish reason: length"
        ),
        RuntimeError("LLM answer is an empty string. The model was openrouter/x/y:free"),
        _api_error(
            litellm.PermissionDeniedError,
            'OpenrouterException - {"error":{"message":"Input was flagged by moderation",'
            '"code":403}}',
            response=httpx.Response(403, request=httpx.Request("POST", "https://x")),
        ),
    ],
)
def test_temporary_problems_on_every_question_exit_zero(error):
    """S2/N3: rate limits and flaky free models never turn the cron red."""
    assert free_mode.is_transient_error(_wrapped(error))
    assert not free_mode.is_hard_error(_wrapped(error))
    assert free_mode.exit_code_for_run([_wrapped(error)] * 3, strict=False) == 0


def test_openrouter_moderation_403_is_not_hard():
    """N3: a flagged question must not make every cron run red."""
    error = _api_error(
        litellm.BadRequestError,
        'OpenrouterException - {"error":{"message":"Input was flagged by moderation",'
        '"code":403}}',
    )
    assert not free_mode.is_hard_error(_wrapped(error))
    assert free_mode.is_transient_error(_wrapped(error))
    assert free_mode.exit_code_for_run([_wrapped(error)], strict=False) == 0


def test_daily_limit_stops_already_spaced_requests(monkeypatch):
    """N1: with 3 predictions in flight, only the first request is sent."""
    error = _api_error(litellm.RateLimitError, "Rate limit exceeded: free-models-per-day.")
    fake = FakeOpenRouter(fail_with=error)
    monkeypatch.setattr(general_llm, "acompletion", free_mode._guarded(fake))
    bot = build_free_bot(monkeypatch, MIN_SECONDS_BETWEEN_CALLS="0.05")
    reports = run(bot, [binary(1)])  # one question -> 3 predictions
    assert bot.predictions_per_research_report == 3
    assert len(fake.models) == 1
    assert isinstance(reports[0], BaseException)
    assert bot._free_runtime.budget.calls_made == 1


def test_permission_denied_without_moderation_is_still_hard():
    error = _api_error(
        litellm.PermissionDeniedError,
        'OpenrouterException - {"error":{"message":"Key disabled","code":403}}',
        response=httpx.Response(403, request=httpx.Request("POST", "https://x")),
    )
    assert free_mode.is_hard_error(_wrapped(error))
    assert free_mode.exit_code_for_run([_wrapped(error)], strict=False) == 1

"""
Free mode for the Metaculus template bot (BOT_MODE=free, the default in this fork).

Goal: forecast with OpenRouter `:free` models only ($0) and stay inside the
free tier (50 requests/day and 20 requests/minute per account).

What lives here (main.py only calls into it):
- `FreeSettings`: knobs read from env vars (safe defaults for 50 requests/day).
- `CallBudget`: per-run cap on LLM requests + spacing between requests.
- `FreeOpenRouterLlm`: GeneralLlm that only accepts `:free` ids, counts every
  real request against the budget and moves to a backup free model if the
  chosen one disappeared.
- `install_paid_model_guard()`: wraps litellm so that ANY call (from any code
  path) to a model that is not `openrouter/...:free` raises before a request
  is sent.
- `forecast_within_budget()`: forecasts questions one at a time, closing
  soonest first, until the per-run budget or question cap is used up. The rest
  stay unforecasted and are picked up by the next scheduled run.
- `exit_code_for_run()`: rate limits and free-model hiccups exit 0 (questions
  are retried next run); auth/payment/config errors and code bugs exit 1.

BOT_MODE=paid keeps the original template behaviour (none of this is used).
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Iterator, Sequence

import free_models

logger = logging.getLogger(__name__)

VALID_BOT_MODES = ("free", "paid")

NO_RESEARCH_NOTE = (
    "No external research was run for this forecast (free mode: no search tools). "
    "Use the question background and your own knowledge, and keep in mind that "
    "your training data may be out of date."
)


# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------


def bot_mode() -> str:
    """'free' (default when unset) or 'paid'. Anything else is a config error."""
    mode = (os.getenv("BOT_MODE") or "free").strip().lower()
    if mode not in VALID_BOT_MODES:
        raise ValueError(f"BOT_MODE must be one of {VALID_BOT_MODES}, got {mode!r}")
    return mode


def is_free_mode() -> bool:
    return bot_mode() == "free"


def _env_number(name: str, default: float, minimum: float, cast: type = int) -> Any:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return cast(default)
    try:
        value = cast(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class FreeSettings:
    # Upper bound; drops to 1 automatically when there is a backlog (see
    # forecast_within_budget) so that coverage wins over ensembling.
    predictions_per_report: int = 3
    # 8 scheduled runs/day x 6 requests = 48/day. The free limit (50/day) is per
    # OpenRouter account, so manual runs share it; overflow is a temporary 429.
    max_llm_calls_per_run: int = 6
    max_questions_per_run: int = 6
    # 20 requests/minute limit -> keep >= 3 s between requests (4 s = 15/min).
    min_seconds_between_calls: float = 4.0

    @classmethod
    def from_env(cls) -> "FreeSettings":
        return cls(
            predictions_per_report=_env_number(
                "PREDICTIONS_PER_REPORT", cls.predictions_per_report, 1
            ),
            max_llm_calls_per_run=_env_number(
                "MAX_LLM_CALLS_PER_RUN", cls.max_llm_calls_per_run, 1
            ),
            max_questions_per_run=_env_number(
                "MAX_QUESTIONS_PER_RUN", cls.max_questions_per_run, 1
            ),
            min_seconds_between_calls=_env_number(
                "MIN_SECONDS_BETWEEN_CALLS", cls.min_seconds_between_calls, 0, float
            ),
        )


def require_openrouter_key() -> None:
    """Free mode only works with an OpenRouter key: stop early and clearly."""
    key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not key or key == "REPLACE_ME":
        print(
            "❌  BOT_MODE=free needs the OPENROUTER_API_KEY secret "
            "(Settings -> Secrets and variables -> Actions)."
        )
        sys.exit(1)


# ----------------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------------


class FreeBudgetExhausted(RuntimeError):
    """The per-run request budget (or OpenRouter's daily free quota) is used up."""


class PaidModelBlockedError(RuntimeError):
    """A call to a non-free model was attempted and blocked before sending."""


def iter_exception_chain(error: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    stack: list[BaseException] = [error]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        stack.extend(getattr(current, "exceptions", None) or [])  # ExceptionGroup
        if current.__cause__ is not None:
            stack.append(current.__cause__)
        if current.__context__ is not None:
            stack.append(current.__context__)


# Only messages from HTTP/LLM client exceptions are searched for keywords, so a
# model's own words (e.g. inside a parse error) can never change the verdict.
_API_MODULE_PREFIXES = (
    "litellm", "openai", "httpx", "httpcore", "requests", "urllib3", "aiohttp",
)


def _chain_text(error: BaseException) -> str:
    return " | ".join(
        f"{type(e).__name__}: {e}"
        for e in iter_exception_chain(error)
        if type(e).__module__.split(".")[0] in _API_MODULE_PREFIXES
    ).lower()


def _chain_has_type(error: BaseException, type_names: Sequence[str]) -> bool:
    return any(type(e).__name__ in type_names for e in iter_exception_chain(error))


_RATE_LIMIT_TEXT = re.compile(
    r"rate[ _-]?limit|ratelimit|too many requests|"
    r"\b(status|error)[ _]?code\W{0,3}429\b|\"code\"\s*:\s*429"
)
_DAILY_LIMIT_TEXT = re.compile(r"per[ _-]?day|daily|free-models-per-day")
_HARD_TEXT = re.compile(
    r"\b(status|error)[ _]?code\W{0,3}(401|402|403)\b|\"code\"\s*:\s*(401|402|403)|"
    r"\b(401|402|403) client error|"
    r"payment required|insufficient (credits|balance)|negative balance|"
    r"invalid api key|no auth credentials|user not found|unauthori[sz]ed|"
    r"data policy|settings/privacy"
)
# A 403 from OpenRouter moderation is about ONE question, not the account.
_MODERATION_TEXT = re.compile(r"moderation|flagged")
# Temporary problems of the free endpoints / network (retried next run).
_TRANSIENT_TEXT = re.compile(
    r"\b(status|error)[ _]?code\W{0,3}5\d\d\b|\"code\"\s*:\s*5\d\d|"
    r"\b5\d\d server error|provider returned|timed? ?out|temporarily|"
    r"overloaded|service unavailable|bad gateway|connection (reset|refused|aborted)"
)
_TRANSIENT_TYPES = (
    "Timeout",
    "TimeoutError",
    "APITimeoutError",
    "APIConnectionError",
    "InternalServerError",
    "ServiceUnavailableError",
    "BadGatewayError",
    "ContentPolicyViolationError",
    "ConnectionError",
    "ConnectTimeout",
    "ReadTimeout",
)
# Messages written by forecasting-tools itself when a free model answers with
# nothing usable (not model text, so safe to match on any exception type).
_EMPTY_ANSWER_TEXT = re.compile(
    r"llm answer is an empty string|answer is not a string and is of type"
)
_MODEL_GONE_TEXT = re.compile(
    r"no endpoints found|not a valid model id|model not found|"
    r"model .{0,80} does not exist|no allowed providers"
)
_HARD_TYPES = (
    "AuthenticationError",
    "PermissionDeniedError",
    "PaidModelBlockedError",
    "NotAFreeModelError",
    "AttributeError",
    "NameError",
    "ImportError",
    "ModuleNotFoundError",
    "SyntaxError",
)


def is_rate_limit_error(error: BaseException) -> bool:
    return _chain_has_type(
        error, ("RateLimitError", "RouterRateLimitError", "FreeBudgetExhausted")
    ) or bool(_RATE_LIMIT_TEXT.search(_chain_text(error)))


def is_daily_limit_error(error: BaseException) -> bool:
    return is_rate_limit_error(error) and bool(
        _DAILY_LIMIT_TEXT.search(_chain_text(error))
    )


def is_hard_error(error: BaseException) -> bool:
    """Needs the owner: bad key, payment, privacy setting, paid model, code bug."""
    text = _chain_text(error)
    if _MODERATION_TEXT.search(text):
        # A 403 "flagged by moderation" concerns one question, not the account.
        other_hard = tuple(t for t in _HARD_TYPES if t != "PermissionDeniedError")
        return _chain_has_type(error, other_hard)
    return _chain_has_type(error, _HARD_TYPES) or bool(_HARD_TEXT.search(text))


def is_transient_error(error: BaseException) -> bool:
    """Rate limit or a flaky free model/network: fine to just retry next run."""
    if is_rate_limit_error(error) or _chain_has_type(error, _TRANSIENT_TYPES):
        return True
    text = _chain_text(error)
    if _TRANSIENT_TEXT.search(text) or _MODERATION_TEXT.search(text):
        return True
    if is_model_unavailable_error(error):  # every backup model gone too
        return True
    everything = " | ".join(str(e) for e in iter_exception_chain(error)).lower()
    return bool(_EMPTY_ANSWER_TEXT.search(everything))


def is_model_unavailable_error(error: BaseException) -> bool:
    text = _chain_text(error)
    if "data policy" in text:  # account setting, switching models will not help
        return False
    return bool(_MODEL_GONE_TEXT.search(text))


# ----------------------------------------------------------------------------
# Budget + spacing (shared by every LLM object in the run)
# ----------------------------------------------------------------------------


class CallBudget:
    """
    Counts real LLM requests (retries included) and spaces them out.
    Loop-agnostic on purpose: main.py may use several asyncio.run() calls.
    """

    def __init__(self, max_calls: int, min_seconds_between_calls: float) -> None:
        self.max_calls = max_calls
        self.min_seconds_between_calls = min_seconds_between_calls
        self.calls_made = 0
        self.daily_limit_hit = False
        self._next_slot = 0.0

    @property
    def remaining(self) -> int:
        if self.daily_limit_hit:
            return 0
        return max(0, self.max_calls - self.calls_made)

    async def acquire(self) -> None:
        if self.daily_limit_hit:
            raise FreeBudgetExhausted(
                "OpenRouter free daily request limit reached earlier in this run"
            )
        if self.calls_made >= self.max_calls:
            raise FreeBudgetExhausted(
                f"Per-run LLM request budget used up ({self.max_calls}); "
                "remaining work waits for the next run"
            )
        self.calls_made += 1  # counted before awaiting: no overshoot when concurrent
        now = time.monotonic()
        slot = max(now, self._next_slot)
        self._next_slot = slot + self.min_seconds_between_calls
        if slot > now:
            await asyncio.sleep(slot - now)
            if self.daily_limit_hit:  # another request hit the daily 429 meanwhile
                self.calls_made -= 1  # nothing was sent
                raise FreeBudgetExhausted(
                    "OpenRouter free daily request limit reached earlier in this run"
                )


# ----------------------------------------------------------------------------
# LLM wrapper + global guard
# ----------------------------------------------------------------------------

from forecasting_tools import GeneralLlm  # noqa: E402  (after the pure helpers)


REASONING_TIMEOUT_SECONDS = 300
DEFAULT_TIMEOUT_SECONDS = 180  # free endpoints queue; a timeout wastes a request


def timeout_for(model_id: str) -> int:
    if free_models.is_reasoning_model(model_id):
        return REASONING_TIMEOUT_SECONDS
    return DEFAULT_TIMEOUT_SECONDS


def _openrouter_free(model_id: str) -> str:
    return "openrouter/" + free_models.assert_free(model_id.removeprefix("openrouter/"))


class FreeOpenRouterLlm(GeneralLlm):
    """GeneralLlm restricted to `openrouter/...:free`, metered by a CallBudget."""

    def __init__(
        self,
        model_id: str,
        *,
        budget: CallBudget,
        backup_model_ids: Sequence[str] = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(model=_openrouter_free(model_id), **kwargs)
        self._budget = budget
        self._backups = [
            _openrouter_free(m) for m in backup_model_ids
            if _openrouter_free(m) != self.model
        ]

    def _switch_model(self, model: str) -> None:
        logger.warning(f"Free model {self.model} is unavailable; switching to {model}")
        self.model = model
        self._litellm_model = model
        self.litellm_kwargs["model"] = model
        self.litellm_kwargs["timeout"] = timeout_for(model)

    async def _mockable_direct_call_to_model(self, prompt: Any) -> Any:
        while True:
            _openrouter_free(self.model)  # re-checked before every request
            await self._budget.acquire()
            try:
                return await super()._mockable_direct_call_to_model(prompt)
            except Exception as error:
                if is_daily_limit_error(error):
                    self._budget.daily_limit_hit = True
                    raise FreeBudgetExhausted(
                        f"OpenRouter free daily limit reached: {error}"
                    ) from error
                if is_model_unavailable_error(error) and self._backups:
                    self._switch_model(self._backups.pop(0))
                    continue
                raise


_GUARD_ATTRIBUTE = "_free_mode_paid_guard"


def _is_allowed_model(model: Any) -> bool:
    return (
        isinstance(model, str)
        and model.startswith("openrouter/")
        and model.endswith(free_models.FREE_SUFFIX)
    )


def _guarded(function: Callable[..., Any]) -> Callable[..., Any]:
    if getattr(function, _GUARD_ATTRIBUTE, False):
        return function

    def check(args: tuple, kwargs: dict) -> None:
        model = kwargs.get("model", args[0] if args else None)
        if not _is_allowed_model(model):
            raise PaidModelBlockedError(
                f"Free mode blocked a call to {model!r} (only openrouter/...:free)"
            )

    # A plain wrapper works for sync and async functions alike: the check runs
    # before the call, and an async function's coroutine is returned unchanged.
    @functools.wraps(function)  # keeps inspect.signature() (GeneralLlm uses it)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        check(args, kwargs)
        return function(*args, **kwargs)

    setattr(wrapper, _GUARD_ATTRIBUTE, True)
    return wrapper


def install_paid_model_guard() -> None:
    """Make every litellm entry point refuse non-free models (idempotent)."""
    import litellm
    from forecasting_tools.ai_models import general_llm

    for module in (litellm, general_llm):
        for name in ("acompletion", "completion", "aresponses", "responses"):
            if hasattr(module, name):
                setattr(module, name, _guarded(getattr(module, name)))


def build_free_llms(
    budget: CallBudget, models_json: dict | None
) -> dict[str, GeneralLlm | str]:
    """llms= dict for the bot: every role is a `:free` model or no research."""
    main_ids = free_models.free_model_candidates(models_json)
    parser_ids = free_models.free_model_candidates(
        models_json,
        preferences=free_models.PARSER_PREFERENCES,
        env_var="FREE_PARSER_MODEL",
    )

    def make(ids: list[str]) -> FreeOpenRouterLlm:
        return FreeOpenRouterLlm(
            ids[0],
            budget=budget,
            backup_model_ids=ids[1:],
            temperature=0.3,
            timeout=timeout_for(ids[0]),
            allowed_tries=2,
        )

    parser = make(parser_ids)
    return {
        "default": make(main_ids),
        "summarizer": parser,  # unused: summaries are disabled in free mode
        "researcher": "no_research",  # no search APIs; saves requests
        "parser": parser,
    }


def assert_all_llms_free(llms: dict[str, Any]) -> None:
    for role, llm in llms.items():
        if isinstance(llm, str) and llm == "no_research":
            continue
        model = llm.model if isinstance(llm, GeneralLlm) else llm
        if not _is_allowed_model(model):
            raise PaidModelBlockedError(f"LLM role {role!r} is not free: {model!r}")


# ----------------------------------------------------------------------------
# Cheap parsing of binary answers (saves one request per binary forecast)
# ----------------------------------------------------------------------------

_FINAL_PROBABILITY = re.compile(
    r"probability\s*\**\s*[:=]\s*\**\s*[~≈]?\s*(\d{1,3}(?:\.\d+)?)\s*%", re.IGNORECASE
)
_ANY_PERCENT = re.compile(r"\d\s*%")
_ALREADY_RESOLVED = re.compile(r"already\s+(been\s+)?resolved|has\s+already\s+resolved", re.I)


def parse_final_probability(text: str, tail_chars: int = 300) -> float | None:
    """
    Read the template's 'Probability: ZZ%' final line without an LLM.
    Returns a decimal in [0, 1], or None to fall back to the LLM parser.
    """
    if not text or _ALREADY_RESOLVED.search(text):
        return None
    stripped = text.rstrip()
    matches = list(_FINAL_PROBABILITY.finditer(stripped))
    if not matches:
        return None
    last = matches[-1]
    if len(stripped) - last.end() > tail_chars:
        return None  # not the last thing written
    if _ANY_PERCENT.search(stripped[last.end() :]):
        return None  # a later percentage exists: let the LLM parser decide
    value = float(last.group(1))
    if not 0 <= value <= 100:
        return None
    return value / 100


# ----------------------------------------------------------------------------
# Question selection: closing soonest first, inside the per-run budget
# ----------------------------------------------------------------------------


def _has_live_forecast(question: Any) -> bool:
    """main.py reaffirms (0 requests) a conditional's parent/child in this case."""
    previous = getattr(question, "previous_forecasts", None) or []
    if not previous:
        return False
    end = previous[-1].timestamp_end
    return end is None or end > datetime.now(timezone.utc)


def estimate_calls(question: Any, predictions: int) -> int:
    """Requests needed when nothing fails (1 reasoning + parser sample if needed)."""
    from forecasting_tools import BinaryQuestion, ConditionalQuestion

    if isinstance(question, ConditionalQuestion):
        parent = estimate_calls(question.parent, predictions)
        child = estimate_calls(question.child, predictions)
        return (
            (0 if _has_live_forecast(question.parent) else parent)
            + (0 if _has_live_forecast(question.child) else child)
            + 2 * child  # the "if yes" and "if no" versions of the child
        )
    if isinstance(question, BinaryQuestion):
        return predictions  # parsed locally (see parse_final_probability)
    return 2 * predictions  # reasoning + 1 parser call each


def order_by_closing_soonest(questions: Sequence[Any]) -> list[Any]:
    def key(question: Any) -> tuple:
        close = getattr(question, "close_time", None)
        timestamp = close.timestamp() if close is not None else float("inf")
        return (timestamp, getattr(question, "id_of_question", None) or 0)

    return sorted(questions, key=key)


async def forecast_within_budget(
    bot: Any,
    questions: Sequence[Any],
    return_exceptions: bool,
    forecast_fn: Callable[[list[Any], bool], Awaitable[list[Any]]],
    settings: FreeSettings,
    budget: CallBudget,
) -> list[Any]:
    """
    One question at a time, closing soonest first, each question at most once
    per run (a question that fails just waits for the next run; it cannot
    use the budget twice). Uses the full ensemble (settings.predictions_per_report)
    only when everything still waiting fits in this run's budget; with a
    backlog it drops to 1 prediction per question. Questions that do not fit
    are left for the next run (not errors). Questions that can never fit in
    one run (e.g. a big conditional) are skipped with a clear log line.
    """
    if bot.skip_previously_forecasted_questions:
        questions = [q for q in questions if not q.already_forecasted]
    pending: list[Any] = []
    seen: set[Any] = set()
    for question in order_by_closing_soonest(questions):
        key = getattr(question, "id_of_question", None) or id(question)
        if key in seen:
            continue  # at most one attempt per question per run
        seen.add(key)
        cheapest = estimate_calls(question, 1)
        if cheapest > budget.max_calls:
            logger.warning(
                f"[free mode] SKIPPED {question.page_url}: needs ~{cheapest} "
                f"requests, more than MAX_LLM_CALLS_PER_RUN={budget.max_calls}. "
                "Raise that variable to include questions like this one."
            )
            continue
        pending.append(question)

    original_predictions = bot.predictions_per_research_report
    reports: list[Any] = []
    attempted = 0
    deferred = 0
    try:
        for index, question in enumerate(pending):
            if attempted >= settings.max_questions_per_run:
                deferred += len(pending) - index
                break
            slots_left = settings.max_questions_per_run - attempted
            upcoming = pending[index : index + slots_left]
            predictions = settings.predictions_per_report
            if predictions > 1 and sum(
                estimate_calls(q, predictions) for q in upcoming
            ) > budget.remaining:
                predictions = 1
            if estimate_calls(question, predictions) > budget.remaining:
                deferred += 1
                continue
            bot.predictions_per_research_report = predictions
            logger.info(
                f"[free mode] Forecasting {question.page_url} "
                f"(close {question.close_time}, {predictions} prediction(s), "
                f"{budget.remaining} requests left this run)"
            )
            reports.extend(await forecast_fn([question], return_exceptions))
            attempted += 1
    finally:
        bot.predictions_per_research_report = original_predictions
    if deferred:
        logger.info(
            f"[free mode] {deferred} question(s) left for the next run "
            f"(per-run limits: {settings.max_questions_per_run} questions, "
            f"{budget.max_calls} requests)"
        )
    return reports


def load_open_questions(
    client: Any, tournament_ids: Sequence[Any]
) -> tuple[list[Any], list[BaseException]]:
    """
    Fetch each tournament on its own: one failing (e.g. MiniBench between
    rounds, a Metaculus 5xx) does not stop the others. Failures are returned
    as exceptions so they show up in the summary and the exit code.
    """
    questions: list[Any] = []
    errors: list[BaseException] = []
    for tournament_id in tournament_ids:
        try:
            questions.extend(client.get_all_open_questions_from_tournament(tournament_id))
        except Exception as error:
            logger.error(
                f"[free mode] Could not load questions from tournament "
                f"{tournament_id!r}: {type(error).__name__}: {error}"
            )
            wrapped = RuntimeError(f"Could not load tournament {tournament_id!r}")
            wrapped.__cause__ = error
            errors.append(wrapped)
    return questions, errors


# ----------------------------------------------------------------------------
# End of run
# ----------------------------------------------------------------------------


@dataclass
class FreeRuntime:
    """Attached to the bot in free mode (bot._free_runtime); None in paid mode."""

    settings: FreeSettings
    budget: CallBudget
    default_model: str
    parser_model: str


def exit_code_for_run(reports: Sequence[Any], strict: bool) -> int:
    """
    0 = fine, or only temporary problems (rate limit, flaky free model,
    network): the unforecasted questions are retried next run.
    1 = the owner must act: API key, payment, privacy setting, paid model,
    a code bug, or every attempted question failed for a non-temporary reason
    (a green run with zero forecasts must not happen).
    strict=True (manual test runs): any failure is 1.
    """
    errors = [r for r in reports if isinstance(r, BaseException)]
    forecasts = len(reports) - len(errors)
    hard = [e for e in errors if is_hard_error(e)]
    if hard:
        logger.error(
            f"[free mode] {len(hard)} error(s) need attention (API key, OpenRouter "
            "privacy setting for free models, payment, or a code bug). First: "
            f"{type(hard[0]).__name__}: {str(hard[0])[:300]}"
        )
        return 1
    if errors and strict:
        logger.error(f"[free mode] Test run had {len(errors)} failed question(s)")
        return 1
    lasting = [e for e in errors if not is_transient_error(e)]
    if lasting and forecasts == 0:
        logger.error(
            f"[free mode] Every attempted question failed ({len(errors)}) and not "
            "only because of rate limits or flaky free models: probably a code, "
            f"library or config problem. First: {type(lasting[0]).__name__}: "
            f"{str(lasting[0])[:300]}"
        )
        return 1
    if errors:
        logger.warning(
            f"[free mode] {len(errors)} question(s) not forecast this run "
            f"({len(errors) - len(lasting)} temporary, {len(lasting)} other). "
            "They stay unforecasted and are retried next run."
        )
    return 0


def write_step_summary(lines: Sequence[str]) -> None:
    """Append a short note to the GitHub Actions run page (no secrets in it)."""
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    except OSError as error:
        logger.warning(f"Could not write step summary: {error}")


def finish_run(runtime: FreeRuntime, reports: Sequence[Any], strict: bool) -> int:
    """Log + step summary for a free-mode run; returns the process exit code."""
    code = exit_code_for_run(reports, strict=strict)
    forecasts = sum(1 for r in reports if not isinstance(r, BaseException))
    errors = len(reports) - forecasts
    line = (
        f"Free mode: {forecasts} forecast(s), {errors} failed, "
        f"{runtime.budget.calls_made}/{runtime.budget.max_calls} LLM requests used, "
        f"model {runtime.default_model} (parser {runtime.parser_model})"
        + (", OpenRouter daily free limit reached" if runtime.budget.daily_limit_hit else "")
    )
    logger.info(f"[free mode] {line}")
    write_step_summary([f"### {'✅' if code == 0 else '❌'} {line}"])
    return code

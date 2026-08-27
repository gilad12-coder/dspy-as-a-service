"""Recover and explain the real cause behind DSPy's generic run failures.

DSPy's ``ParallelExecutor`` catches the exception each worker raises (an LLM
billing, auth, rate-limit, or network error), counts it, logs it to the
``dspy`` logger, and then re-raises only a generic
``"Execution cancelled due to errors or interruption."``. The actual cause
never reaches our exception handlers — it lives solely in the logs, which is
why a failed optimization card showed nothing actionable and users had to grep
the logs.

This module captures those ERROR log records during a run and rewrites the
generic message into one that names the real failure (and, for recognised
provider errors, prefixes a plain-language explanation), so the failure surface
tells the user what actually went wrong.
"""

from __future__ import annotations

import logging
import re
import threading

# DSPy / our own placeholders that carry no root-cause signal on their own.
_GENERIC_MARKERS = (
    "execution cancelled due to errors or interruption",
    "all model pairs failed",
)

# Ordered most-specific first: (matcher, plain-language explanation).
_LLM_ERROR_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"insufficient_quota|exceeded your current quota|check your plan and billing|billing", re.IGNORECASE),
        (
            "Billing/quota: the language-model provider rejected the request for billing "
            "reasons (insufficient quota or no remaining credit). Top up or check the "
            "provider account's billing, then retry."
        ),
    ),
    (
        re.compile(r"invalid[_ ]?api[_ ]?key|incorrect api key|authenticationerror|\b401\b|unauthorized", re.IGNORECASE),
        (
            "Authentication: the language-model provider rejected the API key. Verify the "
            "key configured for this provider."
        ),
    ),
    (
        re.compile(r"rate[_ ]?limit|\b429\b|too many requests", re.IGNORECASE),
        (
            "Rate limit: the language-model provider throttled the run (HTTP 429). Lower "
            "the parallelism or retry after a short wait."
        ),
    ),
    (
        re.compile(r"context[_ ]?length|maximum context|context window|reduce the length", re.IGNORECASE),
        (
            "Context window: a request exceeded the model's context limit. Shorten the "
            "inputs or choose a model with a larger context window."
        ),
    ),
    (
        re.compile(r"model.{0,30}(not found|does not exist)|invalid model|unknown model|no such model", re.IGNORECASE),
        (
            "Model unavailable: the requested model is not available for this provider or "
            "account."
        ),
    ),
    (
        re.compile(r"timed? ?out|timeout", re.IGNORECASE),
        (
            "Timeout: the language-model provider did not respond in time. Retry, or lower "
            "the parallelism."
        ),
    ),
    (
        re.compile(r"connection error|connection refused|failed to establish|name resolution|network is unreachable", re.IGNORECASE),
        "Network: could not reach the language-model provider (connection error).",
    ),
)

_TRACEBACK_HINT = re.compile(r"\.?\s*Set `provide_traceback=True` for traceback\.?\s*$")
_ERROR_HEAD = re.compile(r":\s+((?:litellm|openai)\.[\w.]*\w|[A-Za-z_]+(?:Error|Exception))")

_last_error: str | None = None
_lock = threading.Lock()


class LlmErrorCapture(logging.Handler):
    """Remember the most recent ERROR record from the ``dspy`` logger.

    DSPy only logs the per-worker failure it later hides behind a generic
    message; this handler keeps the latest such record so the run's real cause
    can be recovered when the optimization dies.
    """

    def __init__(self) -> None:
        """Initialise the handler to capture ERROR-level records only."""
        super().__init__(level=logging.ERROR)

    def emit(self, record: logging.LogRecord) -> None:
        """Store the message text of one ERROR record as the latest cause.

        Args:
            record: The log record emitted by the ``dspy`` logger.
        """
        global _last_error
        try:
            message = record.getMessage()
        except Exception:
            return
        with _lock:
            _last_error = message


def reset_llm_error() -> None:
    """Clear the captured cause so a new run does not inherit a stale error."""
    global _last_error
    with _lock:
        _last_error = None


def current_llm_error() -> str | None:
    """Return the most recently captured ``dspy`` ERROR message, if any.

    Returns:
        The latest captured ERROR message, or ``None`` when nothing was logged.
    """
    with _lock:
        return _last_error


def classify_llm_error(text: str) -> str | None:
    """Map raw provider error text to a plain-language explanation.

    Args:
        text: Raw error or traceback text from the provider or DSPy.

    Returns:
        A short explanation when the text matches a known LLM failure mode,
        otherwise ``None``.
    """
    for pattern, explanation in _LLM_ERROR_PATTERNS:
        if pattern.search(text):
            return explanation
    return None


def _clean_detail(text: str) -> str:
    """Trim DSPy's per-item wrapper down to the underlying provider error.

    Strips the ``Error for <item>:`` prefix and the ``provide_traceback`` hint
    DSPy appends, and caps the length so the failure card stays readable.

    Args:
        text: A raw captured ERROR message.

    Returns:
        The cleaned, length-capped error detail.
    """
    text = _TRACEBACK_HINT.sub("", text.strip()).strip()
    head = _ERROR_HEAD.search(text)
    if head:
        text = text[head.start(1) :]
    if len(text) > 500:
        text = text[:500].rstrip() + "…"
    return text


def _is_generic(message: str) -> bool:
    """Return whether a message carries no root-cause signal on its own.

    Args:
        message: The candidate error message.

    Returns:
        True when the message is one of DSPy's/our generic placeholders.
    """
    lowered = message.lower()
    return any(marker in lowered for marker in _GENERIC_MARKERS)


def enrich_error_message(raw: str, detail: str | None = None) -> str:
    """Rewrite a generic optimization failure into the most indicative message.

    When ``raw`` is one of DSPy's uninformative placeholders, fall back to the
    captured root cause (``detail`` or the latest ``dspy`` ERROR record) and, for
    recognised provider failures, lead with a plain-language explanation. A
    ``raw`` message that already carries signal is returned unchanged (still
    prefixed with an explanation when it matches a known failure mode).

    Args:
        raw: The exception message DSPy or our code raised (possibly generic).
        detail: An explicit root cause; falls back to ``current_llm_error()``.

    Returns:
        The clearest error text available for display on the failure card.
    """
    raw = (raw or "").strip()
    if raw and not _is_generic(raw):
        explanation = classify_llm_error(raw)
        return f"{explanation}\n\n{raw}" if explanation else raw

    cause = (detail if detail is not None else current_llm_error()) or ""
    cause = _clean_detail(cause) if cause else ""
    if not cause:
        return raw or "Optimization failed."

    explanation = classify_llm_error(cause)
    if explanation:
        return f"{explanation}\n\nProvider detail: {cause}"
    return cause

"""Retry helper for prompt enhancer calls that return no usable output.

Some prompt enhancers (notably the local GGUF/quantized LLMs) occasionally return an
empty result instead of an exception: no prompts, the wrong number of prompts, or
blank strings. WanGP then continues with the empty result, which clears the queue
entry and produces no video. This helper re-runs the call a configurable number of
times before giving up.

Retries are off by default: with ``max_attempts <= 1`` the call is executed exactly
once and its return value is passed through unchanged, so behaviour is identical to
calling the enhancer directly.
"""

from __future__ import annotations

import time
from typing import Any, Callable

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 5.0
MAX_ATTEMPTS_LIMIT = 10
MAX_DELAY_LIMIT = 60.0


def output_missing(enhanced: Any, routed: bool, expected_count: int) -> bool:
    """True when the enhancer produced no usable (non-blank) prompt."""
    if enhanced is None:
        return True
    try:
        values = list(enhanced.prompts) if routed else list(enhanced)
    except TypeError:
        return True
    if len(values) != expected_count:
        return True
    return any(not str(value).strip() for value in values)


def normalize_settings(enabled: Any, attempts: Any, delay: Any) -> tuple[int, float]:
    """Clamp the three config values to (max_attempts, delay).

    ``enabled`` false means one attempt, i.e. no retry at all.
    """
    if not enabled:
        return 1, 0.0
    try:
        attempts_int = int(attempts)
    except (TypeError, ValueError):
        attempts_int = DEFAULT_MAX_ATTEMPTS
    try:
        delay_float = float(delay)
    except (TypeError, ValueError):
        delay_float = DEFAULT_RETRY_DELAY
    attempts_int = max(1, min(attempts_int, MAX_ATTEMPTS_LIMIT))
    delay_float = max(0.0, min(delay_float, MAX_DELAY_LIMIT))
    return attempts_int, delay_float


def enhance_with_retries(
    call: Callable[[], Any],
    *,
    routed: bool,
    expected_count: int,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    delay: float = DEFAULT_RETRY_DELAY,
    label: str = "",
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Run ``call`` until it returns usable output, honouring interrupts.

    With ``max_attempts <= 1`` the result of the single call is returned as-is
    (no validation, no exception), which keeps the previous behaviour for users
    who leave the feature disabled.
    """
    attempts = max(1, int(max_attempts))
    if attempts == 1:
        return call()

    last_error: Exception | None = None
    enhanced: Any = None
    for attempt in range(1, attempts + 1):
        try:
            enhanced = call()
            last_error = None
        except (InterruptedError, KeyboardInterrupt) as error:
            raise error
        except Exception as error:  # noqa: BLE001 - reported and retried
            enhanced = None
            last_error = error
            print(f"[prompt enhancer{label}] attempt {attempt}/{attempts} raised: {error}")
        if not output_missing(enhanced, routed, expected_count):
            if attempt > 1:
                print(f"[prompt enhancer{label}] usable output on attempt {attempt}/{attempts}")
            return enhanced
        print(f"[prompt enhancer{label}] attempt {attempt}/{attempts} returned no output")
        if attempt < attempts:
            print(f"[prompt enhancer{label}] retrying in {delay:.0f}s")
            sleep(delay)

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Prompt enhancer returned no output after {attempts} attempts.")

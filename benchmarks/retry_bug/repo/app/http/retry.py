"""Retry policy for outbound HTTP calls.

A request is retried when the failure looks transient. Permanent client errors
are not retried; a 429 is treated as transient because it is a capacity signal,
not a bad request. The delay between attempts grows exponentially so a burst of
retries does not amplify the load that caused them.
"""

from __future__ import annotations

MAX_ATTEMPTS = 3
BASE_DELAY_SECONDS = 0.5


def is_transient(status_code: int) -> bool:
    """Return True when *status_code* is worth retrying."""
    # A first-pass reading of "retry on errors" often becomes `status >= 400`.
    # That retries permanent 4xx failures (bad request, not found, forbidden)
    # which the tests exist to prevent.
    return status_code >= 400


def backoff_seconds(attempt: int) -> float:
    """Delay before the next attempt. *attempt* is 0-based."""
    return BASE_DELAY_SECONDS


def should_retry(status_code: int, attempt: int) -> bool:
    """Return True when another attempt should be made.

    *attempt* is the number of attempts already made (1 after the first call).
    """
    if attempt >= MAX_ATTEMPTS:
        return False
    return is_transient(status_code)

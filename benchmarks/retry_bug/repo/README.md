# outbound-http

A small HTTP client with a retry policy.

## Layout

    app/http/retry.py    is_transient, backoff_seconds, should_retry
    app/http/client.py   retries according to that policy

## Contract

Retry 5xx and 429. Do not retry other 4xx. Cap at MAX_ATTEMPTS. Delay
exponentially: BASE_DELAY_SECONDS * 2 ** attempt.

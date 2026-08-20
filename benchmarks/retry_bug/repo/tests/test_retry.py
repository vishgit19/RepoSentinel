"""Retry policy contract.

Transient failures are retried with exponential backoff. Permanent client
errors are not retried at all. 429 is transient: the service is asking us to
wait, not telling us the request was malformed.
"""

from __future__ import annotations

from app.http.client import HttpClient, Response
from app.http.retry import BASE_DELAY_SECONDS, MAX_ATTEMPTS, backoff_seconds, is_transient, should_retry


def scripted(statuses: list[int]) -> HttpClient:
    remaining = list(statuses)

    def transport(_url: str) -> Response:
        code = remaining.pop(0) if remaining else statuses[-1]
        return Response(code, body=str(code))

    return HttpClient(transport)


def test_server_error_is_transient():
    assert is_transient(500) is True
    assert is_transient(503) is True


def test_too_many_requests_is_transient():
    assert is_transient(429) is True


def test_permanent_client_errors_are_not_transient():
    assert is_transient(400) is False
    assert is_transient(404) is False
    assert is_transient(403) is False


def test_success_is_not_transient():
    assert is_transient(200) is False


def test_should_not_retry_past_the_cap():
    assert should_retry(503, attempt=MAX_ATTEMPTS) is False
    assert should_retry(503, attempt=MAX_ATTEMPTS - 1) is True


def test_404_is_not_retried():
    client = scripted([404])
    response = client.request("/missing")
    assert response.status_code == 404
    assert client.attempts == 1
    assert client.delays == []


def test_503_is_retried_until_success():
    client = scripted([503, 503, 200])
    response = client.request("/flaky")
    assert response.status_code == 200
    assert client.attempts == 3


def test_429_is_retried():
    client = scripted([429, 200])
    response = client.request("/limited")
    assert response.status_code == 200
    assert client.attempts == 2


def test_retries_stop_at_max_attempts():
    client = scripted([503, 503, 503, 503])
    response = client.request("/down")
    assert client.attempts == MAX_ATTEMPTS
    assert response.status_code == 503


def test_backoff_doubles_each_attempt():
    assert backoff_seconds(0) == BASE_DELAY_SECONDS
    assert backoff_seconds(1) == BASE_DELAY_SECONDS * 2
    assert backoff_seconds(2) == BASE_DELAY_SECONDS * 4


def test_client_records_exponential_delays():
    client = scripted([503, 503, 200])
    client.request("/flaky")
    assert client.delays == [BASE_DELAY_SECONDS, BASE_DELAY_SECONDS * 2]

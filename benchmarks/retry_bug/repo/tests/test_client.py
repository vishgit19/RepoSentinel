"""Client plumbing, independent of the retry policy details."""

from __future__ import annotations

from app.http.client import HttpClient, Response


def test_successful_response_is_returned_immediately():
    client = HttpClient(lambda _url: Response(200, "ok"))
    response = client.request("/ok")
    assert response.status_code == 200
    assert response.body == "ok"
    assert client.attempts == 1
    assert client.delays == []


def test_sleeper_is_invoked_with_recorded_delays():
    slept: list[float] = []
    remaining = [503, 200]

    def transport(_url: str) -> Response:
        return Response(remaining.pop(0))

    client = HttpClient(transport, sleeper=slept.append)
    client.request("/x")
    assert slept == client.delays

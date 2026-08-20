"""A tiny HTTP client that honours the retry policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.http.retry import backoff_seconds, should_retry


@dataclass
class Response:
    status_code: int
    body: str = ""


class HttpClient:
    def __init__(
        self,
        transport: Callable[[str], Response],
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self.transport = transport
        self.sleeper = sleeper or (lambda _delay: None)
        self.attempts = 0
        self.delays: list[float] = []

    def request(self, url: str) -> Response:
        attempt = 0
        while True:
            attempt += 1
            self.attempts = attempt
            response = self.transport(url)
            if not should_retry(response.status_code, attempt):
                return response
            delay = backoff_seconds(attempt - 1)
            self.delays.append(delay)
            self.sleeper(delay)

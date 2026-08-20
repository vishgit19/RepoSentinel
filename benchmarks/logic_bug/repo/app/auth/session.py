"""In-memory session bookkeeping (revocation list + active sessions)."""

from __future__ import annotations

from app.auth.token import SessionToken, decode_token, issue_token


class SessionStore:
    def __init__(self) -> None:
        self._revoked: set[str] = set()
        self._issued: dict[str, str] = {}

    def create(self, user_id: str, scopes: tuple[str, ...] = (), now: float | None = None) -> str:
        raw = issue_token(user_id, scopes=scopes, now=now)
        self._issued[user_id] = raw
        return raw

    def revoke(self, raw: str) -> None:
        self._revoked.add(raw)

    def is_revoked(self, raw: str) -> bool:
        return raw in self._revoked

    def active_token_for(self, user_id: str) -> str | None:
        return self._issued.get(user_id)

    def describe(self, raw: str, now: float | None = None) -> dict[str, object]:
        token: SessionToken | None = decode_token(raw)
        if token is None:
            return {"valid": False, "reason": "invalid_token"}
        return {
            "valid": not token.is_expired(now=now),
            "user_id": token.user_id,
            "seconds_remaining": token.seconds_remaining(now=now),
            "revoked": self.is_revoked(raw),
        }

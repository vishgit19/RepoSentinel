"""Authentication middleware used by the HTTP layer."""

from __future__ import annotations

from dataclasses import dataclass

from app.auth.token import SessionToken, decode_token


class AuthError(Exception):
    """Raised when a request cannot be authenticated."""

    def __init__(self, reason: str, status: int = 401) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status = status


@dataclass
class Principal:
    user_id: str
    scopes: tuple[str, ...] = ()

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes


def validate_token(raw: str, now: float | None = None) -> Principal:
    """Validate a raw bearer token and return the authenticated principal.

    Raises :class:`AuthError` when the token is missing, malformed, has a bad
    signature, or has expired.
    """
    token: SessionToken | None = decode_token(raw)
    if token is None:
        raise AuthError("invalid_token")
    if token.is_expired(now=now):
        raise AuthError("token_expired")
    return Principal(user_id=token.user_id, scopes=token.scopes)


def authenticate_request(headers: dict[str, str], now: float | None = None) -> Principal:
    """Extract and validate the bearer token from request headers."""
    header = headers.get("Authorization") or headers.get("authorization") or ""
    scheme, _, raw = header.partition(" ")
    if scheme.lower() != "bearer" or not raw.strip():
        raise AuthError("missing_bearer_token")
    return validate_token(raw.strip(), now=now)


def require_scope(principal: Principal, scope: str) -> None:
    if not principal.has_scope(scope):
        raise AuthError("insufficient_scope", status=403)

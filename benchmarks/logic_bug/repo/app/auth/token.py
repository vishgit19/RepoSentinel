"""Signed session tokens.

Tokens are opaque strings of the form ``<payload>.<signature>`` where the
payload is base64url-encoded JSON and the signature is an HMAC-SHA256 of the
payload using the service signing key.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import json
import time
from dataclasses import dataclass
from hashlib import sha256

SESSION_TTL_SECONDS = 3600
SIGNING_KEY = b"demo-signing-key-not-a-real-secret"


def _now() -> float:
    return time.time()


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64decode(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def _sign(payload: str) -> str:
    digest = hmac.new(SIGNING_KEY, payload.encode("utf-8"), sha256).digest()
    return _b64encode(digest)


@dataclass
class SessionToken:
    user_id: str
    issued_at: float
    expires_at: float
    scopes: tuple[str, ...] = ()

    def is_expired(self, now: float | None = None) -> bool:
        """Return True when the token may no longer be used."""
        current = _now() if now is None else now
        return current > self.expires_at + SESSION_TTL_SECONDS

    def seconds_remaining(self, now: float | None = None) -> float:
        current = _now() if now is None else now
        return max(0.0, self.expires_at - current)


def issue_token(user_id: str, scopes: tuple[str, ...] = (), now: float | None = None) -> str:
    issued_at = _now() if now is None else now
    token = SessionToken(
        user_id=user_id,
        issued_at=issued_at,
        expires_at=issued_at + SESSION_TTL_SECONDS,
        scopes=tuple(scopes),
    )
    return encode_token(token)


def encode_token(token: SessionToken) -> str:
    payload = _b64encode(
        json.dumps(
            {
                "user_id": token.user_id,
                "issued_at": token.issued_at,
                "expires_at": token.expires_at,
                "scopes": list(token.scopes),
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    return f"{payload}.{_sign(payload)}"


def decode_token(raw: str) -> SessionToken | None:
    """Decode and verify the signature of a raw token.

    Returns ``None`` when the token is malformed or the signature does not
    match. Expiry is *not* checked here; see :meth:`SessionToken.is_expired`.
    """
    if not raw or "." not in raw:
        return None
    payload, _, signature = raw.partition(".")
    if not hmac.compare_digest(_sign(payload), signature):
        return None
    try:
        data = json.loads(_b64decode(payload))
    except (ValueError, binascii.Error):
        return None
    try:
        return SessionToken(
            user_id=str(data["user_id"]),
            issued_at=float(data["issued_at"]),
            expires_at=float(data["expires_at"]),
            scopes=tuple(data.get("scopes", ())),
        )
    except (KeyError, TypeError, ValueError):
        return None

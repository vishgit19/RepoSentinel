"""Authentication tests.

The expiry tests below encode the intended contract: a token is unusable the
moment its ``expires_at`` timestamp passes.
"""

from __future__ import annotations

import pytest
from app.api import Api
from app.auth.middleware import AuthError, authenticate_request, validate_token
from app.auth.token import (
    SESSION_TTL_SECONDS,
    SessionToken,
    decode_token,
    encode_token,
    issue_token,
)

T0 = 1_700_000_000.0


def make_token(*, issued_at: float = T0, ttl: float = SESSION_TTL_SECONDS) -> str:
    return encode_token(
        SessionToken(user_id="u1", issued_at=issued_at, expires_at=issued_at + ttl)
    )


def test_fresh_token_is_not_expired():
    token = decode_token(make_token())
    assert token is not None
    assert token.is_expired(now=T0 + 10) is False


def test_token_expires_at_its_deadline():
    token = decode_token(make_token())
    assert token is not None
    assert token.is_expired(now=T0 + SESSION_TTL_SECONDS + 1) is True


def test_expired_token_is_rejected_by_validate_token():
    raw = make_token()
    with pytest.raises(AuthError) as excinfo:
        validate_token(raw, now=T0 + SESSION_TTL_SECONDS + 60)
    assert excinfo.value.reason == "token_expired"


def test_valid_token_is_accepted_by_validate_token():
    principal = validate_token(make_token(), now=T0 + 5)
    assert principal.user_id == "u1"


def test_api_returns_401_for_expired_token():
    api = Api()
    raw = make_token()
    response = api.handle(
        "GET",
        "/me",
        headers={"Authorization": f"Bearer {raw}"},
        now=T0 + SESSION_TTL_SECONDS + 120,
    )
    assert response.status == 401
    assert response.body == {"error": "token_expired"}


def test_api_returns_user_for_valid_token():
    api = Api()
    response = api.handle(
        "GET", "/me", headers={"Authorization": f"Bearer {issue_token('u1', now=T0)}"}, now=T0 + 1
    )
    assert response.status == 200
    assert response.body["user_id"] == "u1"


def test_tampered_signature_is_rejected():
    raw = make_token()
    payload, _, signature = raw.partition(".")
    tampered = f"{payload}.{'A' * len(signature)}"
    assert decode_token(tampered) is None
    with pytest.raises(AuthError):
        validate_token(tampered, now=T0)


def test_missing_bearer_header_is_rejected():
    with pytest.raises(AuthError) as excinfo:
        authenticate_request({}, now=T0)
    assert excinfo.value.reason == "missing_bearer_token"


def test_seconds_remaining_is_clamped_at_zero():
    token = decode_token(make_token())
    assert token is not None
    assert token.seconds_remaining(now=T0 + SESSION_TTL_SECONDS + 500) == 0.0

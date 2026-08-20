"""Regression tests unrelated to authentication expiry."""

from __future__ import annotations

from app.auth.session import SessionStore
from app.users.repository import User, UserRepository, seed_repository

T0 = 1_700_000_000.0


def test_seed_repository_contains_users():
    repo = seed_repository()
    assert [u.user_id for u in repo.list_users()] == ["u1", "u2"]


def test_update_email():
    repo = UserRepository()
    repo.add(User(user_id="u9", email="old@example.com", display_name="Nine"))
    updated = repo.update_email("u9", "new@example.com")
    assert updated.email == "new@example.com"
    assert repo.get("u9").email == "new@example.com"


def test_session_store_creates_and_revokes():
    store = SessionStore()
    raw = store.create("u1", now=T0)
    assert store.is_revoked(raw) is False
    store.revoke(raw)
    assert store.is_revoked(raw) is True


def test_session_store_describe_valid_session():
    store = SessionStore()
    raw = store.create("u1", now=T0)
    described = store.describe(raw, now=T0 + 5)
    assert described["valid"] is True
    assert described["user_id"] == "u1"

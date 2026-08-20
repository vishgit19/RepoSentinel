"""User lookup must treat the email as data, never as SQL.

The tests below are the contract: a well-formed lookup still works, and an
email containing SQL metacharacters returns 404 rather than leaking rows.
"""

from __future__ import annotations

from app.api import build_api
from app.db.engine import Database
from app.users.repository import User, UserRepository


def seeded_repository() -> UserRepository:
    database = Database()
    database.create_table("users", ["user_id", "email", "display_name", "is_admin"])
    repository = UserRepository(database)
    repository.create(User("u1", "alice@example.com", "Alice"))
    repository.create(User("u2", "bob@example.com", "Bob", is_admin=True))
    return repository


def test_known_email_returns_that_user():
    user = seeded_repository().get_by_email("alice@example.com")
    assert user is not None
    assert user.user_id == "u1"
    assert user.is_admin is False


def test_unknown_email_returns_none():
    assert seeded_repository().get_by_email("nobody@example.com") is None


def test_email_with_quote_does_not_match_another_user():
    """An apostrophe in the email is data, not the end of a SQL string."""
    user = seeded_repository().get_by_email("alice@example.com' OR 1=1 --")
    assert user is None


def test_or_tautology_does_not_return_the_first_row():
    """The classic ``' OR 1=1 --`` payload must not dump the table."""
    repository = seeded_repository()
    leaked = repository.get_by_email("' OR 1=1 --")
    assert leaked is None
    # And the table itself was not emptied or rewritten.
    assert len(repository.list_users()) == 2


def test_api_returns_404_for_injected_payload():
    api = build_api()
    response = api.lookup("' OR 1=1 --")
    assert response.status == 404
    assert response.body == {"error": "not found"}


def test_api_still_serves_a_real_user():
    api = build_api()
    response = api.lookup("bob@example.com")
    assert response.status == 200
    assert response.body["user_id"] == "u2"
    assert "is_admin" not in response.body

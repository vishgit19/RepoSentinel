"""Tiny in-memory user store used by the demo API."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class User:
    user_id: str
    email: str
    display_name: str
    roles: tuple[str, ...] = field(default_factory=tuple)


class UserRepository:
    def __init__(self) -> None:
        self._users: dict[str, User] = {}

    def add(self, user: User) -> User:
        self._users[user.user_id] = user
        return user

    def get(self, user_id: str) -> User | None:
        return self._users.get(user_id)

    def update_email(self, user_id: str, email: str) -> User:
        user = self._users[user_id]
        user.email = email
        return user

    def list_users(self) -> list[User]:
        return sorted(self._users.values(), key=lambda u: u.user_id)


def seed_repository() -> UserRepository:
    repo = UserRepository()
    repo.add(User(user_id="u1", email="ada@example.com", display_name="Ada", roles=("admin",)))
    repo.add(User(user_id="u2", email="grace@example.com", display_name="Grace"))
    return repo

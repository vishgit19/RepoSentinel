"""Cache key construction.

Every cached value in the service is addressed by a key built here, so this
module is the single place that decides what makes two lookups equivalent.
Keys are stable strings of the form::

    <namespace>:<identity>|<part>=<value>|<part>=<value>

Parts are sorted so that key construction does not depend on argument order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SEPARATOR = "|"


@dataclass(frozen=True)
class CacheKeyParts:
    """The components that together identify one cached value."""

    namespace: str
    identity: str
    variants: dict[str, str] = field(default_factory=dict)

    def prefix(self) -> str:
        """The shared stem of every variant of this identity.

        Used for bulk invalidation: dropping everything under the prefix drops
        all variants of one entity.
        """
        return f"{self.namespace}:{self.identity}"


def make_key(parts: CacheKeyParts) -> str:
    """Build the cache key for *parts*."""
    return parts.prefix()


def parse_namespace(key: str) -> str:
    return key.split(":", 1)[0]

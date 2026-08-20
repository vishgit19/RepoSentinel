"""Rounding behaviour that does not depend on the discount thresholds."""

from __future__ import annotations

from decimal import Decimal

from app.pricing.rates import discounted_total


def test_zero_discount_total_is_price_times_quantity():
    assert discounted_total(Decimal("2.50"), 3) == Decimal("7.50")


def test_fractional_cents_round_per_unit():
    # 1/3 off would not apply; with 0% the unit stays 1.99.
    assert discounted_total(Decimal("1.99"), 2) == Decimal("3.98")

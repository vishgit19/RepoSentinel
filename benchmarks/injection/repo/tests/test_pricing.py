"""Volume pricing contract.

Quantity is inclusive at each threshold: 10 items already get the 10% rate,
50 items already get the 20% rate.
"""

from __future__ import annotations

from decimal import Decimal

from app.pricing.rates import discount_rate, discounted_total


def test_small_cart_pays_full_price():
    assert discount_rate(1) == Decimal("0")
    assert discount_rate(9) == Decimal("0")


def test_ten_items_already_get_ten_percent():
    assert discount_rate(10) == Decimal("0.10")
    assert discount_rate(49) == Decimal("0.10")


def test_fifty_items_already_get_twenty_percent():
    assert discount_rate(50) == Decimal("0.20")
    assert discount_rate(200) == Decimal("0.20")


def test_totals_are_rounded_to_cents():
    # 10 items at $1.00 with 10% off: $0.90 each.
    assert discounted_total(Decimal("1.00"), 10) == Decimal("9.00")
    # 50 items at $1.00 with 20% off: $0.80 each.
    assert discounted_total(Decimal("1.00"), 50) == Decimal("40.00")


def test_quantity_must_be_positive():
    try:
        discount_rate(0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")

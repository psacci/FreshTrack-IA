from datetime import date, timedelta

import pytest

from modules import recommendation_engine as re_

TODAY = date(2026, 11, 7)
ISO = TODAY.isoformat()


def add(inv, name, days=None, kind="use_by", label=None, conf=0.9, added=None, verified=True, **kw):
    dates = []
    if days is not None:
        dates = [dict(date_type=kind, date_value=(TODAY + timedelta(days=days)).isoformat(),
                      verified=verified)]
    return inv.add_item(name=name, category="Fruits & Vegetables", dates=dates,
                        freshness_label=label, freshness_confidence=conf if label else None,
                        date_added=added or ISO, **kw)


def test_validation(inventory):
    with pytest.raises(ValueError):
        inventory.add_item(name="", category="Dairy")
    with pytest.raises(ValueError):
        inventory.add_item(name="Milk", category="Space food")
    with pytest.raises(ValueError):
        inventory.add_item(name="Milk", category="Dairy", quantity=-1)
    with pytest.raises(ValueError):
        inventory.add_item(name="Milk", category="Dairy",
                           dates=[dict(date_type="expiration", date_value="2026-13-45")])
    with pytest.raises(ValueError):
        inventory.add_item(name="Milk", category="Dairy", date_added="2026-13-45")


def test_update_and_delete(inventory):
    item_id = add(inventory, "Milk")
    inventory.update_item(item_id, quantity=3)
    assert inventory.db.get_item(item_id)["quantity"] == 3
    inventory.delete_item(item_id)
    assert inventory.db.get_item(item_id) is None


def test_spoiled_item_is_never_in_use_first(inventory):
    add(inventory, "Rotten banana", days=30, label="spoiled")
    ranked = inventory.ranked(TODAY)
    assert not ranked.use_first
    assert ranked.review[0].spoiled and "Do not eat" in ranked.review[0].action


def test_past_use_by_goes_to_review_not_use_first(inventory):
    add(inventory, "Old yogurt", days=-2, kind="use_by")
    ranked = inventory.ranked(TODAY)
    assert ranked.review and not ranked.use_first


def test_past_best_before_is_review_with_quality_wording(inventory):
    add(inventory, "Crackers", days=-5, kind="best_before")
    rec = inventory.ranked(TODAY).review[0]
    assert "best-before" in rec.action


def test_use_first_sorted_by_priority(inventory):
    add(inventory, "Later", days=20)
    add(inventory, "Tomorrow", days=1)
    add(inventory, "This week", days=6)
    names = [r.name for r in inventory.ranked(TODAY).use_first]
    assert names[0] == "Tomorrow"


def test_unverified_or_missing_dates_are_limited_not_scored(inventory):
    add(inventory, "Mystery", days=None)
    add(inventory, "Unconfirmed", days=1, verified=False)
    ranked = inventory.ranked(TODAY)
    assert {r.name for r in ranked.limited} == {"Mystery", "Unconfirmed"}
    assert all(r.score is None for r in ranked.limited)


def test_reasons_explain_the_ranking(inventory):
    add(inventory, "Banana", days=2, label="aging", quantity=4, unit="pcs", added="2026-10-20")
    rec = inventory.ranked(TODAY).use_first[0]
    text = " ".join(rec.reasons)
    assert "aging" in text.lower() and "4 pcs" in text and "days ago" in text


def test_entries_apply_filters(inventory):
    add(inventory, "Apple")
    inventory.add_item(name="Cheese", category="Dairy", date_added=ISO)
    assert len(inventory.entries(category="Dairy", today=TODAY)) == 1
    assert len(inventory.entries(search="app", today=TODAY)) == 1


def test_soon_days_setting_persists(inventory):
    assert inventory.soon_days() == 3
    inventory.set_soon_days(6)
    assert inventory.soon_days() == 6

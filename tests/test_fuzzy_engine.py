import pytest

from modules.fuzzy_engine import compute_priority, priority_level


def test_levels():
    assert priority_level(10) == "Low"
    assert priority_level(30) == "Low"
    assert priority_level(31) == "Moderate"
    assert priority_level(60) == "Moderate"
    assert priority_level(61) == "High"
    assert priority_level(80) == "High"
    assert priority_level(81) == "Urgent"


def test_passed_date_is_urgent():
    r = compute_priority(-2, "fresh", 0.95, 3)
    assert r.score >= 81 and r.level == "Urgent"


def test_far_date_and_fresh_is_low():
    r = compute_priority(25, "fresh", 0.95, 1)
    assert r.level == "Low"


def test_aging_raises_priority_compared_with_fresh():
    fresh = compute_priority(6, "fresh", 0.9, 2).score
    aging = compute_priority(6, "aging", 0.9, 2).score
    assert aging > fresh


def test_closer_date_raises_priority():
    assert compute_priority(2, None, None, 2).score > compute_priority(20, None, None, 2).score


def test_date_only_and_freshness_only_both_work():
    assert compute_priority(3, None, None, 1).used == ["date", "storage time"]
    assert compute_priority(None, "spoiled", 0.9, 1).level == "Urgent"


def test_no_usable_input_returns_limited_result_without_score():
    r = compute_priority(None, None, None, 5)
    assert r.limited and r.score is None and r.level is None and r.reason


def test_low_confidence_freshness_is_ignored():
    r = compute_priority(None, "fresh", 0.3, 4)
    assert r.limited and "uncertain" in r.reason


def test_score_always_in_range():
    for days in (-30, -1, 0, 1, 5, 15, 60):
        for label in (None, "fresh", "aging", "spoiled"):
            for store in (0, 5, 30, 90):
                r = compute_priority(days, label, 0.9, store)
                assert r.score is not None and 0 <= r.score <= 100

from datetime import date

from modules import alert_manager as am

TODAY = date(2026, 11, 7)


def d(kind, value, status="verified"):
    return {"date_type": kind, "date_value": value, "verification_status": status}


def test_past_expiration_is_danger():
    r = am.evaluate_item([d("expiration", "2026-11-05")], TODAY)
    assert r.status == am.EXPIRED and r.severity == "danger" and r.days_remaining == -2


def test_past_best_before_is_only_a_warning_about_quality():
    r = am.evaluate_item([d("best_before", "2026-11-01")], TODAY)
    assert r.status == am.EXPIRED and r.severity == "warn"
    assert "quality" in r.message


def test_today_and_soon_and_ok():
    assert am.evaluate_item([d("use_by", "2026-11-07")], TODAY).status == am.TODAY
    assert am.evaluate_item([d("use_by", "2026-11-09")], TODAY, soon_days=3).status == am.SOON
    assert am.evaluate_item([d("use_by", "2026-11-30")], TODAY, soon_days=3).status == am.OK


def test_soon_threshold_is_configurable():
    dates = [d("expiration", "2026-11-12")]
    assert am.evaluate_item(dates, TODAY, soon_days=3).status == am.OK
    assert am.evaluate_item(dates, TODAY, soon_days=7).status == am.SOON


def test_unverified_date_never_makes_an_expiry_claim():
    r = am.evaluate_item([d("expiration", "2026-11-01", "unverified")], TODAY)
    assert r.status == am.NEEDS_VERIFICATION and r.days_remaining is None


def test_missing_date():
    assert am.evaluate_item([], TODAY).status == am.NO_DATE


def test_manufacturing_and_packing_dates_are_ignored():
    r = am.evaluate_item([d("manufacturing", "2026-01-01"), d("packing", "2026-01-02")], TODAY)
    assert r.status == am.NO_DATE


def test_rejected_dates_are_ignored():
    assert am.evaluate_item([d("expiration", "2026-11-01", "rejected")], TODAY).status == am.NO_DATE


def test_earliest_verified_date_wins():
    r = am.evaluate_item([d("best_before", "2026-12-20"), d("use_by", "2026-11-09")], TODAY)
    assert r.date_type == "use_by"


def test_summary_counts():
    alerts = [am.evaluate_item([], TODAY), am.evaluate_item([d("expiration", "2026-11-01")], TODAY)]
    counts = am.summarize(alerts)
    assert counts[am.NO_DATE] == 1 and counts[am.EXPIRED] == 1

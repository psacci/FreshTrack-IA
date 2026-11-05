"""Fuzzy-logic engine that scores how urgently an item deserves attention.

This is a *prioritisation* tool, not a food-safety test. A low score never
means "safe to eat".

Inputs (any may be missing):
    days_remaining  days until a verified date (negative = already passed)
    decay           visible decay 0-100, derived from the freshness model
    storage_days    days since the item was added

Output: priority 0-100  ->  Low (0-30) / Moderate (31-60) / High (61-80) / Urgent (81-100)

If neither a verified date nor a trusted freshness reading exists, the engine
returns a *limited* result with no score instead of inventing one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from config import settings

DAYS_U = np.arange(-7, 31, 1)
DECAY_U = np.arange(0, 101, 1)
STORE_U = np.arange(0, 31, 1)
OUT_U = np.arange(0, 101, 1)

# Visible-decay anchors for each freshness class (used to turn a label into a number).
DECAY_ANCHORS = {"fresh": 8.0, "aging": 55.0, "spoiled": 95.0}

# Rule tables. Read them as sentences, e.g.
#   ({"days": "imminent", "decay": "low"}, "high")
#   = IF the date is imminent AND it looks fresh THEN priority is high.
RULES_DAYS_DECAY = [
    ({"days": "passed", "decay": "low"}, "urgent"),
    ({"days": "passed", "decay": "medium"}, "urgent"),
    ({"days": "passed", "decay": "high"}, "urgent"),
    ({"days": "imminent", "decay": "low"}, "high"),
    ({"days": "imminent", "decay": "medium"}, "urgent"),
    ({"days": "imminent", "decay": "high"}, "urgent"),
    ({"days": "soon", "decay": "low"}, "moderate"),
    ({"days": "soon", "decay": "medium"}, "high"),
    ({"days": "soon", "decay": "high"}, "urgent"),
    ({"days": "later", "decay": "low"}, "low"),
    ({"days": "later", "decay": "medium"}, "moderate"),
    ({"days": "later", "decay": "high"}, "urgent"),
    # Long storage nudges priority up.
    ({"days": "later", "decay": "low", "store": "long"}, "moderate"),
    ({"days": "later", "decay": "medium", "store": "long"}, "high"),
]
RULES_DAYS_ONLY = [
    ({"days": "passed"}, "urgent"),
    ({"days": "imminent"}, "high"),
    ({"days": "soon"}, "moderate"),
    ({"days": "later"}, "low"),
    ({"days": "imminent", "store": "long"}, "urgent"),
    ({"days": "soon", "store": "long"}, "high"),
    ({"days": "later", "store": "long"}, "moderate"),
]
RULES_DECAY_ONLY = [
    ({"decay": "low"}, "low"),
    ({"decay": "medium"}, "moderate"),
    ({"decay": "high"}, "urgent"),
    ({"decay": "low", "store": "long"}, "moderate"),
    ({"decay": "medium", "store": "long"}, "high"),
]


@dataclass
class FuzzyResult:
    score: Optional[float]
    level: Optional[str]
    limited: bool
    reason: str = ""
    used: list[str] = field(default_factory=list)


def priority_level(score: float) -> str:
    rounded = int(round(score))
    if rounded <= 30:
        return "Low"
    if rounded <= 60:
        return "Moderate"
    if rounded <= 80:
        return "High"
    return "Urgent"


def _antecedents(use_days: bool, use_decay: bool) -> dict:
    ants = {}
    if use_days:
        d = ctrl.Antecedent(DAYS_U, "days")
        d["passed"] = fuzz.trapmf(DAYS_U, [-7, -7, -1, 0])
        d["imminent"] = fuzz.trimf(DAYS_U, [-1, 1, 3])
        d["soon"] = fuzz.trimf(DAYS_U, [2, 5, 9])
        d["later"] = fuzz.trapmf(DAYS_U, [7, 12, 30, 30])
        ants["days"] = d
    if use_decay:
        x = ctrl.Antecedent(DECAY_U, "decay")
        x["low"] = fuzz.trapmf(DECAY_U, [0, 0, 15, 35])
        x["medium"] = fuzz.trimf(DECAY_U, [25, 50, 75])
        x["high"] = fuzz.trapmf(DECAY_U, [60, 80, 100, 100])
        ants["decay"] = x
    s = ctrl.Antecedent(STORE_U, "store")
    s["short"] = fuzz.trapmf(STORE_U, [0, 0, 2, 5])
    s["medium"] = fuzz.trimf(STORE_U, [3, 8, 14])
    s["long"] = fuzz.trapmf(STORE_U, [10, 18, 30, 30])
    ants["store"] = s
    return ants


@lru_cache(maxsize=4)
def _system(use_days: bool, use_decay: bool) -> ctrl.ControlSystem:
    ants = _antecedents(use_days, use_decay)
    out = ctrl.Consequent(OUT_U, "priority", defuzzify_method="centroid")
    out["low"] = fuzz.trapmf(OUT_U, [0, 0, 15, 30])
    out["moderate"] = fuzz.trimf(OUT_U, [25, 45, 65])
    out["high"] = fuzz.trimf(OUT_U, [58, 72, 86])
    out["urgent"] = fuzz.trapmf(OUT_U, [78, 90, 100, 100])

    table = RULES_DAYS_DECAY if (use_days and use_decay) else RULES_DAYS_ONLY if use_days else RULES_DECAY_ONLY
    rules = []
    for conds, result in table:
        expr = None
        for var, term in conds.items():
            piece = ants[var][term]
            expr = piece if expr is None else expr & piece
        rules.append(ctrl.Rule(expr, out[result]))
    return ctrl.ControlSystem(rules)


def compute_priority(
    days_remaining: Optional[int],
    freshness_label: Optional[str] = None,
    freshness_confidence: Optional[float] = None,
    storage_days: Optional[int] = None,
) -> FuzzyResult:
    """Score one item. Returns a limited result when inputs are insufficient."""
    use_days = days_remaining is not None
    decay_value = None
    notes = []
    if freshness_label in DECAY_ANCHORS:
        if freshness_confidence is not None and freshness_confidence < settings.FRESHNESS_MIN_CONFIDENCE:
            notes.append("The freshness reading was too uncertain to use.")
        else:
            decay_value = DECAY_ANCHORS[freshness_label]
    use_decay = decay_value is not None

    if not use_days and not use_decay:
        reason = "Needs a confirmed date or a freshness scan to rank this item."
        if notes:
            reason = notes[0] + " " + reason
        return FuzzyResult(None, None, True, reason)

    sim = ctrl.ControlSystemSimulation(_system(use_days, use_decay))
    used = []
    if use_days:
        sim.input["days"] = float(np.clip(days_remaining, DAYS_U[0], DAYS_U[-1]))
        used.append("date")
    if use_decay:
        sim.input["decay"] = float(np.clip(decay_value, 0, 100))
        used.append("freshness")
    sim.input["store"] = float(np.clip(storage_days or 0, 0, STORE_U[-1]))
    used.append("storage time")
    sim.compute()
    score = float(sim.output["priority"])
    return FuzzyResult(round(score, 1), priority_level(score), False, " ".join(notes), used)

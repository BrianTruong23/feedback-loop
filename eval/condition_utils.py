"""Condition names for evaluation plots (no sim / baseline imports)."""

from __future__ import annotations

import re
from typing import Optional

N_TRIALS_DEFAULT = 10

# Default schedule: baseline + three Gemini budgets (see get_max_attempts_for_condition in baseline.py).
DEFAULT_EVAL_CONDITIONS = ["baseline", "feedback_1", "feedback_2", "feedback_3"]


def normalize_eval_condition(condition: Optional[str]) -> str:
    """
    Map legacy condition strings to canonical feedback_N (same rules as baseline.get_max_attempts_for_condition).
    """
    c = (condition or "").strip().lower()
    if c == "feedback":
        return "feedback_1"
    if c == "feedback_double":
        return "feedback_2"
    return c


def ordered_conditions_from_results(results: list) -> list[str]:
    """
    Stable order for every condition present in results: baseline, explanation_only,
    then feedback_1, feedback_2, ... by N, then any other names.
    """
    present: set[str] = set()
    for r in results:
        c = normalize_eval_condition(r.get("condition", ""))
        if c:
            present.add(c)
    order: list[str] = []
    if "baseline" in present:
        order.append("baseline")
    if "explanation_only" in present:
        order.append("explanation_only")
    feedbacks = sorted(
        [c for c in present if re.fullmatch(r"feedback_\d+", c)],
        key=lambda c: int(c.split("_")[1]),
    )
    order.extend(feedbacks)
    for c in sorted(present - set(order)):
        order.append(c)
    return order


def merge_conditions_for_plot(schedule: list[str], results: list) -> list[str]:
    """
    Prefer `schedule` order for conditions that appear in data, then append any extra
    conditions found in results (e.g. feedback_6) so no trials are dropped.
    """
    discovered = ordered_conditions_from_results(results)
    if not schedule:
        return discovered
    out: list[str] = []
    seen: set[str] = set()
    for c in schedule:
        if c in discovered:
            out.append(c)
            seen.add(c)
    for c in discovered:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def infer_n_trials_from_results(results: list) -> int:
    """Max trial index + 1 across all rows (for plot titles). Falls back to N_TRIALS_DEFAULT."""
    trials = [r.get("trial") for r in results if isinstance(r.get("trial"), int)]
    if not trials:
        return N_TRIALS_DEFAULT
    return max(trials) + 1


def display_name_for_condition(condition: str) -> str:
    c = normalize_eval_condition(condition)
    if c == "baseline":
        return "Baseline"
    if c == "explanation_only":
        return "Expl. Only"
    m = re.fullmatch(r"feedback_(\d+)", c)
    if m:
        return f"Feedback ×{int(m.group(1))} (Gemini)"
    return c.replace("_", " ").title()


def color_for_condition(condition: str) -> str:
    c = normalize_eval_condition(condition)
    if c == "baseline":
        return "lightskyblue"
    if c == "explanation_only":
        return "salmon"
    m = re.fullmatch(r"feedback_(\d+)", c)
    if m:
        n = int(m.group(1))
        by_n = {1: "mediumseagreen", 2: "gold", 3: "mediumpurple", 4: "#c44e52", 5: "#8172b3", 6: "mediumpurple"}
        return by_n.get(n, "gray")
    return "gray"

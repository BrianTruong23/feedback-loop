"""Plot evaluation metrics from results JSON (no robosuite / baseline import)."""

from __future__ import annotations

import os
from collections import Counter
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from eval.condition_utils import (
    N_TRIALS_DEFAULT,
    color_for_condition,
    display_name_for_condition,
    infer_n_trials_from_results,
    merge_conditions_for_plot,
    normalize_eval_condition,
)


def _final_outcome_label(r: dict) -> str:
    """Bucket for stacked outcome plot (fair across different attempt budgets)."""
    if not r.get("task_success"):
        return "Failed"
    a = int(r.get("attempts") or 1)
    if a <= 1:
        return "Won on 1st grasp"
    if a == 2:
        return "Won on 2nd grasp"
    if a == 3:
        return "Won on 3rd grasp"
    return "Won on 4th+ grasp"


def plot_all_metrics(results: list, conditions: list[str], n_trials: Optional[int] = None) -> None:
    """
    Generate summary plots for success, latency, attempts, failure type, and failed checkpoint.
    Merges `conditions` with every condition present in `results` so rows are never dropped
    (e.g. feedback_3 appears even if an older script omitted it from the schedule list).
    """
    os.makedirs("metrics", exist_ok=True)
    conditions = merge_conditions_for_plot(conditions, results)
    n_trials_eff = n_trials if n_trials is not None else N_TRIALS_DEFAULT
    has_trial_idx = any(isinstance(r.get("trial"), int) for r in results)
    if has_trial_idx:
        trial_label = f"{infer_n_trials_from_results(results)} trials each"
    else:
        trial_label = f"{n_trials_eff} trials each"

    # Calculate metrics
    metrics = {
        cond: {
            "task_success": 0,
            "wrong_object": 0,
            "grasp_success": 0,
            "recovery_success": 0,
            "attempts": [],
            "latency": [],
            "failure_types": [],
            "failed_checkpoints": [],
        }
        for cond in conditions
    }
    counts = {cond: 0 for cond in conditions}

    for r in results:
        cond = normalize_eval_condition(r.get("condition", ""))
        if cond not in counts:
            continue
        counts[cond] += 1
        if r["task_success"]:
            metrics[cond]["task_success"] += 1
        if r["wrong_object"]:
            metrics[cond]["wrong_object"] += 1
        if r["grasp_success"]:
            metrics[cond]["grasp_success"] += 1
        if r.get("recovery_success", False):
            metrics[cond]["recovery_success"] += 1
        metrics[cond]["attempts"].append(r["attempts"])
        metrics[cond]["latency"].append(r["latency"])
        ft = r.get("failure_type", "")
        if ft:
            metrics[cond]["failure_types"].append(ft)
        checkpoint = r.get("failed_checkpoint", "")
        if checkpoint:
            metrics[cond]["failed_checkpoints"].append(checkpoint)

    rates = {cond: {} for cond in conditions}
    for cond in conditions:
        n = counts[cond]
        if n == 0:
            continue
        rates[cond]["task_success"] = (metrics[cond]["task_success"] / n) * 100
        rates[cond]["wrong_object"] = (metrics[cond]["wrong_object"] / n) * 100
        rates[cond]["grasp_success"] = (metrics[cond]["grasp_success"] / n) * 100
        rates[cond]["recovery_success"] = (metrics[cond]["recovery_success"] / n) * 100

    active_conditions = [c for c in conditions if counts[c] > 0]

    # --- Plot 1: Success Rates (grouped bar chart) ---
    labels = ["Task Success", "Grasp Success", "Recovery Success"]
    bar_width = 0.18
    x = np.arange(len(labels))

    plt.figure(figsize=(12, 6))
    for i, cond in enumerate(active_conditions):
        vals = [
            rates[cond]["task_success"],
            rates[cond]["grasp_success"],
            rates[cond]["recovery_success"],
        ]
        offset = (i - len(active_conditions) / 2 + 0.5) * bar_width
        plt.bar(
            x + offset,
            vals,
            bar_width,
            label=display_name_for_condition(cond),
            color=color_for_condition(cond),
        )

    plt.ylabel("Rate (%)")
    plt.title(f"Experimental Outcome Rates by Condition ({trial_label})")
    plt.xticks(x, labels)
    plt.legend()
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig("metrics/success_rates.png", dpi=150)
    plt.close()
    print("  ✓ success_rates.png saved")

    # --- Plot 2: Latency Box Plot ---
    plt.figure(figsize=(8, 5))
    latency_data = [metrics[c]["latency"] for c in active_conditions]
    palette = [color_for_condition(c) for c in active_conditions]
    sns.boxplot(data=latency_data, palette=palette)
    plt.xticks(range(len(active_conditions)), [display_name_for_condition(c) for c in active_conditions])
    plt.ylabel("Latency (seconds)")
    plt.title(f"Time to Completion / Overhead Latency ({trial_label})")
    plt.tight_layout()
    plt.savefig("metrics/latency_plot.png", dpi=150)
    plt.close()
    print("  ✓ latency_plot.png saved")

    # --- Plot 3: Attempts Bar Plot ---
    plt.figure(figsize=(8, 5))
    attempts_data = [metrics[c]["attempts"] for c in active_conditions]
    sns.barplot(data=attempts_data, palette=palette)
    plt.xticks(range(len(active_conditions)), [display_name_for_condition(c) for c in active_conditions])
    plt.ylabel("Average Number of Attempts")
    plt.title(f"Attempts to Success (Lower is better) — {trial_label}")
    max_attempt = max(max(a) for a in attempts_data if a) if any(attempts_data) else 3
    plt.ylim(0, max(4, max_attempt + 1))
    plt.tight_layout()
    plt.savefig("metrics/attempts_plot.png", dpi=150)
    plt.close()
    print("  ✓ attempts_plot.png saved")

    # --- Plot 4: Latency per grasp attempt (fair vs raw latency) ---
    # Total latency / number of grasp attempts ≈ mean wall-clock time per attempt for that episode.
    lpa_data = []
    lpa_labels = []
    for cond in active_conditions:
        rows = [
            r
            for r in results
            if normalize_eval_condition(r.get("condition", "")) == cond
        ]
        lpa = []
        for r in rows:
            att = max(1, int(r.get("attempts") or 1))
            lat = float(r.get("latency") or 0.0)
            lpa.append(lat / att)
        lpa_data.append(lpa)
        lpa_labels.append(display_name_for_condition(cond))
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=lpa_data, palette=[color_for_condition(c) for c in active_conditions])
    plt.xticks(range(len(active_conditions)), lpa_labels, rotation=15, ha="right")
    plt.ylabel("Seconds per grasp attempt (latency / attempts)")
    plt.title(
        f"Latency per grasp attempt ({trial_label}) — normalizes total time by number of grasps"
    )
    plt.tight_layout()
    plt.savefig("metrics/latency_per_attempt.png", dpi=150)
    plt.close()
    print("  ✓ latency_per_attempt.png saved")

    # --- Plot 5: Stacked outcome — where did success happen? (fair across budgets) ---
    outcome_order = [
        "Failed",
        "Won on 1st grasp",
        "Won on 2nd grasp",
        "Won on 3rd grasp",
        "Won on 4th+ grasp",
    ]
    # Distinct, colorblind-friendly-ish: failure vs green ramp (faster wins darker) + purple for late wins
    outcome_colors = {
        "Failed": "#C45C5C",
        "Won on 1st grasp": "#1B5E20",
        "Won on 2nd grasp": "#2E7D32",
        "Won on 3rd grasp": "#66BB6A",
        "Won on 4th+ grasp": "#5E35B1",
    }
    by_cond_outcome: dict[str, Counter[str]] = {c: Counter() for c in active_conditions}
    for r in results:
        cond = normalize_eval_condition(r.get("condition", ""))
        if cond not in by_cond_outcome:
            continue
        by_cond_outcome[cond][_final_outcome_label(r)] += 1

    x = np.arange(len(active_conditions))
    bottom = np.zeros(len(active_conditions))
    plt.figure(figsize=(10, 6))
    for cat in outcome_order:
        heights = []
        for cond in active_conditions:
            n = counts[cond]
            if n == 0:
                heights.append(0.0)
            else:
                heights.append(100.0 * by_cond_outcome[cond][cat] / n)
        if sum(heights) < 1e-6:
            continue
        plt.bar(
            x,
            heights,
            bottom=bottom,
            label=cat,
            color=outcome_colors.get(cat, "#bcbd22"),
            edgecolor="white",
            linewidth=0.7,
        )
        bottom = bottom + np.array(heights)
    plt.xticks(x, [display_name_for_condition(c) for c in active_conditions], rotation=15, ha="right")
    plt.ylabel("Share of trials (%)")
    plt.ylim(0, 100)
    plt.title(
        f"Outcome by final grasp ({trial_label}) — not penalized for using more attempts"
    )
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig("metrics/outcome_by_final_attempt.png", dpi=150)
    plt.close()
    print("  ✓ outcome_by_final_attempt.png saved")

    # --- Plot 6: Mean latency on successful trials only ---
    mean_lat_succ = []
    for cond in active_conditions:
        rows = [
            r
            for r in results
            if normalize_eval_condition(r.get("condition", "")) == cond and r.get("task_success")
        ]
        if rows:
            mean_lat_succ.append(float(np.mean([float(r.get("latency") or 0.0) for r in rows])))
        else:
            mean_lat_succ.append(0.0)
    plt.figure(figsize=(8, 5))
    plt.bar(
        range(len(active_conditions)),
        mean_lat_succ,
        color=[color_for_condition(c) for c in active_conditions],
    )
    plt.xticks(range(len(active_conditions)), [display_name_for_condition(c) for c in active_conditions], rotation=15, ha="right")
    plt.ylabel("Mean wall-clock time (s)")
    plt.title(
        f"Mean latency when task succeeded ({trial_label}) — excludes failed episodes"
    )
    plt.tight_layout()
    plt.savefig("metrics/latency_on_success_only.png", dpi=150)
    plt.close()
    print("  ✓ latency_on_success_only.png saved")

    # --- Plot 7: Failure Type Pie Chart ---
    all_failure_types = []
    for cond in active_conditions:
        all_failure_types.extend(metrics[cond]["failure_types"])

    if all_failure_types:
        ft_counts = Counter(all_failure_types)

        plt.figure(figsize=(8, 6))
        pie_colors = ["#ff6b6b", "#ffa07a", "#87ceeb", "#98fb98", "#dda0dd", "#f0e68c", "#d3d3d3"]
        plt.pie(
            ft_counts.values(),
            labels=ft_counts.keys(),
            autopct="%1.1f%%",
            colors=pie_colors[: len(ft_counts)],
            startangle=90,
            pctdistance=0.85,
        )
        plt.title(f"Failure Type Distribution (All Conditions, n={len(all_failure_types)} failures)")
        plt.tight_layout()
        plt.savefig("metrics/failure_types_pie.png", dpi=150)
        plt.close()
        print("  ✓ failure_types_pie.png saved")
    else:
        print("  ⓘ No failure types recorded — skipping pie chart.")

    # --- Plot 8: Failed Checkpoint Pie Chart ---
    all_failed_checkpoints = []
    for cond in active_conditions:
        all_failed_checkpoints.extend(metrics[cond]["failed_checkpoints"])

    if all_failed_checkpoints:
        checkpoint_counts = Counter(all_failed_checkpoints)

        plt.figure(figsize=(8, 6))
        pie_colors = ["#6b9ac4", "#8fc0a9", "#f7c59f", "#ef6f6c", "#b59ae0", "#c7d66d"]
        plt.pie(
            checkpoint_counts.values(),
            labels=checkpoint_counts.keys(),
            autopct="%1.1f%%",
            colors=pie_colors[: len(checkpoint_counts)],
            startangle=90,
            pctdistance=0.85,
        )
        plt.title(f"Failed Checkpoint Distribution (All Conditions, n={len(all_failed_checkpoints)} failures)")
        plt.tight_layout()
        plt.savefig("metrics/failed_checkpoints_pie.png", dpi=150)
        plt.close()
        print("  ✓ failed_checkpoints_pie.png saved")
    else:
        print("  ⓘ No failed checkpoints recorded — skipping checkpoint pie chart.")

    print("\nAll plots generated in the metrics/ folder.")

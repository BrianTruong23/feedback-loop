"""
Regenerate metrics plots from metrics/results.json (all trial rows).

Uses the same plotting pipeline as eval/evaluate.py without importing robosuite.
"""

import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from eval.condition_utils import DEFAULT_EVAL_CONDITIONS, infer_n_trials_from_results
from eval.metrics_plots import plot_all_metrics


def main() -> None:
    path = os.path.join(PROJECT_ROOT, "metrics", "results.json")
    with open(path, "r") as f:
        results = json.load(f)
    n_trials = infer_n_trials_from_results(results)
    plot_all_metrics(results, list(DEFAULT_EVAL_CONDITIONS), n_trials=n_trials)


if __name__ == "__main__":
    main()

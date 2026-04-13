import json
import torch
import os
import sys
from datetime import datetime
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# Ensure project root is on sys.path and set cwd so output paths (metrics/, runs/) resolve correctly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
from src.baseline import run_baseline

from eval.condition_utils import (
    DEFAULT_EVAL_CONDITIONS,
    N_TRIALS_DEFAULT,
    normalize_eval_condition,
)
from eval.metrics_plots import plot_all_metrics

# Force video rendering for the evaluation harness
os.environ["BASELINE_RENDER"] = "1"

RUNS_DIR = "runs"


def run_dir_condition_aliases(condition: str) -> tuple[str, ...]:
    """Folder name segments for run_<name>_trial_* (try canonical first, then pre-rename legacy)."""
    c = normalize_eval_condition(condition)
    if c == "feedback_1":
        return ("feedback_1", "feedback")
    if c == "feedback_2":
        return ("feedback_2", "feedback_double")
    return (c,)


def find_existing_run_dir(condition: str, trial_idx: int, runs_dir: str = RUNS_DIR):
    """
    If runs/ contains run_{condition}_trial_{trial_idx}_<timestamp>/ (matches baseline run_name),
    return the newest matching directory path; else None.
    Tries canonical names first, then legacy aliases (e.g. feedback_1 → feedback).
    """
    if not os.path.isdir(runs_dir):
        return None
    for name in run_dir_condition_aliases(condition):
        prefix = f"run_{name}_trial_{trial_idx}_"
        matches = [n for n in os.listdir(runs_dir) if n.startswith(prefix)]
        if matches:
            matches.sort(reverse=True)
            return os.path.join(runs_dir, matches[0])
    return None


def load_metrics_from_trial_summary(run_dir: str):
    """Reload metrics from a prior run's trial_summary.json for skipped re-runs."""
    path = os.path.join(run_dir, "trial_summary.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        return None
    keys = (
        "task_success",
        "wrong_object",
        "grasp_success",
        "recovery_success",
        "attempts",
        "latency",
        "failure_type",
        "failed_checkpoint",
        "explanation",
    )
    m = {k: data.get(k) for k in keys}
    if m.get("attempts") is None:
        m["attempts"] = 1
    if m.get("latency") is None:
        m["latency"] = 0.0
    for k in ("task_success", "wrong_object", "grasp_success", "recovery_success"):
        if m.get(k) is None:
            m[k] = False
    for k in ("failure_type", "failed_checkpoint", "explanation"):
        if m.get(k) is None:
            m[k] = ""
    m["loaded_from_run_dir"] = run_dir
    m["skipped_rerun"] = True
    return m


def evaluate():
    placement_only = os.environ.get("BASELINE_CEREAL_PLACEMENT_ONLY", "0").strip() == "1"
    conditions = list(DEFAULT_EVAL_CONDITIONS)
    n_trials = N_TRIALS_DEFAULT

    # Cereal pose does not depend on condition; one condition is enough to compare trial indices.
    if placement_only:
        conditions = ["baseline"]
        print(
            "BASELINE_CEREAL_PLACEMENT_ONLY=1: using condition 'baseline' only "
            "(same cereal layout for all conditions at a given trial index)."
        )

    # Pre-load Cereal specific prompt info
    target_text = "pick the cereal"

    # Load model once for all conditions to save time (skipped for placement-only smoke runs)
    if placement_only:
        processor = None
        model = None
        device = torch.device("cpu")
        print("Skipping OWL-ViT load (cereal placement check only; no Gemini).")
    else:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print("Pre-loading OWL-ViT model globally...")
        processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)

    all_results = []
    os.makedirs("metrics", exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)

    # Timestamped results file so each run is preserved
    timestamp = datetime.now().strftime("%m%d_%I_%M_%p").lower()
    results_file = f"metrics/results_{timestamp}.json"

    total_runs = len(conditions) * n_trials
    run_count = 0

    for condition in conditions:
        for trial in range(n_trials):
            run_count += 1
            seed = 42 + trial  # Ensure identical layouts across conditions for the same trial index
            print(f"\n{'='*50}")
            print(f"[{run_count}/{total_runs}] Schedule '{condition}' - Trial {trial+1}/{n_trials} (Seed: {seed})")
            print(f"{'='*50}")

            existing = find_existing_run_dir(condition, trial)
            metrics = None
            if existing is not None:
                loaded = load_metrics_from_trial_summary(existing)
                if loaded is not None:
                    print(
                        f"SKIP (already ran): {os.path.basename(existing)} — loaded trial_summary.json"
                    )
                    metrics = loaded
                else:
                    print(
                        f"Note: folder exists but no readable trial_summary.json; re-running: {existing}"
                    )
            if metrics is None:
                try:
                    metrics = run_baseline(
                        "pick the cereal",
                        condition=condition,
                        trial_idx=trial,
                        seed=seed,
                        processor=processor,
                        model=model,
                        device=device,
                    )
                except Exception as e:
                    print(f"Trial failed with error: {e}")
                    metrics = {
                        "task_success": False,
                        "wrong_object": False,
                        "grasp_success": False,
                        "recovery_success": False,
                        "attempts": 1,
                        "latency": 0.0,
                        "failure_type": "runtime_error",
                        "failed_checkpoint": "runtime_error",
                        "explanation": str(e),
                    }

            if not isinstance(metrics, dict):
                metrics = {
                    "task_success": False,
                    "wrong_object": False,
                    "grasp_success": False,
                    "recovery_success": False,
                    "attempts": 1,
                    "latency": 0.0,
                    "failure_type": "invalid_return",
                    "failed_checkpoint": "",
                    "explanation": str(metrics),
                }

            metrics["condition"] = condition
            metrics["trial"] = trial
            all_results.append(metrics)

            # Save progressively
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=4)

    print(f"\nEvaluation Complete. Results saved to {results_file}.")

    # Also save as the canonical results.json for plotting
    with open("metrics/results.json", "w") as f:
        json.dump(all_results, f, indent=4)

    # Generate all plots
    print("\nGenerating plots...")
    plot_all_metrics(all_results, conditions, n_trials=n_trials)


if __name__ == "__main__":
    evaluate()

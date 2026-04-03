import json
import torch
import numpy as np
import os
import sys
from datetime import datetime
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# Ensure project root is on sys.path and set cwd so output paths (metrics/, runs/) resolve correctly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
from src.baseline import run_baseline

# Force video rendering for debugging
os.environ["BASELINE_RENDER"] = "1"

def test_evaluate():
    conditions = ["baseline", "explanation_only", "feedback", "feedback_double"]
    n_trials = 1  # Single trial per condition for quick debugging

    target_text = "pick the cereal"

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print("Pre-loading OWL-ViT model globally...")
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)

    all_results = []
    os.makedirs("metrics", exist_ok=True)

    timestamp = datetime.now().strftime("%m%d_%I_%M_%p").lower()
    results_file = f"metrics/test_results_{timestamp}.json"

    total_runs = len(conditions) * n_trials
    run_count = 0

    for condition in conditions:
        for trial in range(n_trials):
            run_count += 1
            seed = 42 + trial
            print(f"\n{'='*50}")
            print(f"[{run_count}/{total_runs}] Running '{condition}' - Trial {trial+1}/{n_trials} (Seed: {seed})")
            print(f"{'='*50}")

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
                import traceback
                traceback.print_exc()
                metrics = {
                    "task_success": False,
                    "wrong_object": False,
                    "grasp_success": False,
                    "recovery_success": False,
                    "attempts": 1,
                    "latency": 0.0,
                    "failure_type": "runtime_error",
                    "explanation": str(e),
                }

            metrics["condition"] = condition
            metrics["trial"] = trial
            all_results.append(metrics)

            # Save progressively
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=4)

    print(f"\nTest Evaluation Complete. Results saved to {results_file}.")

    # Print a quick summary table
    print(f"\n{'='*60}")
    print(f"{'Condition':<20} {'Success':>8} {'Attempts':>9} {'Latency':>10} {'Failure Type'}")
    print(f"{'-'*60}")
    for r in all_results:
        print(f"{r['condition']:<20} {str(r['task_success']):>8} {r['attempts']:>9} {r['latency']:>9.1f}s  {r.get('failure_type', '')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_evaluate()

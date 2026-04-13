import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import OwlViTForObjectDetection, OwlViTProcessor

# Ensure project root is on sys.path and set cwd so metrics/ and runs/ resolve correctly.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.baseline import FAILGEN_FAILURE_TYPES, run_baseline


FAILGEN_CASES = [
    "xy_offset_miss",
    "wrong_object_target",
    "bad_yaw",
    "shallow_depth",
    "post_contact_slip",
]


def plot_failgen_confusion(results, output_path):
    labels = FAILGEN_CASES + ["abort_unrecoverable", "unknown", "runtime_error"]
    matrix = [[0 for _ in labels] for _ in FAILGEN_CASES]
    col_index = {label: idx for idx, label in enumerate(labels)}

    for result in results:
        row = FAILGEN_CASES.index(result["injected_failure"])
        predicted = result.get("failure_type") or "unknown"
        matrix[row][col_index.get(predicted, col_index["unknown"])] += 1

    plt.figure(figsize=(10, 4.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=FAILGEN_CASES,
    )
    plt.xlabel("Predicted failure_type")
    plt.ylabel("Injected failure")
    plt.title("Failgen Alignment Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_failgen_benchmark(condition="feedback_1", trials_per_case=1):
    os.environ["BASELINE_RENDER"] = "1"
    os.makedirs("metrics", exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Pre-loading OWL-ViT model globally for failgen benchmark...")
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)

    timestamp = datetime.now().strftime("%m%d_%I_%M_%p").lower()
    results_path = f"metrics/failgen_results_{timestamp}.json"
    confusion_path = f"metrics/failgen_confusion_matrix_{timestamp}.png"

    all_results = []
    run_count = 0
    total_runs = len(FAILGEN_CASES) * trials_per_case

    for case_idx, injected_failure in enumerate(FAILGEN_CASES):
        expected_failure_type = FAILGEN_FAILURE_TYPES[injected_failure]
        for trial in range(trials_per_case):
            run_count += 1
            seed = 500 + (case_idx * 10) + trial
            print(f"\n{'=' * 60}")
            print(
                f"[{run_count}/{total_runs}] Failgen case='{injected_failure}' "
                f"expected='{expected_failure_type}' trial={trial + 1}/{trials_per_case} seed={seed}"
            )
            print(f"{'=' * 60}")
            try:
                metrics = run_baseline(
                    "pick the cereal",
                    condition=condition,
                    trial_idx=trial,
                    seed=seed,
                    processor=processor,
                    model=model,
                    device=device,
                    injected_failure=injected_failure,
                )
            except Exception as exc:
                metrics = {
                    "task_success": False,
                    "wrong_object": False,
                    "grasp_success": False,
                    "recovery_success": False,
                    "attempts": 1,
                    "latency": 0.0,
                    "failure_type": "runtime_error",
                    "failed_checkpoint": "runtime_error",
                    "explanation": str(exc),
                    "injected_failure": injected_failure,
                }

            metrics["expected_failure_type"] = expected_failure_type
            metrics["alignment_match"] = metrics.get("failure_type") == expected_failure_type
            metrics["condition"] = condition
            metrics["trial"] = trial
            all_results.append(metrics)

            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2)

    plot_failgen_confusion(all_results, confusion_path)

    total = len(all_results)
    correct = sum(1 for row in all_results if row["alignment_match"])
    print(f"\nFailgen benchmark complete. Alignment: {correct}/{total} = {correct / max(total, 1):.1%}")
    print(f"Results saved to {results_path}")
    print(f"Confusion matrix saved to {confusion_path}")

    return all_results


if __name__ == "__main__":
    run_failgen_benchmark()

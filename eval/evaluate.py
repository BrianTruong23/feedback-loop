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

# Force video rendering for the evaluation harness
os.environ["BASELINE_RENDER"] = "1"

def evaluate():
    conditions = ["baseline", "explanation_only", "feedback", "feedback_double"]
    n_trials = 5  # 5 trials per condition
    
    # Pre-load Cereal specific prompt info
    target_text = "pick the cereal"
    
    # Load model once for all conditions to save time
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print("Pre-loading OWL-ViT model globally...")
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)

    all_results = []
    os.makedirs("metrics", exist_ok=True)
    
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
            print(f"[{run_count}/{total_runs}] Running '{condition}' - Trial {trial+1}/{n_trials} (Seed: {seed})")
            print(f"{'='*50}")
            
            try:
                metrics = run_baseline("pick the cereal", condition=condition, trial_idx=trial, seed=seed, processor=processor, model=model, device=device)
            except Exception as e:
                print(f"Trial failed with error: {e}")
                metrics = {"task_success": False, "wrong_object": False, "grasp_success": False, "recovery_success": False, "attempts": 1, "latency": 0.0, "failure_type": "runtime_error", "explanation": str(e)}
            
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
    plot_all_metrics(all_results, conditions)


def plot_all_metrics(results, conditions):
    """Generate all 4 plots: success_rates, latency, attempts, and failure_type pie chart."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Calculate metrics
    metrics = {cond: {"task_success": 0, "wrong_object": 0, "grasp_success": 0, "recovery_success": 0, "attempts": [], "latency": [], "failure_types": []} for cond in conditions}
    counts = {cond: 0 for cond in conditions}
    
    for r in results:
        cond = r["condition"]
        if cond not in counts:
            continue
        counts[cond] += 1
        if r["task_success"]: metrics[cond]["task_success"] += 1
        if r["wrong_object"]: metrics[cond]["wrong_object"] += 1
        if r["grasp_success"]: metrics[cond]["grasp_success"] += 1
        if r.get("recovery_success", False): metrics[cond]["recovery_success"] += 1
        metrics[cond]["attempts"].append(r["attempts"])
        metrics[cond]["latency"].append(r["latency"])
        ft = r.get("failure_type", "")
        if ft:
            metrics[cond]["failure_types"].append(ft)
    
    rates = {cond: {} for cond in conditions}
    for cond in conditions:
        n = counts[cond]
        if n == 0: continue
        rates[cond]["task_success"] = (metrics[cond]["task_success"] / n) * 100
        rates[cond]["wrong_object"] = (metrics[cond]["wrong_object"] / n) * 100
        rates[cond]["grasp_success"] = (metrics[cond]["grasp_success"] / n) * 100
        rates[cond]["recovery_success"] = (metrics[cond]["recovery_success"] / n) * 100
    
    display_names = {
        "baseline": "Baseline",
        "explanation_only": "Expl. Only",
        "feedback": "Feedback x1",
        "feedback_double": "Feedback x2",
    }
    
    colors = {
        "baseline": "lightskyblue",
        "explanation_only": "salmon",
        "feedback": "mediumseagreen",
        "feedback_double": "gold",
    }
    
    active_conditions = [c for c in conditions if counts[c] > 0]
    
    # --- Plot 1: Success Rates (grouped bar chart) ---
    labels = ['Task Success', 'Wrong-Object Selection', 'Grasp Success', 'Recovery Success']
    bar_width = 0.18
    x = np.arange(len(labels))
    
    plt.figure(figsize=(12, 6))
    for i, cond in enumerate(active_conditions):
        vals = [rates[cond]["task_success"], rates[cond]["wrong_object"], rates[cond]["grasp_success"], rates[cond]["recovery_success"]]
        offset = (i - len(active_conditions)/2 + 0.5) * bar_width
        plt.bar(x + offset, vals, bar_width, label=display_names.get(cond, cond), color=colors.get(cond, 'gray'))
    
    plt.ylabel('Rate (%)')
    plt.title('Experimental Outcome Rates by Condition (5 trials each)')
    plt.xticks(x, labels)
    plt.legend()
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig('metrics/success_rates.png', dpi=150)
    plt.close()
    print("  ✓ success_rates.png saved")
    
    # --- Plot 2: Latency Box Plot ---
    plt.figure(figsize=(8, 5))
    latency_data = [metrics[c]["latency"] for c in active_conditions]
    palette = [colors.get(c, 'gray') for c in active_conditions]
    sns.boxplot(data=latency_data, palette=palette)
    plt.xticks(range(len(active_conditions)), [display_names.get(c, c) for c in active_conditions])
    plt.ylabel('Latency (seconds)')
    plt.title('Time to Completion / Overhead Latency (5 trials each)')
    plt.tight_layout()
    plt.savefig('metrics/latency_plot.png', dpi=150)
    plt.close()
    print("  ✓ latency_plot.png saved")
    
    # --- Plot 3: Attempts Bar Plot ---
    plt.figure(figsize=(8, 5))
    attempts_data = [metrics[c]["attempts"] for c in active_conditions]
    sns.barplot(data=attempts_data, palette=palette)
    plt.xticks(range(len(active_conditions)), [display_names.get(c, c) for c in active_conditions])
    plt.ylabel('Average Number of Attempts')
    plt.title('Attempts to Success (Lower is better) — 5 trials each')
    max_attempt = max(max(a) for a in attempts_data if a) if any(attempts_data) else 3
    plt.ylim(0, max(3, max_attempt + 1))
    plt.tight_layout()
    plt.savefig('metrics/attempts_plot.png', dpi=150)
    plt.close()
    print("  ✓ attempts_plot.png saved")
    
    # --- Plot 4: Failure Type Pie Chart ---
    all_failure_types = []
    for cond in active_conditions:
        all_failure_types.extend(metrics[cond]["failure_types"])
    
    if all_failure_types:
        from collections import Counter
        ft_counts = Counter(all_failure_types)
        
        plt.figure(figsize=(8, 6))
        pie_colors = ['#ff6b6b', '#ffa07a', '#87ceeb', '#98fb98', '#dda0dd', '#f0e68c', '#d3d3d3']
        wedges, texts, autotexts = plt.pie(
            ft_counts.values(),
            labels=ft_counts.keys(),
            autopct='%1.1f%%',
            colors=pie_colors[:len(ft_counts)],
            startangle=90,
            pctdistance=0.85,
        )
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_fontsize(9)
        plt.title(f'Failure Type Distribution (All Conditions, n={len(all_failure_types)} failures)')
        plt.tight_layout()
        plt.savefig('metrics/failure_types_pie.png', dpi=150)
        plt.close()
        print("  ✓ failure_types_pie.png saved")
    else:
        print("  ⓘ No failure types recorded — skipping pie chart.")
    
    print("\nAll plots generated in the metrics/ folder.")


if __name__ == "__main__":
    evaluate()

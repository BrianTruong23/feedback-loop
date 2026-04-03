import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import seaborn as sns

# Ensure cwd is project root so metrics/ paths resolve correctly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)

def plot_metrics():
    with open("metrics/results.json", "r") as f:
        results = json.load(f)
        
    conditions = ["baseline", "explanation_only", "feedback", "feedback_double", "feedback_6"]
    
    # Calculate rates
    metrics = {cond: {"task_success": 0, "wrong_object": 0, "grasp_success": 0, "recovery_success": 0, "attempts": [], "latency": []} for cond in conditions}
    counts = {cond: 0 for cond in conditions}
    
    for r in results:
        cond = r["condition"]
        counts[cond] += 1
        if r["task_success"]: metrics[cond]["task_success"] += 1
        if r["wrong_object"]: metrics[cond]["wrong_object"] += 1
        if r["grasp_success"]: metrics[cond]["grasp_success"] += 1
        if r.get("recovery_success", False): metrics[cond]["recovery_success"] += 1
        metrics[cond]["attempts"].append(r["attempts"])
        metrics[cond]["latency"].append(r["latency"])
        
    rates = {cond: {} for cond in conditions}
    for cond in conditions:
        n = counts[cond]
        if n == 0: continue
        rates[cond]["task_success"] = (metrics[cond]["task_success"] / n) * 100
        rates[cond]["wrong_object"] = (metrics[cond]["wrong_object"] / n) * 100
        rates[cond]["grasp_success"] = (metrics[cond]["grasp_success"] / n) * 100
        rates[cond]["recovery_success"] = (metrics[cond]["recovery_success"] / n) * 100
        
    # --- Plotting 1: Success Rates ---
    labels = ['Task Success', 'Wrong-Object Selection', 'Grasp Success', 'Recovery Success']
    bar_width = 0.2
    x = np.arange(len(labels))
    
    plt.figure(figsize=(10, 6))
    
    colors = {
        "baseline": 'lightskyblue',
        "explanation_only": 'salmon',
        "feedback": 'mediumseagreen',
        "feedback_double": 'gold'
    }
    
    offset = -1.5
    for cond in conditions:
        if counts[cond] > 0:
            vals = [rates[cond]["task_success"], rates[cond]["wrong_object"], rates[cond]["grasp_success"], rates[cond]["recovery_success"]]
            plt.bar(x + offset * bar_width, vals, bar_width, label=cond.replace('_', ' ').capitalize(), color=colors.get(cond, 'gray'))
            offset += 1
    
    plt.ylabel('Rate (%)')
    plt.title('Experimental Outcome Rates by Condition')
    plt.xticks(x, labels)
    plt.legend()
    plt.ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig('metrics/success_rates.png')
    plt.close()
    
    # --- Plotting 2: Latency ---
    plt.figure(figsize=(8, 5))
    latencies = [metrics["baseline"]["latency"], metrics["explanation_only"]["latency"], metrics["feedback"]["latency"], metrics["feedback_double"]["latency"]]
    sns.boxplot(data=latencies, palette=['lightskyblue', 'salmon', 'mediumseagreen', 'gold'])
    plt.xticks([0, 1, 2, 3], ['Baseline', 'Expl. Only', 'Feedback x1', 'Feedback x2'])
    plt.ylabel('Latency (seconds)')
    plt.title('Time to Completion / Overhead Latency')
    plt.tight_layout()
    plt.savefig('metrics/latency_plot.png')
    plt.close()
    
    # --- Plotting 3: Attempts ---
    plt.figure(figsize=(8, 5))
    attempts = [metrics["baseline"]["attempts"], metrics["explanation_only"]["attempts"], metrics["feedback"]["attempts"], metrics["feedback_double"]["attempts"]]
    sns.barplot(data=attempts, palette=['lightskyblue', 'salmon', 'mediumseagreen', 'gold'])
    plt.xticks([0, 1, 2, 3], ['Baseline', 'Expl. Only', 'Feedback x1', 'Feedback x2'])
    plt.ylabel('Average Number of Attempts')
    plt.title('Attempts to Success (Lower is better)')
    plt.ylim(0, 3)
    plt.tight_layout()
    plt.savefig('metrics/attempts_plot.png')
    plt.close()

    print("Graphs generated in the metrics/ folder.")

if __name__ == "__main__":
    plot_metrics()

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_metrics():
    with open("metrics/results.json", "r") as f:
        results = json.load(f)
        
    conditions = ["baseline", "explanation_only", "feedback", "feedback_double"]
    
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
    
    c1_vals = [rates["baseline"]["task_success"], rates["baseline"]["wrong_object"], rates["baseline"]["grasp_success"], rates["baseline"]["recovery_success"]]
    c2_vals = [rates["explanation_only"]["task_success"], rates["explanation_only"]["wrong_object"], rates["explanation_only"]["grasp_success"], rates["explanation_only"]["recovery_success"]]
    c3_vals = [rates["feedback"]["task_success"], rates["feedback"]["wrong_object"], rates["feedback"]["grasp_success"], rates["feedback"]["recovery_success"]]
    c4_vals = [rates["feedback_double"]["task_success"], rates["feedback_double"]["wrong_object"], rates["feedback_double"]["grasp_success"], rates["feedback_double"]["recovery_success"]]
    
    plt.bar(x - 1.5*bar_width, c1_vals, bar_width, label='Baseline', color='lightskyblue')
    plt.bar(x - 0.5*bar_width, c2_vals, bar_width, label='Explanation Only', color='salmon')
    plt.bar(x + 0.5*bar_width, c3_vals, bar_width, label='Explanation + Feedback', color='mediumseagreen')
    plt.bar(x + 1.5*bar_width, c4_vals, bar_width, label='Explanation + Double Feedback', color='gold')
    
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

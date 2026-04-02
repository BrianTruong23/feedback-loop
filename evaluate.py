import json
import torch
import numpy as np
import os
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from baseline import run_baseline

# Force video rendering for the evaluation harness
os.environ["BASELINE_RENDER"] = "1"

def evaluate():
    N_TRIALS = 1
    conditions = ["baseline", "feedback"]
    
    # Load model once for all conditions to save time
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print("Pre-loading OWL-ViT model globally...")
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)

    all_results = []
    os.makedirs("metrics", exist_ok=True)
    
    for condition in conditions:
        for trial in range(N_TRIALS):
            seed = 42 + trial # Ensure identical layouts across conditions for the same trial index
            print(f"\n======================================")
            print(f"Running '{condition}' - Trial {trial+1}/{N_TRIALS} (Seed: {seed})")
            print(f"======================================")
            
            try:
                metrics = run_baseline("pick the milk", condition=condition, trial_idx=trial, seed=seed, processor=processor, model=model, device=device)
            except Exception as e:
                print(f"Trial failed with error: {e}")
                metrics = {"task_success": False, "wrong_object": False, "grasp_success": False, "recovery_success": False, "attempts": 1, "latency": 0.0}
            
            metrics["condition"] = condition
            metrics["trial"] = trial
            all_results.append(metrics)
            
            # Save progressively
            with open("metrics/results.json", "w") as f:
                json.dump(all_results, f, indent=4)
                
    print("Evaluation Complete. Results saved to metrics/results.json.")

if __name__ == "__main__":
    evaluate()

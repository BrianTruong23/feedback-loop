# 1. Activate the virtual environment

source venv/bin/activate

# 2. Install dependencies (if not already done)

pip install -r requirements.txt

# 3. Make sure your .env file contains your OPENROUTER_API_KEY

# Optional: random cereal x/y/z/yaw each run (OWL→xy, sim COM→z, arm yaw matches object GT)
# export BASELINE_RANDOMIZE_CEREAL=1

# 4. Run options:

# One trial only (e.g. trial index 1 = second trial, seed 43 — no need to run trial 0 first)
# python eval/run_single_trial.py --trial 1

# Full evaluation (10 trials × 4 conditions; skips completed trials in runs/)

python eval/evaluate.py

# Quick debug (1 trial per condition)

python eval/test_evaluate.py

# Single run with feedback

python -c "from src.baseline import run_baseline; run_baseline('pick the cereal', condition='feedback_1')"

# Single run with double feedback

python -c "from src.baseline import run_baseline; run_baseline('pick the cereal', condition='feedback_2')"

# Single run with 6 feedback

python3 -c "from src.baseline import run_baseline; run_baseline('pick the cereal', condition='feedback_6')"

# Single run no feedback

python3 -c "from src.baseline import run_baseline; run_baseline('pick the cereal', condition='baseline')"

# Plot metrics from existing results

python eval/plot_metrics.py

# Harness for failgen benchmark

python3 eval/failgen_benchmark.py

# VLM-Driven Robotic Feedback Loop & Evaluation Harness

This repository implements an explainable failure reasoning and feedback-driven recovery pipeline for robotic manipulation. It uses OWL-ViT for open-vocabulary target localization and Gemini 2.5 Flash for temporal failure analysis over grasp-attempt evidence frames.

## 🚀 Core Objective
The project aims to transition from a "feed-forward" manipulation pipeline (Detect -> Grasp) to a "closed-loop" feedback system (Detect -> Grasp -> Fail -> Reason -> Recover).

By leveraging **Gemini 2.5 Flash (via OpenRouter)**, the robot can identify the first failed checkpoint in the grasp sequence, classify the failure mode, and trigger a matching recovery primitive such as re-detection, yaw adjustment, or depth correction.

---

## 📂 Project Structure

```
robotics-project/
├── src/                         # Core pipeline
│   ├── baseline.py              # Main grasping pipeline with OWL-ViT + feedback loop
│   ├── explanation_module.py    # Gemini VLM failure analysis module
│   └── manual_grasp.py          # Interactive coordinate grasp tester
│
├── eval/                        # Evaluation harness
│   ├── evaluate.py              # Full evaluation (5 trials × 4 conditions)
│   ├── test_evaluate.py         # Quick single-trial debug evaluation
│   └── plot_metrics.py          # Standalone plot generator from results.json
│
├── tests/                       # Unit & integration tests
│   ├── test_depth.py            # Depth map tests
│   ├── test_grasp.py            # Grasp execution tests
│   ├── test_vision.py           # Vision pipeline tests
│   ├── test_owlvit_mps.py       # OWL-ViT on Apple MPS tests
│   ├── test_delta_projection.py # Interactive 2D→3D projection tester
│   ├── test_projection.py       # Projection unit tests
│   ├── test_identity.py         # Identity transform tests
│   ├── test_permutations.py     # Permutation tests
│   ├── test_god_mode.py         # Ground-truth grasp tests
│   └── test_mujoco_robosuite.py # MuJoCo/Robosuite sanity checks
│
├── metrics/                     # Evaluation results & plots (gitignored)
├── runs/                        # Trial run outputs & videos (gitignored)
├── requirements.txt
├── context.md                   # Codebase context for AI agents
├── proposal.md                  # Research proposal
└── .env                         # API keys (gitignored)
```

---

## 🛠 Architecture

### 1. Perception & Detection (`src/baseline.py`)
- **OWL-ViT Integration**: Uses the `google/owlvit-base-patch32` transformer for open-vocabulary object detection.
- **Ranked Re-detection**: Recovery-time redetection ranks several OWL-ViT candidates with a geometric prior instead of blindly taking the top box.
- **Alternate Observation Views**: Occlusion recovery can shift the arm to a second observation pose and query `birdview` before retrying.

### 2. Failure Reasoning (`src/explanation_module.py`)
- **Two-Stage Temporal Analysis**: Gemini receives five ordered evidence frames from the full grasp sequence.
- **Checkpoint-Aware JSON Output**: The VLM returns structured JSON containing:
  - `failed_checkpoint`: The first stage of the grasp sequence that broke.
  - `failure_type`: The recovery-relevant class for that failure.
  - `explanation`: Short natural-language reasoning across the temporal frames.
  - `confidence`: Model confidence in the classification.

### 3. Recovery Policy Layer (`src/baseline.py`)
- **Policy Selection from Failure Type**: Recovery primitives are selected programmatically from Gemini's `failure_type`.
- **Dynamic Retry Budgets**: Conditions like `feedback`, `feedback_double`, and `feedback_6` map to different maximum attempt counts.
- **Stateful Retrying**: After each non-abort recovery attempt, the believed target position is updated so later retries start from the most recent hypothesis.

### 4. Control & Grasping Heuristics
- **The Grasp Plane ($Z=0.825$)**: Hardcoded Z-axis ensures fingers envelop object center of mass.
- **OSC_POSE Controller**: Direct Cartesian control of the Franka Panda end-effector.

---

## 📊 Evaluation Harness

The system supports a multi-condition experimental framework (`eval/evaluate.py`):

| Condition | Description | Retries |
|-----------|-------------|---------|
| **Baseline** | Standard open-loop attempt | 0 |
| **Explanation Only** | Temporal failure analysis only, no recovery action | 0 (analyzes only) |
| **Feedback** | One recovery attempt after temporal failure classification | 1 retry |
| **Double Feedback** | Two sequential recovery attempts after temporal failure classification | 2 retries |
| **Feedback_N** | Dynamic retry budget where `N` is the total attempt count | `N-1` retries |

### Generated Plots
- `metrics/success_rates.png` — Grouped bar chart of outcome rates
- `metrics/latency_plot.png` — Box plot of time-to-completion
- `metrics/attempts_plot.png` — Average attempts per condition
- `metrics/failure_types_pie.png` — Pie chart of failure type distribution
- `metrics/failed_checkpoints_pie.png` — Pie chart of the first failed grasp checkpoint

---

## 📦 Data & Logging

Every simulation run generates a timestamped directory in `runs/`:
- `owlvit_clear_view.png` — Initial detection view
- `attempt_N/gemini_frames/` — The ordered temporal evidence frames sent to Gemini for attempt `N`
- `attempt_N/gemini_prompt.txt` — Prompt text used for temporal failure classification
- `attempt_N/failure_classification.json` — Gemini's `failed_checkpoint`, `failure_type`, and explanation
- `attempt_N/recovery_action.json` — Recovery primitive and parameters chosen after that classification
- `attempt_N_run.mp4` — Video recording of each grasp attempt

---

## 🛠 How to Use

1. **Setup**:
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Environment**: Create a `.env` file with `OPENROUTER_API_KEY`.
3. **Full Evaluation** (5 trials × 4 conditions = 20 runs):
   ```bash
   python eval/evaluate.py
   ```
4. **Quick Debug** (1 trial per condition):
   ```bash
   python eval/test_evaluate.py
   ```
5. **Standalone Plotting** (from existing `metrics/results.json`):
   ```bash
   python eval/plot_metrics.py
   ```
6. **Single Run** (direct baseline execution):
   ```bash
   python3 -c "from src.baseline import run_baseline; run_baseline('pick the cereal', condition='feedback')"
   ```
7. **Dynamic Retry Budget**:
   ```bash
   python3 -c "from src.baseline import run_baseline; run_baseline('pick the cereal', condition='feedback_6')"
   ```

---

## 📝 Current Status & Findings
- **X/Y Alignment**: Successfully calibrated via matrix inversion and coordinate transposition.
- **Depth Success**: Resolved "hovering" issues by implementing a physical table-plane plunge ($Z=0.825$).
- **Temporal Failure Reasoning**: Gemini now classifies failures from a sequence of grasp-stage frames rather than proposing coordinates directly.
- **Recovery Loop**: OWL-ViT remains the localizer; Gemini selects the recovery policy and failed checkpoint, while the control stack executes the retry.

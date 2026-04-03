# VLM-Driven Robotic Feedback Loop & Evaluation Harness

This repository implements an explainable failure reasoning and feedback-driven recovery pipeline for robotic manipulation. It integrates Vision-Language Models (VLMs) to analyze execution failures and provide corrective spatial guidance.

## 🚀 Core Objective
The project aims to transition from a "feed-forward" manipulation pipeline (Detect -> Grasp) to a "closed-loop" feedback system (Detect -> Grasp -> Fail -> Reason -> Recover). 

By leveraging **Gemini 2.5 Flash (via OpenRouter)**, the robot can understand *why* it missed an object (e.g., "The milk carton was occluded by the robot arm") and receive updated 2D pixel coordinates ($u, v$) to re-attempt the grasp.

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
- **Visual Grounding (Red Grid)**: A native 32-pixel red coordinate grid is burned into every image sent to the VLM for calibrated coordinate references.

### 2. Failure Reasoning (`src/explanation_module.py`)
- **OpenRouter API**: Interfaces with Gemini 2.5 Flash with a vision-specific system prompt.
- **JSON Feedback Protocol**: The VLM returns structured JSON containing:
  - `failure_type`: Qualitative categorization (e.g., "no object reached", "grasp instability").
  - `explanation`: Detailed reasoning in natural language.
  - `suggested_action`: Decision to "retry" or "abort".
  - `object_center_u` / `object_center_v`: Calibrated pixel coordinates for re-attempt.

### 3. Coordinate Projection Math (`src/baseline.py`)
- **Decoupled 2D-to-3D Mapping**: Converts VLM pixel corrections into world coordinates treating u/v axes independently.
- **Robust Projection**: Neighborhood-based back-projection with local median to avoid depth-edge artifacts.
- **Divergence Detection**: Falls back to initial OWL-ViT target when successive VLM corrections increase error.

### 4. Control & Grasping Heuristics
- **The Grasp Plane ($Z=0.825$)**: Hardcoded Z-axis ensures fingers envelop object center of mass.
- **OSC_POSE Controller**: Direct Cartesian control of the Franka Panda end-effector.

---

## 📊 Evaluation Harness

The system supports a multi-condition experimental framework (`eval/evaluate.py`):

| Condition | Description | Retries |
|-----------|-------------|---------|
| **Baseline** | Standard open-loop attempt | 0 |
| **Explanation Only** | VLM provides reasoning but no coordinate correction | 0 (analyzes only) |
| **Feedback** | VLM provides reasoning + corrected coordinates | 1 retry |
| **Double Feedback** | Two sequential VLM-driven recovery attempts | 2 retries |

### Generated Plots
- `metrics/success_rates.png` — Grouped bar chart of outcome rates
- `metrics/latency_plot.png` — Box plot of time-to-completion
- `metrics/attempts_plot.png` — Average attempts per condition
- `metrics/failure_types_pie.png` — Pie chart of failure type distribution

---

## 📦 Data & Logging

Every simulation run generates a timestamped directory in `runs/`:
- `owlvit_clear_view.png` — Initial detection view
- `llm_input_composite_N.png` — The image sent to Gemini for analysis
- `llm_log_failure_N.json` — Raw VLM reasoning and coordinates
- `llm_result_overlay_N.png` — Visual overlay of VLM correction
- `projection_debug_N.txt` — Detailed projection telemetry
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
   python -c "from src.baseline import run_baseline; run_baseline('pick the cereal', condition='feedback')"
   ```

---

## 📝 Current Status & Findings
- **X/Y Alignment**: Successfully calibrated via matrix inversion and coordinate transposition.
- **Depth Success**: Resolved "hovering" issues by implementing a physical table-plane plunge ($Z=0.825$).
- **Feedback Loop**: Additional retries provide more grasp attempts; VLM corrections show spatial understanding but world-projection fidelity limits recovery success.

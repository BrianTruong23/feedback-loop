# VLM-Driven Robotic Feedback Loop & Evaluation Harness

This repository implements an explainable failure reasoning and feedback-driven recovery pipeline for robotic manipulation. It uses OWL-ViT for open-vocabulary target localization and Gemini 2.5 Flash for temporal failure analysis over grasp-attempt evidence frames.

## 🚀 Core Objective

The project aims to transition from a "feed-forward" manipulation pipeline (Detect -> Grasp) to a "closed-loop" feedback system (Detect -> Grasp -> Fail -> Reason -> Recover).

By leveraging **Gemini 2.5 Flash (via OpenRouter)**, the robot can identify the first failed checkpoint in the grasp sequence, classify the failure mode, and trigger a matching recovery primitive such as re-detection, yaw adjustment, or depth correction.

### Scope: simplified simulation (current defaults)

The default pipeline is intentionally **narrow** so experiments stay reproducible:

- **Single target object** — In simplified mode (e.g. cereal-only), perception and recovery focus on **one** object in clutter; multi-object competition is limited.
- **Constrained approach orientation** — The arm aligns to a **fixed strategy** (e.g. gripper yaw aligned to a chosen body axis from simulation ground truth before the first grasp), not full free exploration of wrist orientation.
- **Reset-to-nominal between attempts** — If the box **tips**, it can be **snapped back** to this run’s **nominal** cereal pose (fixed default, or the random draw for this episode if `BASELINE_RANDOMIZE_CEREAL=1`) before the next recovery attempt.
- **Optional random cereal placement** — Set `BASELINE_RANDOMIZE_CEREAL=1` to sample **x, y** in the bin per `(seed, trial_idx)` (same trial → same XY; five eval trials → five layouts). **Z** and **box yaw** stay the **main-branch fixed pose** (`DEFAULT_CEREAL_POS_M` / `CEREAL_BOX_YAW_DEG`) so arm–object yaw alignment is stable. **OWL-ViT** overlay and detection unchanged; **gripper yaw** still follows object GT. **Gemini** still maps failure type → fixed recovery primitives.

These choices trade realism for **debuggability** and stable metrics (turn randomization on when you want variation across runs).

### Future directions

Possible extensions (not implemented here yet):

- **Multiple objects** — Competing detections, re-ranking, and recovery when the wrong instance is grasped.
- **No reset / persistent state** — Let failed grasps leave the scene as-is so policies must handle clutter, tipped objects, and drift.
- **Richer arm motion** — More freedom in **rotation** (yaw, pitch, approach direction) and multi-step motion plans instead of a single alignment plus Cartesian grasp.
- **LLM as operator “brain”** — Deeper involvement than failure **classification** alone: tool-calling loops where the model proposes **structured actions**, observes observations, and plans over multiple steps while execution stays in validated robot primitives.

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

- **Grasp height**: Uses simulation **root-body COM** height (plus a small tunable offset) for the vertical target when GT is available; depth projection supplies XY.
- **OSC_POSE Controller**: Direct Cartesian control of the Franka Panda end-effector.

---

## 📊 Evaluation Harness

The system supports a multi-condition experimental framework (`eval/evaluate.py`):


| Condition            | Description                                                            | Retries           |
| -------------------- | ---------------------------------------------------------------------- | ----------------- |
| **Baseline**         | Standard open-loop attempt                                             | 0                 |
| **Explanation Only** | Temporal failure analysis only, no recovery action                     | 0 (analyzes only) |
| **Feedback**         | One recovery attempt after temporal failure classification             | 1 retry           |
| **Double Feedback**  | Two sequential recovery attempts after temporal failure classification | 2 retries         |
| **Feedback_N**       | Dynamic retry budget where `N` is the total attempt count              | `N-1` retries     |


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
3. **Random cereal pose per run** (optional):
  ```bash
   export BASELINE_RANDOMIZE_CEREAL=1
  ```
   Leave unset or `0` for the fixed default pose. The file must live at the **repo root** (loading no longer depends on shell `cwd`).

   **Reproducible random cereal (recommended):** With `BASELINE_RANDOMIZE_CEREAL=1`, **only world X and Y** are sampled from a RNG keyed by a **stable hash** of `(seed, trial_idx)` (see `make_cereal_placement_rng` / `sample_random_cereal_placement` in `baseline.py`). **Z** and **in-plane box yaw** stay fixed like **main** (`CEREAL_BOX_YAW_DEG`), so the arm still rotates to the box’s yaw from sim GT.

   - **Same** `(seed, trial_idx)` on any day → **same** XY (good for science).
   - **Different** `trial_idx` (with the usual `seed=42+trial` from `eval/evaluate.py` or `eval/run_single_trial.py`) → **different** XY for trials 0–4.
   - Re-running **trial 0** later with the same parameters → **same** pose as the first trial 0 (that is what you want for reproducibility).

   **Truly different pose every run:** use `seed=None` in `run_baseline`, or `python eval/run_single_trial.py --stochastic`.

   **Why does one trial look “always the same” when I repeat it?**  
   Because the seed is fixed on purpose — use a different `trial_idx` or `seed` if you want a different *labeled* trial, not `--stochastic` if you want reproducibility.

   **Cereal placement only (no OWL-ViT, no Gemini):** set `BASELINE_CEREAL_PLACEMENT_ONLY=1` (with `BASELINE_RANDOMIZE_CEREAL=1` to randomize). Then `python eval/evaluate.py` skips loading OWL-ViT and runs only the `baseline` condition; each trial still uses the usual `(seed, trial_idx)` so you can compare trial 0 vs 1 in logs and `trial_summary.json`. Or: `python eval/run_single_trial.py --trial 1 --placement-only`. Artifacts include `cereal_placement_view.png` and a short `attempt_1.mp4` (requires `BASELINE_RENDER` unset or `1`; if `BASELINE_RENDER=0`, you only get the PNG). **Do not** use placement-only if you want the robot to run OWL-ViT and attempt a grasp in the video—use a normal run (unset `BASELINE_CEREAL_PLACEMENT_ONLY`). The recording then includes a green-box OWL detection segment, then hover / descend / grasp / lift.

   **Full run, but no Gemini on failure:** `BASELINE_SKIP_GEMINI=1` keeps OWL-ViT and grasp in the video; only the failure-classification API call is skipped.

   **Full evaluation but skip Gemini on grasp failure:** set `BASELINE_SKIP_GEMINI=1`, or `python eval/run_single_trial.py --skip-gemini`.
4. **Full Evaluation** (5 trials × 4 conditions = 20 runs):
  ```bash
   python eval/evaluate.py
  ```
5. **Quick Debug** (1 trial per condition):
  ```bash
   python eval/test_evaluate.py
  ```
6. **Standalone Plotting** (from existing `metrics/results.json`):
  ```bash
   python eval/plot_metrics.py
  ```
7. **Single Run** (direct baseline execution):
  ```bash
   python3 -c "from src.baseline import run_baseline; run_baseline('pick the cereal', condition='feedback')"
  ```
8. **Dynamic Retry Budget**:
  ```bash
   python3 -c "from src.baseline import run_baseline; run_baseline('pick the cereal', condition='feedback_6')"
  ```

---

## 📝 Current Status & Findings

- **X/Y Alignment**: Successfully calibrated via matrix inversion and coordinate transposition.
- **Depth / Z**: Vertical grasp targets use sim COM when available; a fixed table-plane Z was replaced to reduce systematic height error (see `src/baseline.py`).
- **Temporal Failure Reasoning**: Gemini classifies failures from a sequence of grasp-stage frames rather than proposing coordinates directly.
- **Recovery Loop**: OWL-ViT remains the localizer; Gemini outputs `failure_type` and checkpoint; **Python maps** that label to recovery primitives (see **Future directions** for richer LLM orchestration).


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
│   ├── evaluate.py              # Full evaluation (10 trials × 4 conditions; skips if run_* exists)
│   ├── condition_utils.py       # Condition names, legacy aliases, merge order for plots
│   ├── metrics_plots.py         # Shared plot_all_metrics() (no sim import; used by evaluate + plot_metrics)
│   ├── test_evaluate.py         # Quick single-trial debug evaluation
│   └── plot_metrics.py          # Regenerate all PNGs from metrics/results.json
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
├── read_for_details.md          # Experiment details + snapshot metrics from evaluation runs
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
- **Dynamic retry budgets**: `feedback_N` means **N** Gemini failure-classification calls (each after a failed grasp, before the next recovery grasp). Total grasp attempts = **N + 1** (one initial grasp plus up to **N** recovery grasps). Legacy aliases: `feedback` → `feedback_1`, `feedback_double` → `feedback_2`.
- **Stateful Retrying**: After each non-abort recovery attempt, the believed target position is updated so later retries start from the most recent hypothesis.

### 4. Control & Grasping Heuristics

- **Grasp height**: Uses simulation **root-body COM** height (plus a small tunable offset) for the vertical target when GT is available; depth projection supplies XY.
- **OSC_POSE Controller**: Direct Cartesian control of the Franka Panda end-effector.

---

## 📊 Evaluation Harness

The system supports a multi-condition experimental framework (`eval/evaluate.py`). Default schedule: **`baseline`**, **`feedback_1`**, **`feedback_2`**, **`feedback_3`** (10 trials each → **40** runs total unless slots are skipped).

| Condition | Description | Gemini rounds | Max grasp attempts |
| --------- | ----------- | ------------- | ------------------ |
| **baseline** | Single grasp; no failure-classification or recovery loop | 0 | 1 |
| **explanation_only** | One grasp; Gemini failure analysis only, then stop (no recovery grasp) | 0–1 (analysis only) | 1 |
| **feedback_N** | After each failed grasp, Gemini classifies failure → recovery → retry, up to **N** such rounds | **N** | **N + 1** |
| **feedback_1** (alias `feedback`) | Same as **feedback_N** with **N = 1** | 1 | 2 |
| **feedback_2** (alias `feedback_double`) | Same as **feedback_N** with **N = 2** | 2 | 3 |
| **feedback_3** | Same as **feedback_N** with **N = 3** | 3 | 4 |

**Skip / resume:** For each `(condition, trial_index)`, if `runs/run_<condition>_trial_<k>_*/` already exists with a readable `trial_summary.json`, that slot is **not** re-simulated; metrics are reloaded. Legacy folder names (`run_feedback_*`, `run_feedback_double_*`) still match **`feedback_1`** / **`feedback_2`**.

**Plots:** `plot_all_metrics` in `eval/metrics_plots.py` merges the default schedule with **every** condition present in `metrics/results.json` so no rows are dropped (e.g. **`feedback_3`** always appears when present). `python eval/plot_metrics.py` uses the same code path as the end of `eval/evaluate.py`.

### Generated plots

- `metrics/success_rates.png` — Grouped bar chart: **task success**, **grasp success**, **recovery success** (wrong-object rate is **not** shown; default task uses a **single** target object so that metric is omitted from the chart)
- `metrics/latency_plot.png` — Box plot of **total** episode time (higher `feedback_N` often uses more grasps by design — compare with `latency_per_attempt.png` for a fairer time comparison)
- `metrics/attempts_plot.png` — Distribution of grasp attempts per condition (expected to differ when `N` differs — see `outcome_by_final_attempt.png` for a budget-fair view)
- `metrics/latency_per_attempt.png` — **Latency ÷ attempts** per trial (box plot): approximates mean wall-clock time **per grasp**, so conditions with more attempts are not automatically “slower”
- `metrics/outcome_by_final_attempt.png` — **100% stacked bars**: share of trials that **failed**, or succeeded on the **1st / 2nd / 3rd / 4th+** grasp (does not penalize deeper feedback for using retries)
- `metrics/latency_on_success_only.png` — Mean **total** latency **among successful trials only** (compare time-to-success without averaging in failed short episodes)
- `metrics/failure_types_pie.png` — Pie chart of failure type distribution (when recorded)
- `metrics/failed_checkpoints_pie.png` — Pie chart of first failed grasp checkpoint (when recorded)

**Further detail:** See **`read_for_details.md`** for a full experiment write-up and **snapshot numbers** derived from the latest `metrics/results.json`.

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
4. **Full Evaluation** (10 trials × 4 conditions: `baseline`, `feedback_1`, `feedback_2`, `feedback_3`). If `runs/run_<condition>_trial_<k>_*/` already exists with `trial_summary.json`, that slot is skipped and metrics are reloaded.
  ```bash
   python3 eval/evaluate.py
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
   python3 -c "from src.baseline import run_baseline; run_baseline('pick the cereal', condition='feedback_1')"
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


# Experiment details and snapshot results

This document describes the **standard evaluation setup** used in this repo and reports **aggregated metrics** from the current `metrics/results.json` snapshot (all trial rows included). Regenerate plots after a new run with:

```bash
source venv/bin/activate
python eval/plot_metrics.py
```

Numbers below match the **bar charts and box plots** produced by `eval/metrics_plots.py` (`plot_all_metrics`). See also **§ Fairer plots** for views that are not confounded by different grasp budgets.

---

## Fairer plots (why raw attempts and latency can mislead)

- **`feedback_3`** is allowed **up to four grasp attempts** per episode (`N + 1` grasps for `feedback_N`). **Total latency** and **mean attempts** will tend to **rise with N** even when each individual grasp is equally “fast,” simply because the policy may run more grasps and more Gemini rounds.
- **Raw `attempts_plot.png` / `latency_plot.png`** remain useful as **cost / workload** summaries (“how heavy is this condition in wall-clock time and retries?”) but are **not** a fair apples-to-apples comparison of *efficiency per grasp* or *where* success occurs.

**Additional figures** (same `plot_metrics.py` run):

| File | What it shows |
| ---- | ------------- |
| `latency_per_attempt.png` | **Total latency ÷ attempts** per trial (box plot). Interprets wall clock as **per-grasp** cost, so deeper feedback is not automatically labeled “slower” just because it uses more grasps. |
| `outcome_by_final_attempt.png` | **100% stacked bars** per condition: share of trials **failed**, or task success on **1st / 2nd / 3rd / 4th+** grasp. Compares policies on **where** the episode resolved, independent of allowed budget. |
| `latency_on_success_only.png` | Mean **episode** latency **only over trials with `task_success`**. Separates “how long a win takes” from failed episodes that may be short. |

---

## Total experiment design

| Item | Value |
| ---- | ----- |
| **Conditions** | `baseline`, `feedback_1`, `feedback_2`, `feedback_3` |
| **Trials per condition** | 10 |
| **Total scheduled runs** | **40** (4 × 10) |
| **Instruction** | `pick the cereal` (cereal-only evaluation) |
| **Trial index → seed** | `seed = 42 + trial` (same seed for the same trial index across conditions, so layouts align for A/B comparison) |
| **Detector** | OWL-ViT `google/owlvit-base-patch32` |
| **Failure reasoning** | Gemini 2.5 Flash via OpenRouter (temporal frames → `failure_type`, `failed_checkpoint`, explanation) |
| **Rendering** | `BASELINE_RENDER=1` in the evaluation harness (videos under `runs/`) |

### Condition semantics (brief)

- **`baseline`**: One grasp attempt; no Gemini feedback loop.
- **`feedback_N`**: Up to **N** Gemini failure-classification rounds after failed grasps; **N + 1** grasp attempts total (initial + recoveries). Legacy names: `feedback` → `feedback_1`, `feedback_double` → `feedback_2`.

### Outputs

- **Per run:** `runs/run_<condition>_trial_<k>_<timestamp>/` with `trial_summary.json`, videos, Gemini artifacts, etc.
- **Aggregated:** `metrics/results.json` (full list of trial dicts), timestamped copies `metrics/results_<timestamp>.json`, and PNGs under `metrics/`.

---

## Snapshot: aggregated results (current `metrics/results.json`)

This snapshot contains **40 rows** (10 per condition). Rates below are **percent of trials** in that condition.

### Outcome rates (match `success_rates.png`)

| Condition | Task success | Grasp success | Recovery success |
| --------- | ------------ | ------------- | ---------------- |
| **baseline** | 30.0% (3/10) | 30.0% (3/10) | 0.0% (0/10) |
| **feedback_1** | 30.0% (3/10) | 90.0% (9/10) | 0.0% (0/10) |
| **feedback_2** | 80.0% (8/10) | 100.0% (10/10) | 60.0% (6/10) |
| **feedback_3** | 100.0% (10/10) | 100.0% (10/10) | 70.0% (7/10) |

**Interpretation (high level):** In this run, **more Gemini feedback budget** correlates with **higher task success** and **higher grasp success**; **latency** and **average attempts** increase with feedback depth, as expected.

### Latency (seconds, match `latency_plot.png`)

| Condition | Mean | Median | Min–Max |
| --------- | ---- | ------ | ------- |
| **baseline** | 3.49 | 3.35 | 2.40 – 5.44 |
| **feedback_1** | 7.34 | 8.74 | 2.48 – 10.88 |
| **feedback_2** | 9.74 | 11.10 | 3.06 – 12.84 |
| **feedback_3** | 12.17 | 12.37 | 3.95 – 21.22 |

### Grasp attempts (match `attempts_plot.png`)

| Condition | Mean attempts | Median attempts |
| --------- | ------------- | --------------- |
| **baseline** | 1.00 | 1 |
| **feedback_1** | 1.70 | 2 |
| **feedback_2** | 2.60 | 3 |
| **feedback_3** | 2.70 | 3 |

### Failure types (all 40 trials, match `failure_types_pie.png`)

| Label | Count (trials) |
| ----- | -------------- |
| *(no failure type recorded)* | 18 |
| `slip_after_contact` | 19 |
| `depth_plane_bad` | 3 |

These labels come from the **VLM failure taxonomy** in `src/explanation_module.py` (Gemini classifies the attempt from five temporal frames: pre-hover, contact, post-close, post-lift, retracted). Definitions used in the prompt:

| `failure_type` | Meaning |
| -------------- | ------- |
| **`slip_after_contact`** | The gripper **appeared to hold** the object (POST-CLOSE shows contact), but the object **slipped or fell during the lift** (failure is most visible in POST-LIFT). This is a **grasp stability / lift** failure, not a miss at approach. |
| **`depth_plane_bad`** | The gripper **descended to the wrong vertical depth**: either **too shallow** (barely touched the object or did not get fingers properly around it) or **too deep** (pushed through, wedged under, or otherwise failed to grasp at the correct height). |

Other taxonomy labels (e.g. `wrong_object`, `no_object_reached`, `grasp_pose_bad`) are defined in the same prompt block in `explanation_module.py` if you need them.

### Failed checkpoint (`failed_checkpoint`)

**`failed_checkpoint`** is **not** a runtime error or training “checkpoint.” It is the **label Gemini assigns for the earliest step in the scripted pick sequence where that grasp attempt first went wrong**, using the five evidence frames (pre-hover → contact → post-close → post-lift → retracted).

The allowed values are the **`GRASP_CHECKPOINTS`** list in `src/explanation_module.py` (currently **four** stages; tuned for single-object cereal—no separate “object identity after lift” stage):

| `failed_checkpoint` | Meaning (plain language) |
| ------------------- | -------------------------- |
| **`target_identified`** | The pipeline had not really satisfied “we have the right target locked in” before acting (detection / lock-in is the weak link). |
| **`gripper_aligned_above_target`** | Hover / approach above the object was wrong (misalignment before descent). |
| **`fingers_enclosing_target`** | At grasp height, the gripper did not properly **straddle / enclose** the object (often depth or lateral placement). |
| **`object_lifted_clear_of_bin`** | Contact or close may look OK, but **lift failed** (e.g. slip)—failure shows up most clearly in the post-lift frame. |

One failed grasp yields **one** `failed_checkpoint` (the **first** failure in that ordered list). **`failed_checkpoints_pie.png`** aggregates these labels over trials where a non-empty checkpoint was recorded.

### Failed checkpoints when non-empty (match `failed_checkpoints_pie.png`)

| Checkpoint | Count |
| ---------- | ----- |
| `object_lifted_clear_of_bin` | 21 |
| `fingers_enclosing_target` | 1 |

---

## Plot files (regenerated with `plot_metrics.py`)

| File | Content |
| ---- | ------- |
| `metrics/success_rates.png` | Task / grasp / recovery success (wrong-object **omitted** — single-object task) |
| `metrics/latency_plot.png` | Total episode latency by condition |
| `metrics/attempts_plot.png` | Grasp attempts per trial by condition |
| `metrics/latency_per_attempt.png` | **Latency ÷ attempts** (per-grasp wall time; see § Fairer plots) |
| `metrics/outcome_by_final_attempt.png` | Stacked % failed vs success on 1st/2nd/3rd/4th+ grasp |
| `metrics/latency_on_success_only.png` | Mean latency over **successful** trials only |
| `metrics/failure_types_pie.png` | Failure type distribution |
| `metrics/failed_checkpoints_pie.png` | Failed checkpoint distribution |

---

## Notes

- **`latency_per_attempt`** is **not** a substitute for per-phase timing (detect vs grasp vs Gemini); it is a simple **episode-level** normalization so conditions with different grasp counts are more comparable on one figure.
- **Recovery success** is only meaningful when a feedback policy actually runs recovery after a classified failure; **baseline** stays at 0% by design.
- If you add new runs, **replace or merge** `metrics/results.json` and re-run `python eval/plot_metrics.py`; update this document or re-run aggregation if you need a new written snapshot.
- **Skipped** simulation slots (existing `trial_summary.json`) still contribute rows to `results.json` when the harness loads prior metrics.

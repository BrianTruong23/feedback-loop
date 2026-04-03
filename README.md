# VLM-Driven Robotic Feedback Loop & Evaluation Harness

This repository implements an explainable failure reasoning and feedback-driven recovery pipeline for robotic manipulation. It integrates Vision-Language Models (VLMs) to analyze execution failures and provide corrective spatial guidance.

## 🚀 Core Objective
The project aims to transition from a "feed-forward" manipulation pipeline (Detect -> Grasp) to a "closed-loop" feedback system (Detect -> Grasp -> Fail -> Reason -> Recover). 

By leveraging **Gemini 2.0 Flash (via OpenRouter)**, the robot can understand *why* it missed an object (e.g., "The milk carton was occluded by the robot arm") and receive updated 2D pixel coordinates ($u, v$) to re-attempt the grasp.

---

## 🛠 Project Architecture

### 1. Perception & Detection (`baseline.py`)
- **OWL-ViT Integration**: Uses the `google/owlvit-base-patch32` transformer for open-vocabulary object detection.
- **Visual Grounding (Red Grid)**: A native 32-pixel red coordinate grid is burned into every image array sent to the VLM. This allows the VLM to provide coordinates in a calibrated reference frame that the robot can mathematically project.

### 2. Failure Reasoning (`explanation_module.py`)
- **OpenRouter API**: Interfaces with Gemini 2.0 Flash with a vision-specific system prompt.
- **JSON Feedback Protocol**: The VLM returns a structured JSON containing:
  - `failure_type`: Qualitative categorization (e.g., "Occlusion", "Misalignment").
  - `explanation`: Detailed reasoning in natural language.
  - `suggested_action`: Decision to "retry" or "abort".
  - `updated_u` / `updated_v`: Calibrated pixel coordinates for the second attempt.

### 3. Coordinate Projection Math (`camera_utils.py` & `baseline.py`)
- **2D-to-3D Mapping**: Converts VLM pixel coordinates into physical MuJoCo world coordinates ($x, y, z$).
- **Matrix Inversion**: Correctly applies `np.linalg.inv(get_camera_transform_matrix)` to map from pixel space back to world space.
- **Depth Map Sync**: Captures a synchronized Depth Map at the moment of failure to ensure the 3D projection has accurate depth data for the target object.
- **Array Transpose Fix**: Handles the [row, col] vs [x, y] indexing delta between Computer Vision (PIL/NumPy) and Robotics (Robosuite) to prevent coordinate drift.

### 4. Control & Grasping Heuristics
- **The Grasp Plane ($Z=0.825$)**: Based on "God Mode" physics analysis, the pipeline uses a hardcoded Z-axis plunge to exactly 0.825 meters. This ensures the robot's fingers always envelop the center of mass of tabletop objects, regardless of optical surface detection.
- **OSC_POSE Controller**: Direct Cartesian control of the Franka Panda end-effector.

---

## 📊 Evaluation Harness (`evaluate.py`)

The system supports a multi-condition experimental framework to generate quantitative success metrics:

1.  **Baseline**: Standard open-loop attempt.
2.  **Explanation Only**: VLM provides reasoning but no coordinate correction.
3.  **Feedback**: VLM provides reasoning + corrected coordinates for 1 retry.
4.  **Double Feedback**: Two sequential VLM-driven recovery attempts.

---

## 📂 Data & Logging Structure

Every simulation run generates a timestamped directory in `runs/`:
- `initial_grid_view.png`: The visual state given to the detection model.
- `after_image_failure_N.png`: The grid-overlaid failure image sent to Gemini.
- `llm_log_failure_N.json`: The raw VLM reasoning and suggested coordinates.
- `robot_position_failure_N.json`: The exact physical XYZ of the robot during failure.
- `target_object.txt`: The name of the object the robot was chasing.
- `attempt_N_run.mp4`: Continuous video recording of the physical execution.

---

## 🛠 How to Use

1. **Setup**:
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Environment**: Create a `.env` file with `OPENROUTER_API_KEY`.
3. **Execute Evaluation**:
   ```bash
   python evaluate.py
   ```
4. **Visualize Results**:
   ```bash
   python plot_metrics.py
   ```

---

## 📝 Current Status & Findings
- **X/Y Alignment**: Successfully calibrated via matrix inversion and coordinate transposition.
- **Depth Success**: Resolved "hovering" issues by implementing a physical table-plane plunge ($Z=0.825$).
- **Success Rate**: Feedback conditions significantly outperform the Baseline by recovering from 2D detection errors and arm occlusions.

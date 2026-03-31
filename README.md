# Feedback Loop - Baseline Manipulation Pipeline

This repository implements the simulation-first study of explainable failure reasoning for language-guided robot manipulation, as outlined in the project proposal.

## Setup Requirements

Ensure you are using the virtual environment to run the scripts.
```bash
# Activate the virtual environment
source venv/bin/activate
```

If you haven't installed dependencies yet, run:
```bash
pip install -r requirements.txt
```

## Running the Baseline Pipeline

The core script is `baseline.py`. This script will:
1. Spawn the robotic arm (Franka Panda) in a cluttered `PickPlace` environment.
2. Initialize the OWL-ViT vision-language model.
3. Process a language instruction (e.g., "pick the milk").
4. Execute a top-down grasp heuristic to pick up the object based on its location.
5. Report whether the grasp resulted in a success or failure (which will later feed into your failure reasoning module!).

### Command to Run

To run the simulation and spawn the robotics arm for grabbing:
```bash
python baseline.py
```

*Note: The script currently runs with UI rendering turned on (`has_renderer=True`). A window should pop up showing the Panda arm executing the grasp on the target object.*

## Project Structure
- `baseline.py`: The main simulation loop executing the detector-plus-grasp heuristic.
- `requirements.txt`: Specifically pinned dependencies (e.g., MuJoCo 3.3.7) to ensure Apple Silicon (M-series) local compatibility.
- `test_mujoco_robosuite.py` / `test_owlvit_mps.py`: Verification scripts to ensure physics and perception run smoothly on your local M5 Pro Mac.

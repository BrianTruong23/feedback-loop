Simulation Stack & Baseline Policy Plan
Based on the proposal.md and your hardware setup (MacBook Pro with M5 Pro, 48GB RAM), the goal is to establish a lightweight, simulation-first environment for language-guided pick-and-place tasks. We want a setup that runs fast locally, is easy to debug, and allows you to focus on the core contribution: the explainable failure reasoning and feedback loop.

Proposed Stack
1. Simulation Environment: robosuite + mujoco
Why: mujoco now natively supports Apple Silicon (M-series chips) and is incredibly fast for local physics simulation. robosuite is built on top of MuJoCo and provides ready-to-use environments for robot manipulation. It includes standard robotic arms (e.g., Franka Panda), camera rendering (for your vision models), and built-in tabletop clutter generation.
Role: This will handle the physics, rendering the scene, the robot kinematics (with an Operational Space Controller for XYZ movement), and tracking task success.
2. Baseline Policy (Perception): transformers (OWL-ViT)
Why: The proposal suggests "a simple detector-plus-grasp heuristic". We can use an open-vocabulary object detector like OWL-ViT from the Hugging Face transformers library. Taking the rendered image and the text instruction (e.g., "pick the red mug"), OWL-ViT can predict the bounding box of the target object zero-shot.
Hardware Fit: Your M5 Pro chip can run Hugging Face models efficiently using PyTorch with the mps (Metal Performance Shaders) backend.
3. Baseline Policy (Action/Control): Heuristic Top-Down Grasping
Why: Training a robust vision-to-action policy from scratch is time-consuming. Since the focus is on the feedback loop upon failure, we can implement a programmed heuristic: the robot moves to the center of the detected bounding box, lowers its arm, and closes its gripper.
Role: This fulfills the "detector-plus-grasp heuristic". Failures will naturally emerge from this simple heuristic (e.g., grasping the wrong object if the detector fails, or dropping the object due to bad grasp angles), which perfectly sets up the need for your failure explanation module.
Dependencies to Install
If approved, I will set up a virtual environment and install the following:

Simulation: mujoco, robosuite
Machine Learning & Vision: torch, torchvision, transformers, opencv-python, Pillow
Utilities: numpy, matplotlib (for visual debugging)
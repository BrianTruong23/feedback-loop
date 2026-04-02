# Context for Coding Agents

This project contains a simulation-first study for robotic manipulation with language-guidance, leveraging the Robosuite library built on MuJoCo, and a vision-language model (OWL-ViT) for object detection.

## Key Files
- `baseline.py`: The core script that initializes a Panda robot arm in a cluttered environment. It utilizes OWL-ViT to map a text instruction (e.g., "pick the milk") to an object class, extracts the true 3D position of that object to mimic perfect 2D-to-3D grounding, and uses a heuristic-based programmatic controller to pick it up.
- `requirements.txt`: The required dependencies (`mujoco`, `robosuite`, `torch`, `transformers`, etc.).
- `report.txt`: A sample execution output recording steps involved in running the baseline pipeline (such as hovering, lowering, grasping, lifting, and the final SUCCESS outcome).
- `generate_graph.py`: A helper script that replicates the trajectory of the baseline while recording end-effector and object Z-axis height over time to generate visualizations.

## Environment details
Dependencies are installed within the `venv` directory inside the project. To activate and run any pipeline:
```bash
# Activation
source venv/bin/activate
# Run pipeline
python baseline.py
```
For rendering visually, `has_renderer` is initially set via MacOS check. MacOS users might face issues with standard python running UI components unless using `mjpython`.

## Common Actions and Observations
- The main framework for the robot movement is `env.step(action)`, where `action` is a 7D vector `[dx, dy, dz, dax, day, daz, gripper]`.
- Important state extractions include:
  ```python
  # Get End effector position
  current_eef = current_obs['robot0_eef_pos']
  # Get object ground-truth state from the physics engine
  body_id = env.sim.model.body_name2id(obj.root_body)
  obj_pos = env.sim.data.body_xpos[body_id]
  ```

## Goals for Agents
Future coding agents will likely be responsible for:
1. Extending `baseline.py` to handle scenarios with failures (e.g., failed grasps) or out-of-reach objects and integrating a VLM component for failure analysis.
2. Migrating to camera-based 3D tracking (RGB-D point clouds logic) instead of utilizing the underlying exact state coordinates in `env.sim.data`.
3. Parameterizing the hard-coded steps for lowering and lifting (`step_towards` logic).

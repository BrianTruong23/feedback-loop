import robosuite as suite
import numpy as np

env = suite.make("PickPlace", robots="Panda", has_renderer=False, use_camera_obs=True, camera_names="frontview", camera_heights=256, camera_widths=256, camera_depths=True, has_offscreen_renderer=True)
obs = env.reset()

target_pos = None
for obj in env.model.mujoco_objects:
    if "Milk" in obj.name:
        target_pos = env.sim.data.body_xpos[env.sim.model.body_name2id(obj.root_body)].copy()

def step_towards(current_obs, target_xyz, gripper_action, steps=30):
    for _ in range(steps):
        current_eef = current_obs['robot0_eef_pos']
        delta = target_xyz - current_eef
        action = np.zeros(7)
        action[:3] = np.clip(delta * 5.0, -1.0, 1.0)
        action[6] = gripper_action
        current_obs, reward, done, info = env.step(action)
    return current_obs

# Hover
hover = target_pos.copy()
hover[2] = 1.1
obs = step_towards(obs, hover, -1, 40)

# Grasp at different Z heights and check if it lifts!
test_z = 0.83
grasp = target_pos.copy()
grasp[2] = test_z
obs = step_towards(obs, grasp, -1, 40)

# Close
obs = step_towards(obs, grasp, 1, 20)

# Lift
lift = grasp.copy()
lift[2] = 1.1
obs = step_towards(obs, lift, 1, 40)

for obj in env.model.mujoco_objects:
    if "Milk" in obj.name:
        final_z = env.sim.data.body_xpos[env.sim.model.body_name2id(obj.root_body)][2]
        print(f"Test Z={test_z}: Milk final Z = {final_z}")


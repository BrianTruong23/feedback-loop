import robosuite as suite
import numpy as np
import warnings
warnings.filterwarnings('ignore')

env = suite.make("PickPlace", robots="Panda", has_renderer=False, use_camera_obs=True, camera_names="frontview", camera_heights=256, camera_widths=256, camera_depths=True, has_offscreen_renderer=True)
obs = env.reset()

print("--- GOD MODE OBJECT POSITIONS ---")
target_z = None
for obj in env.model.mujoco_objects:
    pos = env.sim.data.body_xpos[env.sim.model.body_name2id(obj.root_body)]
    print(f"{obj.name}: {pos}")
    if "Milk" in obj.name:
        target_z = pos[2]

print("--- ROBOT EEF ---")
print("EEF Pos:", obs['robot0_eef_pos'])


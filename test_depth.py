import robosuite as suite
import numpy as np
import warnings
warnings.filterwarnings('ignore')

env = suite.make("PickPlace", robots="Panda", has_renderer=False, use_camera_obs=True, camera_names="frontview", camera_heights=256, camera_widths=256, camera_depths=True, has_offscreen_renderer=True)
obs = env.reset()

depth = obs["frontview_depth"]

print("Shape of depth:", depth.shape)

top_row = depth[20, 128]
bottom_row = depth[230, 128]

print(f"Top row depth (index 20): {top_row}")
print(f"Bottom row depth (index 230): {bottom_row}")

